---
id: "04.05"
course: "ml-toolkit"
module: "04"
slug: "data-pipelines-dataset-and-dataloader"
title: "Data pipelines: Dataset and DataLoader"
keyIdeas: ["A map-style `Dataset` answers two questions, `__len__` and `__getitem__(i)`; an `IterableDataset` yields a stream. Keep `__getitem__` cheap and per-example; batching belongs to the `DataLoader`.", "The `DataLoader` combines a **sampler** (which indices, in what order), a **collate function** (how a list of examples becomes one batch) and **workers** (separate processes that run `__getitem__` in parallel).", "`default_collate` stacks tensors of equal shape and recurses into dicts and tuples. Variable-length data (text, detection boxes) needs a custom `collate_fn` that pads and returns masks.", "With `num_workers > 0` each worker holds its own copy of the dataset object and its own RNG. An `IterableDataset` must shard itself with `get_worker_info()`, or every worker yields the same data.", "If GPU utilisation is low and the loss loop is simple, time the loader alone. `num_workers`, `pin_memory=True`, `persistent_workers=True` and doing decode work once offline are the usual fixes."]
question: "You launch the same script on 4 GPUs. Each rank builds `DataLoader(ds, batch_size=32, num_workers=8)` over an `IterableDataset` that streams JSON lines from one big file. An epoch takes 32 times as many optimiser steps as you expected, and the training loss drops unusually fast. What is happening, and how do you fix it?"
answer: "Nothing shards the stream. Each of the 4 ranks opens the whole file, and within each rank the 8 workers each iterate their own copy of the dataset from the first line, so every line is read once per worker per rank: 32 copies per epoch. You get 32 times the expected steps, and every batch is repeated 32 times, so the model memorises the data and the training loss falls faster than real progress justifies. Samplers cannot help: `DataLoader` rejects a sampler such as `DistributedSampler` for an `IterableDataset`. Shard inside `__iter__` instead: combine the rank and world size from `torch.distributed` with `get_worker_info().id` and `num_workers` into a global reader index from 0 to 31, and have each reader yield only its own lines (line number modulo 32 equals its index), or split the file into shard files and give each reader its own set."
sources: [{"title": "PyTorch docs: torch.utils.data", "url": "https://pytorch.org/docs/stable/data.html", "note": "Map-style vs iterable datasets, samplers, collate, worker sharding and the randomness notes."}, {"title": "PyTorch tutorial: Datasets and DataLoaders", "url": "https://pytorch.org/tutorials/beginner/basics/data_tutorial.html", "note": "A minimal custom Dataset and how DataLoader iterates it."}, {"title": "PyTorch performance tuning guide", "url": "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html", "note": "num_workers, pinned memory and other input-pipeline settings that affect throughput."}]
practice: ["custom-dataset-class", "balanced-dataloader", "transform-pipeline", "padding-and-truncation"]
tags: ["DataLoader", "collate", "workers"]
tools: ["PyTorch"]
---

## Two objects with separate jobs

PyTorch splits input handling into a `Dataset`, which knows how to fetch one example, and a `DataLoader`, which decides order, groups examples into batches and parallelises the fetching. Keeping them separate means the same dataset can be shuffled for training, read in order for evaluation, or sharded across GPUs with a `DistributedSampler`, without changing its code.

```python
class Windows(torch.utils.data.Dataset):
    # Sliding windows over one long series, for forecasting.
    def __init__(self, series, ctx, horizon):
        self.x = torch.as_tensor(series, dtype=torch.float32)  # one tensor, not a list
        self.ctx, self.h = ctx, horizon
    def __len__(self):
        return len(self.x) - self.ctx - self.h + 1
    def __getitem__(self, i):
        return self.x[i:i + self.ctx], self.x[i + self.ctx:i + self.ctx + self.h]

loader = torch.utils.data.DataLoader(
    Windows(series, ctx=96, horizon=24), batch_size=64, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
```

`__getitem__` should return tensors (or NumPy arrays and numbers, which collate converts) for **one** example. Doing heavy decoding there is fine, because workers run it in parallel; doing global work there (re-reading a whole file per item) is not.

~~~dg-flow What happens to produce one batch.
Sampler -(indices)-> Workers -(dataset[i] for each i)-> collate_fn -(stack or pad)-> *pinned batch -(non_blocking copy)-> GPU
~~~

## Samplers and order

- `shuffle=True` is shorthand for a `RandomSampler`, which draws a fresh permutation every epoch. `shuffle=False` uses a `SequentialSampler`.
- `WeightedRandomSampler(weights, num_samples)` samples with replacement in proportion to per-example weights, a simple fix for class imbalance; a custom `batch_sampler` can instead build each batch with a fixed number of examples per class.
- `DistributedSampler` gives each rank a disjoint slice of a shared permutation. Call `sampler.set_epoch(epoch)` every epoch, or every epoch uses the same order.
- `drop_last=True` discards a final short batch, which keeps batch-norm statistics and compiled shapes stable.

## Collate: from a list of examples to a batch

The default collate function takes a list of `batch_size` examples and stacks matching fields: a list of `(x, y)` pairs becomes `(stack(xs), stack(ys))`, dicts are collated per key, and Python numbers become tensors. Stacking requires equal shapes, so variable-length inputs fail with `stack expects each tensor to be equal size`. Write your own:

```python
def pad_collate(batch, pad_id=0):
    seqs = [torch.as_tensor(ids) for ids, _ in batch]
    lens = torch.tensor([len(s) for s in seqs])
    x = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    mask = torch.arange(x.size(1))[None, :] < lens[:, None]     # True on real tokens
    y = torch.tensor([label for _, label in batch])
    return {"input_ids": x, "attention_mask": mask, "labels": y}
```

Padding to the longest sequence in the batch wastes compute when lengths vary widely. Bucketing (grouping similar lengths into the same batch) or packing several documents into one fixed-length row with a block-diagonal attention mask, which is the standard for LLM pretraining, cuts the waste.

## Workers, memory and randomness

With `num_workers=k`, the loader starts `k` processes, each holding a copy of the dataset object, and each prepares whole batches (`prefetch_factor` batches ahead, default 2). Three consequences:

- **Copy-on-access memory growth.** Under the `fork` start method, workers share the parent's memory pages until they are written. Python reference counting writes to every object it touches, so a dataset holding a large Python list of strings or dicts gets copied page by page into every worker. Store data in NumPy arrays, tensors or memory-mapped files instead.
- **Randomness.** PyTorch seeds each worker's `torch` and `random` generators from a base seed plus the worker id, but other libraries' generators can come out identical across workers. Seed NumPy yourself in a `worker_init_fn` using `torch.utils.data.get_worker_info().seed`, or use torch's RNG for augmentation.
- **IterableDataset sharding.** Each worker iterates its own copy, so with 4 workers you get every example 4 times. Inside `__iter__`, read `get_worker_info()` and yield only every `num_workers`-th shard.

`persistent_workers=True` keeps workers alive between epochs instead of restarting them (and re-pickling the dataset). `pin_memory=True` places batches in page-locked host memory so that `batch.to("cuda", non_blocking=True)` can overlap the copy with compute.

## Is the loader the bottleneck?

| symptom | likely cause | first fix
| --- | --- | --- |
| GPU utilisation saw-tooths, steps wait on `next(it)` | too few workers or slow decode | raise `num_workers` (start near CPU cores per GPU), move decoding offline
| first batch of each epoch is slow | workers restarting | `persistent_workers=True`
| host RAM grows each epoch with workers | Python objects copied into workers | NumPy or mmap storage
| fast loader, slow step | not a data problem | profile the model instead

Measure it directly: iterate the loader with no model for a few hundred batches and compare batches per second with the training loop's steps per second.

<div class="ex">
<div class="lab">Worked example</div>

**how many workers?**

A vision model trains at 20 steps per second with batches of 256 images, so it needs 5,120 decoded and augmented images per second.

One worker decodes and augments about 400 images per second, so it takes 13 workers just to keep up; with 8 CPU cores per GPU that is impossible.

Options: decode once and store preprocessed tensors, use GPU-side decoding and augmentation, or reduce per-image work (smaller resize). Adding workers past the core count only adds context switching.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a Dataset returns one example, the DataLoader decides order, batching and parallelism. Collate variable-length data yourself, store data so workers do not copy it, and measure the loader before blaming the model.

</div>
