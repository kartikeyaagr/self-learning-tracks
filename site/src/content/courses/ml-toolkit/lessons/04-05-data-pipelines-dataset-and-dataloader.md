---
id: "04.05"
course: "ml-toolkit"
module: "04"
slug: "data-pipelines-dataset-and-dataloader"
title: "Data pipelines: Dataset and DataLoader"
keyIdeas: ["A map-style `Dataset` answers two questions, `__len__` and `__getitem__(i)`; an `IterableDataset` yields a stream. Keep `__getitem__` cheap and per-example: batching belongs to the `DataLoader`.", "The `DataLoader` combines a **sampler** (which indices, in what order), a **collate function** (how a list of examples becomes one batch) and **workers** (separate processes that run `__getitem__` in parallel). One epoch has $\\lceil N/B \\rceil$ batches, or $\\lfloor N/B \\rfloor$ with `drop_last=True`.", "`default_collate` stacks tensors of equal shape and recurses into dicts and tuples. Variable-length data (text, detection boxes) needs a custom `collate_fn` that pads and returns a mask; bucketing similar lengths keeps padding waste small.", "With `num_workers > 0` each worker holds its own copy of the dataset object. An `IterableDataset` must shard itself with `get_worker_info()`, or every worker yields the same data, and random generators created in `__init__` must be reseeded per worker.", "If training is slow, time the loader alone. `num_workers`, `pin_memory=True`, `persistent_workers=True`, array or memory-mapped storage, and doing decode work once offline are the usual fixes."]
question: "You launch the same script on 4 GPUs. Each rank builds `DataLoader(ds, batch_size=32, num_workers=8)` over an `IterableDataset` that streams JSON lines from one big file. An epoch takes 32 times as many optimiser steps as you expected, and the training loss drops unusually fast. What is happening, and how do you fix it?"
answer: "Nothing shards the stream. Each of the 4 ranks opens the whole file, and within each rank the 8 workers each iterate their own copy of the dataset from the first line, so every line is read once per worker per rank: $4 \\times 8 = 32$ copies per epoch. You get 32 times the expected steps, and every example is repeated 32 times, so the model memorises the data and the training loss falls faster than real progress justifies (validation loss will not follow). Samplers cannot help: `DataLoader` rejects a sampler such as `DistributedSampler` for an `IterableDataset`. Shard inside `__iter__` instead: read the rank and world size from `torch.distributed` and the worker id and count from `get_worker_info()`, combine them into a global reader index $g = \\text{rank} \\times 8 + \\text{worker id}$ from 0 to 31, and have each reader yield only lines whose line number modulo 32 equals $g$. Better still, split the file into many shard files and assign whole files to readers, so no reader has to scan lines it then discards. Afterwards verify the fix by counting examples per epoch across all ranks: the total should equal the number of lines, and the step count should drop by a factor of 32."
sources: [{"title": "PyTorch docs: torch.utils.data", "url": "https://pytorch.org/docs/stable/data.html", "note": "Map-style vs iterable datasets, samplers, collate, worker sharding and the randomness notes."}, {"title": "PyTorch tutorial: Datasets and DataLoaders", "url": "https://pytorch.org/tutorials/beginner/basics/data_tutorial.html", "note": "A minimal custom Dataset and how DataLoader iterates it."}, {"title": "PyTorch performance tuning guide", "url": "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html", "note": "num_workers, pinned memory and other input-pipeline settings that affect throughput."}]
practice: ["custom-dataset-class", "balanced-dataloader", "transform-pipeline", "padding-and-truncation"]
tags: ["DataLoader", "collate", "workers"]
tools: ["PyTorch"]
---

## What you will learn

- What a `Dataset` is (an object that hands you one example by index) and what a `DataLoader` adds on top (order, batching, parallel fetching).
- How to write your own map-style dataset in about ten lines, and how to count how many batches it will produce.
- How a list of examples becomes one batch (the **collate** step), and how to handle examples of different lengths with padding and masks.
- How samplers control order: shuffling, class rebalancing, and splitting data across GPUs.
- How worker processes speed loading up, and the three classic worker bugs: duplicated streams, identical random numbers, and memory that keeps growing.
- How to measure whether the input pipeline, not the model, is what slows training down.

*Before you start:* you should be comfortable with tensors and shapes (04.01) and with a model that maps a batch to a loss (04.03, 04.04). Lesson 04.06 builds the full training loop on top of the loaders made here.

Lesson 04.04 computed a loss on one batch and quietly assumed the batch already existed. Real data lives in files, has examples of different sizes, and is often too slow to read on the main thread. This lesson covers `torch.utils.data`, the part of PyTorch that turns "a folder of data" into "a steady stream of well-shaped batches".

## The idea in one picture

Take the smallest possible dataset: ten examples, where example $i$ is the pair $(i, i^2)$. We ask PyTorch for batches of four.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class Squares(Dataset):
    def __init__(self, n):
        self.x = torch.arange(n, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.x[i] ** 2

ds = Squares(10)
print(len(ds), ds[3])            # 10 (tensor(3.), tensor(9.))

g = torch.Generator().manual_seed(0)
for xb, yb in DataLoader(ds, batch_size=4, shuffle=True, generator=g):
    print(xb.tolist(), yb.tolist())
# [3.0, 7.0, 5.0, 2.0] [9.0, 49.0, 25.0, 4.0]
# [0.0, 8.0, 1.0, 6.0] [0.0, 64.0, 1.0, 36.0]
# [9.0, 4.0] [81.0, 16.0]
```

~~~dg-fig 04.05-batches.svg
The same ten indices cut into batches of four. Without shuffling the order is fixed; with shuffling every epoch gets a fresh order; drop_last throws away the short final batch.
~~~

The dataset never shuffles or batches anything: it only answers "how many examples?" and "give me example 3". The loader drew a random order (3, 7, 5, 2, 0, ...), fetched each example, and glued four at a time into one tensor of inputs and one of targets. Ten is not a multiple of four, so the last batch has two examples. Next epoch the order differs.

Three words carry the whole lesson:

- A **dataset** knows how to produce one example, given its position.
- A **sampler** decides which positions to ask for, and in what order.
- A **collate function** turns a list of examples into one batch.

The **DataLoader** is the machine that wires these three together, optionally with several **worker** processes doing the fetching in parallel.

~~~dg-flow What happens to produce one batch.
Sampler -(indices)-> Workers -(dataset[i] for each i)-> collate_fn -(stack or pad)-> *pinned batch -(non_blocking copy)-> GPU
~~~

Because the jobs are separate, the same dataset can be shuffled for training, read in order for evaluation, rebalanced, or split across GPUs without touching its code.

## Step 1: write a map-style Dataset

A **map-style dataset** is anything with two methods: `__len__` returns the number of examples, and `__getitem__(i)` returns example number `i` for any `i` from 0 to `len - 1`. It behaves like a read-only list.

A realistic case is forecasting: from one long series, cut many pairs of a past window (the **context**) and the stretch that follows it (the **horizon**).

```python
class Windows(Dataset):
    # Sliding windows over one long series, for forecasting.
    def __init__(self, series, ctx, horizon):
        self.x = torch.as_tensor(series, dtype=torch.float32)  # one tensor, not a list
        self.ctx, self.h = ctx, horizon
    def __len__(self):
        return len(self.x) - self.ctx - self.h + 1
    def __getitem__(self, i):
        return self.x[i:i + self.ctx], self.x[i + self.ctx:i + self.ctx + self.h]

w = Windows(torch.sin(torch.arange(1000) / 20.), ctx=96, horizon=24)
print(len(w), w[0][0].shape, w[0][1].shape)     # 881 torch.Size([96]) torch.Size([24])
```

The length formula counts how many start positions $i$ leave room for a full window. With a series of length $T$, a context of $c$ and a horizon of $h$, the last valid start is $T - c - h$, so there are

$$N = T - c - h + 1$$

windows. Here $T$ is the series length, $c$ the number of past steps the model sees, $h$ the number of future steps it must predict.

Two rules make a dataset fast and safe:

- `__getitem__` returns **one** example (tensors, arrays or numbers); batching is the loader's job.
- Per-example work (decoding one image) belongs in `__getitem__`; global work (parsing a whole file, building a vocabulary) belongs in `__init__` or an offline step. A `__getitem__` that re-reads a big file per example is the classic slow pipeline.

For tensors already in memory, `TensorDataset(X, y)` needs no class, and `random_split` cuts a dataset into pieces:

```python
from torch.utils.data import TensorDataset, random_split
X = torch.randn(1000, 8); y = torch.randint(0, 2, (1000,))
ds = TensorDataset(X, y)
train, val = random_split(ds, [800, 200], generator=torch.Generator().manual_seed(0))
print(len(train), len(val), type(train).__name__)   # 800 200 Subset
```

A `Subset` is the original dataset plus a list of indices, so splitting copies no data.

## Step 2: let the DataLoader batch it

Wrap the dataset in a loader and iterate. The loader's length is the number of batches per epoch:

$$\text{batches} = \left\lceil \frac{N}{B} \right\rceil \quad\text{or, with drop\_last,}\quad \left\lfloor \frac{N}{B} \right\rfloor,$$

where $N$ is the number of examples, $B$ the batch size, $\lceil\cdot\rceil$ rounds up and $\lfloor\cdot\rfloor$ rounds down. For the ten squares with $B = 4$ that is $\lceil 2.5 \rceil = 3$ batches, or 2 with `drop_last=True`, exactly as in the figure.

```python
loader = DataLoader(w, batch_size=64, shuffle=True)
print(len(loader), len(DataLoader(w, batch_size=64, drop_last=True)))   # 14 13
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)       # torch.Size([64, 96]) torch.Size([64, 24])
```

Each example was `(96,)`; the batch is `(64, 96)`. The loader added a leading **batch dimension**, which is what an `nn.Module` expects (04.03).

<div class="ex">
<div class="lab">Worked example</div>

**counting windows and batches by hand**

A series has $T = 1000$ points. The model sees $c = 96$ past points and predicts $h = 24$ future points.

Number of windows: $N = 1000 - 96 - 24 + 1 = 881$. The first window uses points 0 to 95 as context and 96 to 119 as target; the last starts at 880 and uses points 880 to 975 and 976 to 999, which is the final point.

Batches of $B = 64$: $881 / 64 = 13.77$, so 13 full batches hold $13 \times 64 = 832$ windows and one final batch holds $881 - 832 = 49$. Total: $\lceil 13.77 \rceil = 14$ batches.

With `drop_last=True` only the 13 full batches are used, and 49 windows (5.6% of the data) are skipped this epoch. Because the order is reshuffled, a different 49 are skipped next epoch, so nothing is lost permanently.

</div>

Why drop it? A batch of 49 gives noisier batch-norm statistics, and a compiled model (04.07) may recompile for the new shape. For evaluation never drop anything: every example must be scored.

## Step 3: collate, from a list to a batch

After fetching `batch_size` examples, the loader has a Python list such as `[(x0, y0), (x1, y1), ...]`. The **collate function** turns that list into one batch. The built-in `default_collate` follows the structure of an example:

- tensors of the same shape are stacked along a new first dimension;
- tuples and lists are collated element by element, so a list of `(x, y)` pairs becomes `(stack(xs), stack(ys))`;
- dicts are collated key by key;
- Python numbers and NumPy arrays become tensors.

```python
from torch.utils.data import default_collate
print(default_collate([{'x': torch.ones(3), 'y': 1}, {'x': torch.zeros(3), 'y': 0}]))
# {'x': tensor([[1., 1., 1.], [0., 0., 0.]]), 'y': tensor([1, 0])}

default_collate([torch.ones(3), torch.ones(5)])
# RuntimeError: stack expects each tensor to be equal size, but got [3] at entry 0 and [5] at entry 1
```

Everyone meets that error with text, audio or detection boxes, where examples differ in length: stacking needs a rectangle. The fix is your own `collate_fn` that **pads** every sequence to the longest in the batch and returns a **mask** marking the real positions:

```python
def pad_collate(batch, pad_id=0):
    seqs = [torch.as_tensor(ids) for ids, _ in batch]
    lens = torch.tensor([len(s) for s in seqs])
    x = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    mask = torch.arange(x.size(1))[None, :] < lens[:, None]     # True on real tokens
    y = torch.tensor([label for _, label in batch])
    return {"input_ids": x, "attention_mask": mask, "labels": y}

data = [([5, 8, 2], 1), ([4, 9, 9, 3, 7, 1, 6], 0), ([2, 2], 1), ([7, 1, 3, 8, 5], 0)]
b = pad_collate(data)
print(b["input_ids"])
# tensor([[5, 8, 2, 0, 0, 0, 0],
#         [4, 9, 9, 3, 7, 1, 6],
#         [2, 2, 0, 0, 0, 0, 0],
#         [7, 1, 3, 8, 5, 0, 0]])
print(b["attention_mask"].float().mean())   # tensor(0.6071)
loader = DataLoader(data, batch_size=4, collate_fn=pad_collate)
```

The mask line is broadcasting (04.01): a row of positions `[0, ..., 6]` compared with a column of lengths gives a `(4, 7)` grid, `True` where the position is below that row's length. The model uses it to ignore padding in attention and in the loss.

~~~dg-fig 04.05-padding.svg
Left: the four sequences above, padded to length 7; 11 of 28 slots are padding. Right: on 20,000 texts with skewed lengths (measured here), random batches are mostly padding, while grouping similar lengths cuts waste to a few percent.
~~~

When lengths vary a lot, one long text forces every row in its batch to be padded to its length. With random batches of 128, 83% of all slots here were padding: most compute went on zeros. **Bucketing** (sort by length within large chunks, cut batches from the sorted chunks, shuffle the batch order) brought that to about 3%. LLM pretraining instead **packs** several documents into one fixed-length row, with a block-diagonal attention mask.

## Step 4: samplers decide the order

A **sampler** is an iterable of indices. The loader's flags usually pick one for you:

- `shuffle=False` uses a `SequentialSampler`: 0, 1, 2, ... every epoch. Use it for validation and test.
- `shuffle=True` uses a `RandomSampler`, which draws a fresh permutation every epoch. Pass `generator=` for reproducible order.
- `WeightedRandomSampler(weights, num_samples)` draws indices with replacement, with probability proportional to a per-example weight.
- `DistributedSampler` gives each GPU process (**rank**) a disjoint slice of one shared permutation. Call `sampler.set_epoch(epoch)` every epoch, or the order never changes.

The weighted sampler is the quickest fix for class imbalance: give each example the weight $1/n_c$, where $n_c$ is the size of its class, so every class has the same total weight.

```python
from torch.utils.data import WeightedRandomSampler
labels = torch.tensor([0] * 900 + [1] * 100)
counts = torch.bincount(labels)                       # tensor([900, 100])
weights = 1.0 / counts[labels].float()                # 0.00111 for class 0, 0.01 for class 1
sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
loader = DataLoader(TensorDataset(labels), batch_size=32, sampler=sampler)   # no shuffle= with a sampler
```

Check the arithmetic: class 0 has total weight $900 \times \frac{1}{900} = 1$ and class 1 has $100 \times \frac{1}{100} = 1$, so each draw is 50/50.

~~~dg-fig 04.05-sampler.svg
Share of the rare class in each batch of one epoch (measured). Plain shuffling gives about 10% and three batches with none at all; the weighted sampler gives about 50%.
~~~

With plain shuffling the rare class averaged 9.8% of each batch; with the weighted sampler, 51%. The price: drawing with replacement shows each rare example about five times per epoch while many common ones are skipped (one run drew only 484 distinct indices in 1,000 draws), so watch for overfitting on the rare class. Reweighting the loss (04.04) is the alternative.

## Step 5: workers fetch in parallel

With `num_workers=0` the main process runs `__getitem__` itself, so the model waits while data is read. With `num_workers=k` the loader starts `k` processes; each holds its own copy of the dataset, builds whole batches, and keeps `prefetch_factor` batches (default 2) ready ahead of time.

~~~dg-fig 04.05-workers.svg
Measured on this machine (timings are machine-dependent). Left: a loader whose items each wait about 2 ms scales almost linearly with workers. Right: without persistent workers, every epoch pays the start-up cost again.
~~~

The left panel's `__getitem__` sleeps 2 ms, standing in for reading one image. Measured throughput was about 14 batches per second with no workers, 14 with one (same work, another process), 28 with two, 55 with four and 100 with eight. Sleeping is waiting, not computing, so it kept scaling past this machine's four cores; CPU-bound decoding stops scaling near your core count.

Starting workers means launching processes and sending each a copy of the dataset. With `persistent_workers=True` that happens once; later epochs took about 0.07 s instead of 0.11 s.

With a GPU, `pin_memory=True` puts batches in **page-locked** (pinned) host memory, so `batch.to("cuda", non_blocking=True)` can overlap the copy with GPU compute. It changes speed, not results. A typical training loader:

```python
loader = DataLoader(
    Windows(series, ctx=96, horizon=24), batch_size=64, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
```

## Iterable datasets: when data is a stream

Some data cannot be indexed: a 500 GB file of JSON lines, a database cursor. An **IterableDataset** implements `__iter__` and yields examples. The loader rejects samplers here, so order and splitting are your job. The trap: each worker iterates its own copy from the start, so with four workers every line comes out four times.

```python
from torch.utils.data import IterableDataset, get_worker_info

class ShardedLines(IterableDataset):
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        info = get_worker_info()            # None in the main process
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
        return iter(range(wid, self.n, nw))  # worker wid reads lines wid, wid+nw, ...

print([b.tolist() for b in DataLoader(ShardedLines(12), batch_size=4, num_workers=2)])
# [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10], [9, 11]]
```

~~~dg-fig 04.05-iterable.svg
Counting how often each of 12 lines is yielded in one epoch with 4 workers (measured). Without sharding every line appears 4 times; with sharding by worker id each appears exactly once.
~~~

Batches alternate between workers, and each worker forms its own partial last batch, so streams can give several short batches per epoch.

## Reading the result: is the loader the bottleneck?

Slowness can come from the model or the data, so measure them separately. Iterate the loader alone for a few hundred batches:

```python
import time
t = time.perf_counter(); n = 0
for xb, yb in loader:
    n += 1
    if n == 300: break
print(f"{n / (time.perf_counter() - t):.0f} batches/s")   # the 800-row in-memory split above: ~5000 here
```

Then time the training step on one fixed batch reused repeatedly. If the loader delivers fewer batches per second than the model consumes, the loader is the bottleneck; otherwise profile the model (04.08). GPU utilisation that saw-tooths between high and near zero is the signature of a starved model.

Also check correctness: `len(loader)` matches your hand count; one batch has the expected shapes and dtypes; a stream yields as many examples per epoch as it has lines; and augmentation values differ between workers.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| `stack expects each tensor to be equal size` | variable-length examples with `default_collate` | custom `collate_fn` that pads and returns a mask
| Epoch has k times too many steps, loss drops suspiciously fast | `IterableDataset` not sharded across k workers (or ranks) | shard in `__iter__` using `get_worker_info()` and the rank
| Augmentations repeat in pairs | random generator created in `__init__` and copied into every worker | reseed it in `worker_init_fn` from `get_worker_info().seed`
| Host RAM grows epoch after epoch | dataset holds big Python lists or dicts that workers copy page by page | store data in NumPy arrays, tensors or memory-mapped files
| Every epoch has the same order on multi-GPU | `DistributedSampler.set_epoch` never called | call `sampler.set_epoch(epoch)` each epoch
| First batch of each epoch is slow | workers restarting | `persistent_workers=True`
| GPU utilisation saw-tooths | too few workers or heavy decode | more workers, decode once offline, or decode on the GPU
| Validation numbers change between runs | `shuffle=True` with `drop_last=True` on the validation loader | `shuffle=False`, `drop_last=False` for evaluation

The random-number bug is silent. A NumPy generator created in `__init__` is copied identically into each worker:

```python
import numpy as np
class Aug(Dataset):
    def __init__(self): self.rng = np.random.default_rng(0)     # created once, in the parent
    def __len__(self): return 4
    def __getitem__(self, i): return int(self.rng.integers(0, 1000))

def reseed(worker_id):
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed)         # a different seed per worker

print([int(v) for v in DataLoader(Aug(), batch_size=None, num_workers=2)])                       # [850, 850, 636, 636]
print([int(v) for v in DataLoader(Aug(), batch_size=None, num_workers=2, worker_init_fn=reseed)])  # [189, 748, 638, 709]
```

(`batch_size=None` turns off batching.) PyTorch seeds each worker's `torch`, `random` and global NumPy generators from a base seed plus the worker id, but cannot know about generator objects you created. Using torch's random functions for augmentation avoids the problem.

<div class="ex">
<div class="lab">Worked example</div>

**how many workers does a vision model need?**

A model trains at 20 steps per second with batches of 256 images, so it consumes $20 \times 256 = 5{,}120$ images per second.

One worker decodes and augments about 400 images per second, so keeping up needs $5{,}120 / 400 = 12.8$, that is 13 workers.

With 8 CPU cores per GPU that is impossible: workers beyond the core count only add context switching.

Options: decode once and store preprocessed tensors, decode and augment on the GPU, or reduce per-image work (a smaller resize), so fewer workers suffice.

</div>

## Going deeper

**Where the memory goes.** On Linux workers start by `fork`: they share the parent's memory pages, and a page is copied only when written. But Python's reference counting writes a counter into every object it touches, so reading a list of a million strings copies their pages into every worker, and memory grows with `num_workers` like a leak. A NumPy array or tensor is one object with one buffer, so its pages stay shared; memory-mapped files (`np.memmap`, Arrow) let the OS share the page cache too. On macOS and Windows the default is `spawn`: each worker gets a pickled copy of the dataset, and scripts need the `if __name__ == "__main__":` guard.

**Splitting a stream across GPUs and workers.** With $R$ ranks and $W$ workers per rank there are $RW$ readers. Reader $(r, w)$ gets the global index $g = rW + w$ and keeps items whose position modulo $RW$ equals $g$. For big data, split it into many shard files and assign whole files, so no reader scans lines it discards.

**Batch samplers.** `batch_sampler=` yields whole lists of indices: the natural place for bucketing or "exactly $m$ examples per class per batch".

**When CPUs cannot keep up**, production systems decode on the GPU or preprocess once into compact shards so `__getitem__` is a cheap read. Fixing the loader's `generator=` seed makes shuffle order and worker seeds reproducible, though augmentations still change if `num_workers` changes.

## Key terms

| Term | Meaning
| --- | --- |
| Map-style `Dataset` | Object with `__len__` and `__getitem__(i)`; behaves like a read-only list of examples
| `IterableDataset` | Object whose `__iter__` yields examples one after another; for streams that cannot be indexed
| `DataLoader` | Combines a dataset, a sampler, a collate function and workers into an iterator of batches
| Sampler | Iterable of indices that decides which examples are read and in what order
| Collate function | Turns a list of examples into one batch, by stacking or by padding
| Padding and mask | Filling short sequences up to a common length, plus a boolean grid marking real positions
| Worker | Separate process that runs `__getitem__` and builds batches in parallel
| Pinned memory | Page-locked host memory the GPU can copy from asynchronously

<div class="take">
<div class="lab">Key takeaway</div>

a Dataset returns one example; the DataLoader decides order (sampler), shape (collate) and parallelism (workers). Count your batches, collate variable-length data yourself, shard streams and reseed generators per worker, store data so workers do not copy it, and time the loader alone before blaming the model.

</div>
