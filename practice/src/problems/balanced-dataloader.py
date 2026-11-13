import numpy as np

META = dict(title='Class-balanced batches with a custom Sampler', tags=['Classification', 'Data Processing'], diff='Medium', runtime='torch',
            lessons=['air-fanout-v1:10.03'],
            related=['custom-dataset-class', 'hashed-dataset-split', 'log-loss-binary-cross-entropy'])

STATEMENT = r"""
With a 95/5 class split, random batches of 32 often contain no minority example at all. Two standard fixes: weight each sample by inverse class frequency (for `WeightedRandomSampler`), or build every batch with the same number of examples per class. Implement both, deterministically so they can be tested.

## Sample weights
For labels $y_1, \ldots, y_N$ over $C$ distinct classes with counts $n_c$:
$$w_i = \frac{1}{C\, n_{y_i}}.$$
Every class then carries total weight $1/C$, and the weights sum to 1.

## Balanced batch sampler
Write `BalancedBatchSampler(torch.utils.data.Sampler)` built from `(labels, per_class, num_batches)`:

- Classes are the distinct labels in ascending order. For each class, keep its sample indices in ascending order and a cursor that starts at 0.
- Each batch takes the next `per_class` indices of every class in turn (smallest label first), advancing that class's cursor and **wrapping around** to the start of its list when it runs out. So minority samples repeat.
- `__iter__` yields `num_batches` batches, each a list of `C * per_class` ints; `__len__` returns `num_batches`.

## The function that's graded
Write `balanced_batches(labels, per_class, num_batches)` where `labels` is an int64 tensor `(N,)`. Return `(weights, batches, n)`:

- `weights`: float64 `(N,)`;
- `batches`: int64 `(num_batches, C * per_class)`, the sampler's batches in order;
- `n`: `len(sampler)`.

In practice you'd shuffle each class's list every time it wraps; here the order is fixed so the output is checkable.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, per_class, num_batches):
        self.labels, self.per_class, self.num_batches = labels, per_class, num_batches

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_batches

def balanced_batches(labels, per_class, num_batches):
    """Return (weights (N,), batches (num_batches, C * per_class), len(sampler))."""
    weights = torch.full(labels.shape, 1.0 / len(labels), dtype=torch.float64)
    sampler = BalancedBatchSampler(labels, per_class, num_batches)
    return weights, torch.tensor(list(sampler)), len(sampler)
'''

SOLUTION = '''import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, per_class, num_batches):
        self.per_class, self.num_batches = per_class, num_batches
        classes = torch.unique(labels)                                   # sorted ascending
        self.pools = [torch.nonzero(labels == c).flatten().tolist() for c in classes]

    def __iter__(self):
        cursors = [0] * len(self.pools)
        for _ in range(self.num_batches):
            batch = []
            for c, pool in enumerate(self.pools):
                for _ in range(self.per_class):
                    batch.append(pool[cursors[c] % len(pool)])            # wrap: minority samples repeat
                    cursors[c] += 1
            yield batch

    def __len__(self):
        return self.num_batches

def balanced_batches(labels, per_class, num_batches):
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    weights = 1.0 / (len(counts) * counts[inverse].to(torch.float64))     # each class sums to 1/C
    sampler = BalancedBatchSampler(labels, per_class, num_batches)
    return weights, torch.tensor(list(sampler), dtype=torch.int64), len(sampler)
'''

HINTS = [
    '`torch.unique(labels, return_inverse=True, return_counts=True)` gives the sorted classes, each sample\'s class position, and the counts. `counts[inverse]` is $n_{y_i}$ for every sample.',
    'Store one index list per class (`torch.nonzero(labels == c)`), and one integer cursor per class that persists across batches within one `__iter__` call.',
    'Wrap with `pool[cursor % len(pool)]`. Reset the cursors at the start of `__iter__` so iterating the sampler twice gives the same batches.',
]

TESTS = [
    (np.array([0, 0, 0, 1]), 1, 3),
    (np.array([1, 0, 1, 1, 0, 1]), 2, 2),
    (np.array([2, 2, 5, 2, 7, 5, 2]), 1, 4),
    (np.array([3, 3, 3]), 2, 2),
    (np.array([0, 1, 2, 0, 1, 2]), 2, 3),
    (np.array([1] * 9 + [0]), 3, 2),
    (np.random.default_rng(4).choice(4, size=25, p=[0.6, 0.25, 0.1, 0.05]), 2, 5),
]

EXPLANATION = r"""
## Why it works
A `Sampler` only decides **which indices** the `DataLoader` loads; the dataset still does the loading. Passing it as `DataLoader(ds, batch_sampler=sampler)` makes each yielded list one batch, so every batch has exactly `per_class` examples of each class regardless of the label distribution.

The weights do the same job in expectation: `WeightedRandomSampler(weights, num_samples, replacement=True)` draws index $i$ with probability $w_i$, so each class is drawn with probability $n_c \cdot \frac{1}{C n_c} = \frac1C$.

## Which one to use
- **Weighted sampling** is a one-liner and keeps batches random, but a batch can still be unbalanced by chance and minority samples are drawn with replacement.
- **Balanced batches** guarantee composition, which metric-learning losses (triplet, supervised contrastive) need because they compare within a batch.
- Either way you are changing the training prior: predicted probabilities are calibrated to a balanced world, so rescale by the true priors if you need calibrated outputs.

## Complexity
Building pools is $O(N)$; each batch costs $O(C \cdot \text{per\_class})$.

## Common bugs
- Cursors stored on `self` and never reset: the second epoch starts where the first stopped (fine in training, but non-reproducible and it breaks `len` assumptions).
- Weights of $1/n_c$ without normalising: harmless for `WeightedRandomSampler`, but wrong if they're reused as loss weights expecting a fixed total.
- Upsampling the minority before the train/validation split, which leaks duplicates into validation.

## Interview follow-ups
- Compare class-balanced sampling with a class-weighted loss and with focal loss. When does each work best?
- How do you keep balanced sampling with `DistributedDataParallel`, where each rank must see different indices?
"""
