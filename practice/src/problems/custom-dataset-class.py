import numpy as np

META = dict(title='A sliding-window Dataset for forecasting', tags=['Time Series', 'Data Processing'], diff='Easy', runtime='torch',
            lessons=['air-fanout-v1:10.03', 'air-fanout-v1:04.27'],
            related=['balanced-dataloader', 'transform-pipeline', 'delayed-label-monitoring'])

STATEMENT = r"""
Forecasting models train on (past window, future values) pairs cut from one long series. A `torch.utils.data.Dataset` makes that slicing lazy: nothing is copied until an item is asked for.

## The Dataset
Write `WindowDataset(Dataset)` constructed from `(series, window, horizon, stride)`:

- `series`: float64 tensor `(L,)` or `(L, F)`.
- Item `i` starts at `t = i * stride` and returns the pair
  - `x = series[t : t + window]`,
  - `y = series[t + window : t + window + horizon]`.
- Only complete pairs count, so `len(ds)` is $\max\big(0,\ \lfloor (L - \text{window} - \text{horizon}) / \text{stride} \rfloor + 1\big)$.
- Negative indices work like Python lists (`ds[-1]` is the last item). An index outside `[-len, len)` raises `IndexError`.

## The function that's graded
Write `window_items(series, window, horizon, stride, idxs)`:

- `window`, `horizon`, `stride` are ints $\ge 1$; `idxs` is a non-empty list of valid indices.
- Return `(n, X, Y)`: `n = len(ds)`, and `X`, `Y` the items at `idxs` stacked along a new first axis: `X` is `(len(idxs), window)` or `(len(idxs), window, F)`, and `Y` likewise with `horizon`.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, series, window, horizon, stride):
        self.series, self.window, self.horizon, self.stride = series, window, horizon, stride

    def __len__(self):
        return len(self.series) // self.stride

    def __getitem__(self, i):
        t = i * self.stride
        return self.series[t:t + self.window], self.series[t + self.window:t + self.window + self.horizon]

def window_items(series, window, horizon, stride, idxs):
    """Return (len(ds), X stacked, Y stacked)."""
    ds = WindowDataset(series, window, horizon, stride)
    xs, ys = zip(*(ds[i] for i in idxs))
    return len(ds), torch.stack(xs), torch.stack(ys)
'''

SOLUTION = '''import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, series, window, horizon, stride):
        self.series, self.window, self.horizon, self.stride = series, window, horizon, stride
        span = window + horizon                                   # samples one item needs
        self.n = max(0, (len(series) - span) // stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if not -self.n <= i < self.n:
            raise IndexError(f'index {i} out of range for {self.n} items')
        i %= self.n                                               # Python-style negative indexing
        t = i * self.stride
        x = self.series[t:t + self.window]                        # views: no copy until stacking
        y = self.series[t + self.window:t + self.window + self.horizon]
        return x, y

def window_items(series, window, horizon, stride, idxs):
    ds = WindowDataset(series, window, horizon, stride)
    xs, ys = zip(*(ds[i] for i in idxs))
    return len(ds), torch.stack(xs), torch.stack(ys)
'''

HINTS = [
    'An item needs `window + horizon` consecutive samples. Count how many start positions `0, stride, 2*stride, ...` still leave that much room.',
    'Clamp the count at 0 for series shorter than one window, and check `-n <= i < n` before converting a negative index with `i % n`.',
    'Slicing along the first axis works for both `(L,)` and `(L, F)` series, so you don\'t need a special case for multivariate data.',
]

TESTS = [
    (np.arange(6.0), 3, 1, 1, [0, 2]),
    (np.arange(10.0), 4, 2, 2, [0, -1]),
    (np.arange(9.0), 2, 3, 3, [1, -2]),
    (np.arange(24.0).reshape(12, 2), 3, 2, 2, [0, 1, 3]),
    (np.arange(5.0), 4, 1, 1, [0, -1]),
    (np.arange(11.0), 2, 2, 4, [1, 0, -2]),
    (np.random.default_rng(2).normal(size=(30, 3)), 7, 3, 5, [-1, 0, 2, 4]),
]

EXPLANATION = r"""
## Why it works
A map-style `Dataset` is just `__len__` plus `__getitem__`. The `DataLoader` asks for indices `0..len-1` (shuffled or not) and stacks the returned items into a batch with its default `collate_fn`, which is what `torch.stack` does here. Basic slicing returns views, so the dataset holds one copy of the series no matter how many overlapping windows it serves.

The count follows from the last valid start: $t = (n-1)\cdot\text{stride}$ must satisfy $t + \text{window} + \text{horizon} \le L$.

## Why the length must be exact
`DataLoader` trusts `__len__`. One too many and the last batch reads a short window, which then fails in `collate` with a shape mismatch (or worse, silently trains on a truncated target if batch size is 1). The starter's `len(series) // stride` overcounts.

## Complexity
`__getitem__` is $O(1)$ to create views, $O(\text{window} + \text{horizon})$ when the batch is stacked.

## Common bugs
- Off-by-one in the count (`- 1` too many or too few).
- Allowing negative indices to fall through to tensor slicing: `series[-3:...]` slices from the end of the series, not from the end of the dataset.
- Splitting train/validation by shuffling windows: overlapping windows leak future values into training. Split by time first, then window.

## Interview follow-ups
- When would you switch from a map-style `Dataset` to an `IterableDataset`?
- With `num_workers > 0`, each worker gets a copy of the dataset object. How do you avoid duplicating a 10 GB series in memory? (Memory-map it.)
"""
