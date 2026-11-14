import numpy as np

META = dict(title='Time-based train/validation split with a gap', tags=['Time Series', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            related=['k-fold-cross-validation', 'lag-features', 'point-in-time-feature-join'])

STATEMENT = r"""
Write `time_splits(times, n_splits, val_size, gap)` producing expanding-window splits for time-ordered data.

- `times`: 1-D array of timestamps (integers), not necessarily sorted. Work with the **sorted unique** timestamps $u_0 < u_1 < \dots < u_{m-1}$.
- Split $s$ (for $s = 0, \dots,$ `n_splits` $-1$) validates on the block of `val_size` unique timestamps ending at position $m - 1 - (\text{n\_splits} - 1 - s)\cdot\text{val\_size}$. So the last split's validation block is the last `val_size` timestamps, and earlier splits step back by `val_size`.
- Training uses every unique timestamp strictly before the validation block, except the `gap` timestamps immediately before it.
- Return a list of `(train_idx, val_idx)` pairs of **row** indices into `times` (sorted ascending). Skip a split if its training set would be empty.
"""

STARTER = '''import numpy as np

def time_splits(times, n_splits, val_size, gap):
    """Return a list of (train_idx, val_idx) arrays."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def time_splits(times, n_splits, val_size, gap):
    times = np.asarray(times)
    u = np.unique(times)
    m = len(u)
    out = []
    for s in range(n_splits):
        end = m - 1 - (n_splits - 1 - s) * val_size
        start = end - val_size + 1
        if start < 0:
            continue
        val_t = u[start:end + 1]
        train_t = u[:max(start - gap, 0)]
        if len(train_t) == 0:
            continue
        out.append((np.where(np.isin(times, train_t))[0], np.where(np.isin(times, val_t))[0]))
    return out
'''

HINTS = [
    'Work in units of unique timestamps; several rows can share one timestamp and must land on the same side of the split.',
    'For each split, find the validation block first, then train on everything before it minus the gap.',
    '`np.isin(times, block)` turns a set of timestamps back into row indices.',
]

TESTS = [
    (np.arange(10), 2, 2, 0),
    (np.arange(10), 2, 2, 1),
    (np.array([5, 1, 3, 3, 2, 4, 1, 5]), 1, 1, 1),
    (np.repeat(np.arange(6), 2), 3, 1, 2),
    (np.arange(4), 3, 2, 0),
    (np.array([20240101, 20240102, 20240102, 20240103, 20240104, 20240105]), 2, 1, 0),
]

EXPLANATION = r"""
## Why not random k-fold
With time-ordered data, random folds let the model train on the future and validate on the past. Features like rolling averages and target encodings then leak information across the split, and validation scores are optimistic.

## Why a gap
Labels and features often overlap in time: a 7-day rolling feature at validation time uses days that sit at the end of training, and a 30-day churn label for a training day is only known 30 days later. Leaving a gap at least as long as that overlap keeps the split honest.

## Expanding versus sliding windows
This version grows the training set; a sliding window keeps a fixed history length, which helps when old data no longer reflects the present.

## Interview follow-ups
- How would you choose the gap for a model whose label is "purchases in the next 14 days"?
"""
