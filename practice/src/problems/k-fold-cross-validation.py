import numpy as np

META = dict(title='K-fold cross-validation splits', tags=['Metrics & Evaluation', 'Data Processing'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:04.23'], related=['stacking-ensemble', 'regression-metrics'])

STATEMENT = r"""
K-fold cross-validation partitions the rows into $k$ folds; each fold is the validation set once while the other $k - 1$ folds form the training set.

Write `kfold_indices(n, k, seed)` that returns a list of $k$ pairs `(train_idx, val_idx)`.

## Rules
1. **Order**: if `seed` is `None`, use the rows in order `0, 1, ..., n-1`. Otherwise use `np.random.default_rng(seed).permutation(n)`.
2. **Fold sizes**: the first `n % k` folds get `n // k + 1` rows, the rest get `n // k`. Fold $f$ takes the next contiguous chunk of the ordering above.
3. **Output**: `val_idx` is fold $f$'s rows and `train_idx` is every other row, **both sorted ascending**, as int arrays. Folds are listed in order $f = 0, \dots, k-1$.

## Input
`n >= 2`; `2 <= k <= n`; `seed`: int or `None`.
"""

STARTER = '''import numpy as np

def kfold_indices(n, k, seed):
    """List of k (train_idx, val_idx) pairs, both sorted ascending."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def kfold_indices(n, k, seed):
    order = np.arange(n) if seed is None else np.random.default_rng(seed).permutation(n)
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1                                  # spread the remainder over the first folds
    folds, start = [], 0
    for s in sizes:
        val = np.sort(order[start:start + s])
        folds.append((np.setdiff1d(np.arange(n), val), val))   # setdiff1d returns sorted output
        start += s
    return folds
'''

HINTS = [
    'Separate the two jobs: decide the row order (identity or a seeded permutation), then cut it into contiguous chunks.',
    'Fold sizes: start with `n // k` everywhere and add 1 to the first `n % k` folds. `np.array_split` does exactly this.',
    'Sort each validation chunk, and get the training rows with `np.setdiff1d(np.arange(n), val)`, which is already sorted.',
]

TESTS = [
    (6, 3, None),
    (7, 3, None),
    (10, 4, 0),
    (5, 5, None),
    (2, 2, 3),
    (9, 2, 42),
    (23, 5, 7),
]

EXPLANATION = r"""
## Why it works
Every row is validated exactly once by a model that never saw it, so the average validation score uses all $n$ rows for evaluation and almost all of them for training each time. Compared with a single hold-out split, the estimate has lower variance and wastes less data.

## Bias and variance of the estimate
Each model trains on $(k-1)/k$ of the data, so CV slightly underestimates the performance of a model trained on everything (pessimistic bias, larger for small $k$). Larger $k$ reduces that bias but the $k$ estimates become highly correlated, and leave-one-out ($k = n$, test 4) can have high variance. $k = 5$ or $10$ is the usual compromise.

## Complexity
$O(n)$ to build the indices; the real cost is training $k$ models.

## Common bugs
- Giving the remainder to the last fold only, so one fold is much bigger.
- Shuffling with the global NumPy RNG (`np.random.shuffle`) instead of a seeded generator, which makes splits irreproducible.
- Shuffling time-series data. Use forward-chaining splits there instead.

## Interview follow-ups
- Implement stratified k-fold, so each fold keeps the class proportions of the full data.
- You tuned hyperparameters with 5-fold CV and report the best CV score. Why is that optimistic, and how does nested CV fix it?
"""
