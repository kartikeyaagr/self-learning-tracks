import numpy as np

META = dict(title='Permutation feature importance', tags=['Feature Engineering', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            lessons=[], related=['random-forest', 'regression-metrics'])

STATEMENT = r"""
Permutation importance asks: how much worse does a fitted model get if one feature's values are shuffled, breaking its link to the target while keeping its distribution?

The model here is a fixed linear regressor $\hat y = Xw + b$, scored by mean squared error. Write `permutation_importance(X, y, w, b, n_repeats, seed)` that returns one importance per feature:

$$I_j = \frac{1}{R}\sum_{r=1}^{R}\Big(\mathrm{MSE}\big(y, \hat y(X^{(j, r)})\big) - \mathrm{MSE}\big(y, \hat y(X)\big)\Big),$$

where $X^{(j,r)}$ is a copy of $X$ with column $j$ permuted and $R$ = `n_repeats`.

## Randomness (follow exactly)
- Create one generator: `rng = np.random.default_rng(seed)`.
- Loop over features $j = 0, \dots, d-1$ (outer) and repeats (inner). Each (feature, repeat) pair calls `perm = rng.permutation(n)` once, and sets column $j$ of the copy to `X[perm, j]`.

## Input / output
- `X`: float `(n, d)`; `y`: float `(n,)`; `w`: float `(d,)`; `b`: float; `n_repeats >= 1`; `seed`: int.
- Return a float array `(d,)`. Importances can be negative (shuffling made the model better by chance).
"""

STARTER = '''import numpy as np

def permutation_importance(X, y, w, b, n_repeats, seed):
    """Mean increase in MSE of y_hat = Xw + b when each column is shuffled. Returns (d,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def permutation_importance(X, y, w, b, n_repeats, seed):
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    mse = lambda A: np.mean((y - (A @ w + b)) ** 2)
    base = mse(X)
    out = np.zeros(d)
    for j in range(d):
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(n), j]       # shuffle one column, keep the rest intact
            out[j] += mse(Xp) - base
    return out / n_repeats
'''

HINTS = [
    'Compute the baseline MSE once, on the unshuffled data.',
    'For each feature and repeat: copy `X`, replace column `j` with `X[perm, j]`, and record the increase in MSE.',
    'The order of `rng.permutation(n)` calls matters: features in the outer loop, repeats in the inner loop, one generator for everything.',
]

_r = np.random.default_rng(43)
_X = _r.normal(size=(60, 4))
_w = np.array([3.0, 0.0, -1.0, 0.2])
_y = _X @ _w + 0.5 + 0.3 * _r.normal(size=60)
TESTS = [
    (np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]]), np.array([2.0, 4.0, 6.0, 8.0]), np.array([2.0, 0.0]), 0.0, 3, 0),
    (np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]), np.array([1.0, 1.0, 4.0]), np.array([1.0, 1.0]), 0.0, 2, 1),
    (np.array([[1.0], [1.0], [1.0]]), np.array([3.0, 3.0, 3.0]), np.array([2.0]), 1.0, 5, 2),
    (np.array([[2.0, -1.0]]), np.array([0.0]), np.array([1.0, 1.0]), 0.0, 4, 3),
    (np.array([[1.0, 10.0], [2.0, -10.0], [3.0, 20.0], [4.0, 0.0]]), np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 0.0]), 0.0, 10, 4),
    (_X, _y, _w, 0.5, 5, 7),
    (_X * 1000.0, _y * 1000.0, _w, 500.0, 3, 8),
]

EXPLANATION = r"""
## Why it works
Shuffling a column keeps that feature's marginal distribution but destroys its relationship with the target and with the other features. If the model relies on the feature, its error rises; if not, nothing changes (test 1's constant column and test 5's zero-weight column both score exactly 0). Unlike coefficient magnitudes, this works for any model and is in the units of the metric.

## Complexity
$d \cdot R$ re-scorings of the model, each $O(nd)$ here. For an expensive model that's the dominant cost, but there's no retraining.

## Common bugs
- Shuffling the column in place in `X` and never restoring it, so later features are scored on corrupted data.
- Permuting rows of the whole matrix instead of one column (which changes nothing for the model).
- Computing the importance on training data and reading it as generalisation. Use held-out data in practice.

## Pitfalls of the method
- Correlated features share credit: shuffle one and the model leans on its twin, so both look unimportant.
- Shuffling creates unrealistic rows (for example, a 3-year-old with a mortgage), and the model is scored on points far outside the training distribution.

## Interview follow-ups
- How is this different from impurity-based (MDI) importance in random forests, and why is MDI biased toward high-cardinality features?
- How would you fix the correlated-features problem? (Permute groups of correlated features together, or use conditional permutation / drop-column importance.)
"""
