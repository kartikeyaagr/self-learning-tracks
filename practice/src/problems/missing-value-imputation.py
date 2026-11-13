import numpy as np

META = dict(title='Column-wise missing value imputation', tags=['Data Processing', 'Feature Engineering'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:11.12'], related=['feature-scaling', 'categorical-encoding'])

STATEMENT = r"""
Missing values are encoded as `nan`. Write `impute(X_train, X_test, strategy)` that learns one fill value per column from `X_train` and uses it to replace the `nan`s in **both** arrays.

## Strategies
| `strategy` | fill value for a column |
| --- | --- |
| `'mean'` | mean of the column's non-missing training values |
| `'median'` | median of the non-missing training values (average of the two middle values when their count is even) |
| `'most_frequent'` | the most common non-missing training value; ties go to the **smallest** value |

If a column has **no** non-missing training values, its fill value is `0.0`.

## Input / output
- `X_train`: float `(n, d)`; `X_test`: float `(m, d)`; either may contain `nan` anywhere.
- Return `(train_filled, test_filled, fill)`: float arrays `(n, d)`, `(m, d)` and `(d,)`. Don't modify the inputs.
- Fill values come from the training data only, never from `X_test`.
"""

STARTER = '''import numpy as np

def impute(X_train, X_test, strategy):
    """Fill nan per column using statistics of X_train. Return (train_filled, test_filled, fill_values)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def impute(X_train, X_test, strategy):
    X_train, X_test = np.array(X_train, dtype=float), np.array(X_test, dtype=float)   # copies
    fill = np.zeros(X_train.shape[1])
    for j in range(X_train.shape[1]):
        col = X_train[:, j][~np.isnan(X_train[:, j])]
        if len(col) == 0:
            continue                                         # nothing observed: keep 0.0
        if strategy == 'mean':
            fill[j] = col.mean()
        elif strategy == 'median':
            fill[j] = np.median(col)
        else:
            vals, counts = np.unique(col, return_counts=True)
            fill[j] = vals[counts.argmax()]                  # vals ascending: first max is the smallest value
    for A in (X_train, X_test):
        rows, cols = np.nonzero(np.isnan(A))
        A[rows, cols] = fill[cols]
    return X_train, X_test, fill
'''

HINTS = [
    '`np.nan` is not equal to itself, so `X == np.nan` finds nothing. Use `np.isnan` to build the mask.',
    'For each column, drop the missing entries first, then compute the statistic on what\'s left. `np.nanmean`/`np.nanmedian` exist, but they warn and return `nan` on an all-missing column.',
    'Fill with fancy indexing: `rows, cols = np.nonzero(np.isnan(A)); A[rows, cols] = fill[cols]`. For the mode, `np.unique(..., return_counts=True)` sorts the values, so `argmax` breaks ties toward the smallest.',
]

nan = np.nan
_r = np.random.default_rng(137)
_A = _r.integers(0, 5, size=(30, 4)).astype(float)
_A[_r.uniform(size=_A.shape) < 0.25] = nan
_B = _r.normal(size=(6, 4))
_B[_r.uniform(size=_B.shape) < 0.3] = nan
TESTS = [
    (np.array([[1.0, nan], [3.0, 4.0], [nan, 8.0]]), np.array([[nan, nan]]), 'mean'),
    (np.array([[1.0, 2.0], [nan, 2.0], [7.0, 5.0], [4.0, nan]]), np.array([[nan, 1.0]]), 'median'),
    (np.array([[1.0], [2.0], [2.0], [1.0], [nan]]), np.array([[nan]]), 'most_frequent'),
    (np.array([[nan, 1.0], [nan, 2.0]]), np.array([[nan, nan]]), 'mean'),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0]]), 'median'),
    (np.array([[-1e9, 5.0], [1e9, nan], [3.0, 5.0], [nan, -2.0]]), np.array([[nan, nan], [0.0, nan]]), 'median'),
    (_A, _B, 'most_frequent'),
    (_A, _B, 'mean'),
]

EXPLANATION = r"""
## Why fit on train only
The fill value is a learned parameter, just like a scaler's mean. Computing it on the test set (or on train + test) leaks information about the evaluation data into the pipeline and makes offline metrics optimistic. In production there's no "test set" to peek at anyway: new rows are filled with the training statistic.

## Choosing a strategy
- **Mean** keeps the column mean unchanged but shrinks its variance and is pulled by outliers (test 6 has $\pm 10^9$).
- **Median** is robust to outliers and skew.
- **Most frequent** works for discrete or categorical-coded columns.
All three ignore relationships between features. k-NN or iterative (regression-based) imputation use them, at a higher cost.

## Complexity
$O(nd)$ for mean and mask work; median and mode add a sort per column, $O(n\log n)$ each.

## Common bugs
- Checking `x == np.nan` or `x is np.nan`.
- Using `np.nanmean` on an all-`nan` column and propagating `nan` into the output (test 4).
- Mutating the caller's arrays in place.

## Interview follow-ups
- Why add a binary "was missing" indicator feature, and when does missingness itself carry signal (missing not at random)?
- How do gradient-boosted tree libraries handle `nan` without imputation? (Each split learns a default direction for missing values.)
"""
