import numpy as np

META = dict(title='Standardise with train statistics', tags=['Feature Engineering', 'Probability and Statistics'],
            diff='Medium', runtime='numpy', lessons=['mlmaths:10.04'], related=['column-scaling', 'winsorize', 'row-scaling'])

STATEMENT = r"""
Standardising a feature means subtracting its mean and dividing by its standard deviation. To avoid leaking information, the statistics must come from the **training** data only and then be applied unchanged to new data.

Write `standardize(train, test)` that returns `(Z, mu, sd)`.

## Input
- `train`: float array `(n, d)`; `test`: float array `(m, d)`, with `n, m, d >= 1`.
- Either may contain `NaN` for missing values. Every column of `train` has at least one non-NaN value.

## Rules
1. `mu[j]` is the mean of the non-NaN values in column `j` of `train`.
2. `sd[j]` is their **population** standard deviation (`ddof=0`), also ignoring NaN.
3. If `sd[j] == 0` (a constant column), use `sd[j] = 1.0` instead, so the column is only centred.
4. `Z = (test - mu) / sd`, column-wise. A NaN in `test` stays NaN in `Z`.

## Output
`Z`: `float64` array `(m, d)`. `mu`, `sd`: `float64` arrays `(d,)`, where `sd` is the value actually used (after rule 3).
"""

STARTER = '''import numpy as np

def standardize(train, test):
    """Return (Z (m, d), mu (d,), sd (d,)) using NaN-aware, ddof=0 statistics from train."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def standardize(train, test):
    train, test = np.asarray(train, dtype=float), np.asarray(test, dtype=float)
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)            # ddof=0 is the default
    sd = np.where(sd == 0, 1.0, sd)          # constant columns: centre only
    return (test - mu) / sd, mu, sd          # NaN in test propagates through the arithmetic
'''

HINTS = [
    'NaN poisons ordinary reductions: `np.mean([1, nan])` is `nan`. NumPy has NaN-ignoring versions of most reductions.',
    'Use `np.nanmean(train, axis=0)` and `np.nanstd(train, axis=0)` (whose default is `ddof=0`). Both give one value per column.',
    'Replace zero deviations with `np.where(sd == 0, 1.0, sd)`, then `(test - mu) / sd` broadcasts across the test rows. Return the corrected `sd`.',
]

nan = np.nan
_r = np.random.default_rng(47)
_tr = _r.normal(size=(30, 4)) * np.array([1.0, 5.0, 0.1, 50.0]) + np.array([0.0, 10.0, -3.0, 100.0])
_tr[_r.random(size=_tr.shape) < 0.1] = nan
_te = _r.normal(size=(6, 4)) * np.array([1.0, 5.0, 0.1, 50.0]) + np.array([0.0, 10.0, -3.0, 100.0])
_te[0, 2] = nan
TESTS = [
    (np.array([[1.0, 10.0], [3.0, 30.0]]), np.array([[2.0, 0.0], [5.0, 30.0]])),
    (np.array([[1.0, nan], [3.0, 4.0], [nan, 8.0]]), np.array([[2.0, nan]])),
    (np.array([[7.0, 7.0], [7.0, 9.0]]), np.array([[8.0, 8.0]])),
    (np.array([[5.0]]), np.array([[5.0], [6.0]])),
    (np.array([[1e9 + 1.0], [1e9 + 3.0]]), np.array([[1e9 + 2.0], [1e9 + 7.0]])),
    (np.array([[-1.0, nan, 2.0], [1.0, 3.0, 2.0], [nan, 5.0, 2.0]]), np.array([[0.0, 4.0, 3.0], [nan, nan, nan]])),
    (_tr, _te),
]

EXPLANATION = r"""
## Why it works
`np.nanmean` and `np.nanstd` compute each column's statistics over the values that are present, as if the NaNs were deleted from that column only. Dropping whole rows with any NaN would waste the other columns' data. With `ddof=0` the deviation is $\sqrt{\tfrac{1}{k}\sum (x-\mu)^2}$ over the $k$ present values, which is what scikit-learn's `StandardScaler` uses.

The `(d,)` vectors broadcast across the test rows, and arithmetic with NaN yields NaN, so missing test values stay missing without extra code. Replacing a zero deviation by 1 keeps the column finite (all zeros after centring) instead of filling it with `0/0 = NaN` or `inf`.

## Complexity
$O((n + m)d)$ time.

## Common bugs
- Fitting on `test` or on train and test together. Validation scores then look better than they will in production (leakage).
- Using `ddof=1` (sample standard deviation). That is also a fine convention, but it is not the one specified.
- Plain `np.mean` / `np.std` on data with NaN, which makes the entire column NaN.
- Testing `sd < 1e-12` instead of `sd == 0`. That is a reasonable engineering choice, but it changes the answer for nearly constant columns.

## Interview follow-ups
- After imputing missing values with the training mean, what is the standardised value of an imputed cell? (Exactly 0, which is why "impute then standardise" and "standardise then fill with 0" agree.)
- Why do tree models not need standardisation, while k-NN, SVMs and gradient-descent-trained linear models do? (Trees split on thresholds per feature and are invariant to monotone rescaling. Distance- and gradient-based methods are not.)
"""
