import numpy as np

META = dict(title='NaN-aware row extremes', tags=['Data Processing'], diff='Medium', runtime='numpy',
            related=['boolean-masking', 'filter-and-extract', 'aggregation-functions'])

STATEMENT = r"""
Write `row_extremes(X)` that summarises each row of a matrix with missing values by its minimum, its maximum and the column of its maximum, **ignoring NaN**.

## Input
`X`: float array of shape `(n, d)` with `n, d >= 1`. It may contain `NaN` (missing) and `inf` / `-inf` (valid values).

## Output
A tuple `(mins, maxs, argmax)`:
- `mins`, `maxs`: `float64` arrays `(n,)`, the smallest and largest non-NaN value of each row.
- `argmax`: `int64` array `(n,)`, the column index of the row's maximum. If the maximum occurs more than once, return the **first** (leftmost) such column.
- A row that is **entirely NaN** gets `mins = maxs = NaN` and `argmax = -1`.

## Notes
- `np.nanargmax` raises `ValueError` on an all-NaN row, so it cannot be applied to the whole matrix as-is.
- Watch rows whose maximum is `-inf`: in `[nan, -inf]` the answer is column 1.
- Vectorise it: no Python loop over rows.
"""

STARTER = '''import numpy as np

def row_extremes(X):
    """Return (mins, maxs, argmax) per row ignoring NaN; all-NaN rows -> (nan, nan, -1)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def row_extremes(X):
    X = np.asarray(X, dtype=float)
    valid = ~np.isnan(X)
    empty = ~valid.any(axis=1)
    # Replace NaN by the identity of each reduction so it can never win.
    mins = np.where(valid, X, np.inf).min(axis=1)
    maxs = np.where(valid, X, -np.inf).max(axis=1)
    # First valid column equal to the max. Comparing values (not argmax of the filled matrix)
    # matters when the max is -inf, which a NaN filled with -inf would also equal.
    hit = valid & (X == maxs[:, None])
    argmax = np.where(empty, -1, hit.argmax(axis=1))
    mins[empty] = np.nan
    maxs[empty] = np.nan
    return mins, maxs, argmax.astype(np.int64)
'''

HINTS = [
    'Build a boolean `valid = ~np.isnan(X)` once. Every part of the answer, including "is this row empty?", comes from it.',
    'For min and max, temporarily fill NaN with `+inf` or `-inf` using `np.where` and reduce along `axis=1`. Afterwards, overwrite empty rows with NaN.',
    'For the argmax, `np.argmax` on the `-inf`-filled matrix fails for a row like `[nan, -inf]` (it returns 0). Instead take `(valid & (X == maxs[:, None])).argmax(axis=1)`, whose first `True` is the first valid maximum, and set empty rows to -1.',
]

nan, inf = np.nan, np.inf
_r = np.random.default_rng(73)
_big = _r.normal(size=(8, 5)).round(1)
_big[_r.random(size=_big.shape) < 0.3] = nan
_big[3] = nan
TESTS = [
    (np.array([[3.0, nan, 7.0], [nan, nan, nan]]),),
    (np.array([[2.0, 5.0, 5.0, 1.0]]),),
    (np.array([[nan, -inf], [-inf, nan]]),),
    (np.array([[inf, nan, inf]]),),
    (np.array([[4.0]]),),
    (np.array([[-1e300, nan, 1e300], [0.0, -0.0, nan]]),),
    (_big,),
    (_r.integers(0, 4, size=(6, 6)).astype(float),),
]

EXPLANATION = r"""
## Why it works
Filling NaN with the identity element of a reduction (`+inf` for min, `-inf` for max) makes the missing cells unable to win, so ordinary `min` / `max` along `axis=1` give the NaN-ignoring answer for every row that has at least one value. All-NaN rows come out as `inf` / `-inf` and are fixed afterwards with the `empty` mask.

The argmax needs more care. The filled matrix cannot tell a real `-inf` from a NaN filled with `-inf`, so `argmax` of `[-inf(filled), -inf(real)]` returns column 0. Asking "which **valid** cells equal the row max?" removes the ambiguity, and `argmax` on a boolean array returns the first `True`, which is exactly the leftmost tie rule. For empty rows that boolean row is all False and `argmax` returns 0, which is why those rows are overwritten with -1.

## Complexity
$O(nd)$ time with a few boolean and float temporaries.

## Common bugs
- `np.nanargmax(X, axis=1)` on data with an all-NaN row raises `ValueError`.
- `np.nanmax` returns the right values but warns ("All-NaN slice encountered") on empty rows.
- Taking `argmax` of the `-inf`-filled matrix, which breaks for rows whose maximum is `-inf`.
- `0.0 == -0.0` is True, so in `[0.0, -0.0]` the first column wins the tie.

## Interview follow-ups
- How would you return the top-2 values per row ignoring NaN? (Fill NaN with `-inf`, `np.partition(-filled, 1, axis=1)[:, :2]`, then sort those two.)
- In pandas, `df.idxmax(axis=1, skipna=True)` handles this. What does it return for an all-NaN row? (In pandas 2.x it warns or raises depending on version, which is exactly why explicit rules like `-1` are worth writing down.)
"""
