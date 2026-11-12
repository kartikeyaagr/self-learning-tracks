import numpy as np

META = dict(title='Forward-fill gaps with a limit', tags=['Time Series', 'Data Processing'], diff='Hard', runtime='numpy',
            related=['filter-and-extract', 'row-extremes', 'tile-and-diff'])

STATEMENT = r"""
Sensor logs have gaps. A common repair is **forward fill**: replace each missing reading with the most recent valid reading above it. Filling forever is dangerous (a sensor that died yesterday should not report yesterday's value all day), so a `limit` caps how far a value may be carried.

Write `ffill(X, limit=None)` for a 2-D float array `X` of shape `(n, d)`, with `n >= 1`, that fills **down each column**. `NaN` marks a missing value.

## Rule for a NaN at row i of column j
Let `r` be the largest row index `<= i` such that `X[r, j]` is not NaN.
- If no such `r` exists (the column starts with NaNs), the cell stays `NaN`.
- Otherwise it becomes `X[r, j]` if `limit is None` or `i - r <= limit`, and stays `NaN` if `i - r > limit`.

Non-NaN values, including `inf` and `-inf`, are kept unchanged. `limit = 0` therefore changes nothing.

## Requirements
- Return a new `float64` array of shape `(n, d)` and leave `X` unmodified.
- Vectorise it: no Python loop over rows or columns.

## Example
Column `[1, nan, nan, 4, nan]` with `limit=1` becomes `[1, 1, nan, 4, 4]`.
"""

STARTER = '''import numpy as np

def ffill(X, limit=None):
    """X: (n, d) float with NaN gaps. Forward-fill down columns, carrying a value at most `limit` rows."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ffill(X, limit=None):
    X = np.asarray(X, dtype=float)
    rows = np.arange(X.shape[0])[:, None]                  # (n, 1), broadcasts across columns
    valid = ~np.isnan(X)
    # Row index of each valid cell, -1 for gaps; a running max gives "last valid row at or above".
    last = np.maximum.accumulate(np.where(valid, rows, -1), axis=0)
    filled = np.take_along_axis(X, np.maximum(last, 0), axis=0)
    ok = last >= 0                                          # a valid value exists above
    if limit is not None:
        ok &= rows - last <= limit                          # and it is close enough
    return np.where(ok, filled, np.nan)
'''

HINTS = [
    'Think in row indices, not values. For each cell, find the index of the last valid row at or above it. Once you have that index matrix, the values come from one gather.',
    'Put the row number where `X` is valid and `-1` where it is NaN, then take `np.maximum.accumulate(..., axis=0)`. The running max is exactly "the last valid row so far".',
    'Gather with `np.take_along_axis(X, np.maximum(last, 0), axis=0)`. Keep a cell only where `last >= 0` and, if a limit is given, `row - last <= limit`; set the rest to NaN with `np.where`.',
]

nan, inf = np.nan, np.inf
_r = np.random.default_rng(17)
_big = _r.normal(size=(12, 4))
_big[_r.random(size=_big.shape) < 0.45] = nan
TESTS = [
    (np.array([[1.0], [nan], [nan], [4.0], [nan]]), 1),
    (np.array([[nan, 2.0], [3.0, nan], [nan, nan]]),),
    (np.array([[1.0, nan], [nan, 5.0]]), 0),
    (np.array([[nan, nan, nan]]),),
    (np.array([[7.0], [nan], [nan], [nan]]), 2),
    (np.array([[inf], [nan], [-inf], [nan], [nan]]), 1),
    (np.array([[1e300, -2.0], [nan, nan], [nan, 3.0], [nan, nan]]),),
    (_big,),
    (_big, 1),
]

EXPLANATION = r"""
## Why it works
The value that fills a gap is "the value at the last valid row above". Finding that row is a prefix scan. Mark each valid cell with its row number and each gap with `-1`; then the running maximum down a column is the most recent valid row seen so far, because row numbers only increase. `np.maximum.accumulate` performs that scan for every column at once in C.

With the index matrix `last`, `take_along_axis` gathers `X[last[i, j], j]` for every cell. Two boolean masks then decide which cells keep the gathered value: `last >= 0` (something valid exists above) and `i - last <= limit` (the gap is short enough). Valid cells have `last == i`, distance 0, so they pass through unchanged.

## Complexity
$O(nd)$ time and a few $O(nd)$ temporaries. A Python loop over rows has the same complexity but runs one to two orders of magnitude slower.

## Common bugs
- Finding gaps with `X == np.nan`. NaN compares unequal to everything, itself included, so that mask is all False; use `np.isnan`.
- Gathering with `last` directly. `-1` is a valid index in NumPy (the last row), so leading gaps silently pick up the bottom value unless you clamp and mask.
- Measuring the limit as the length of the whole gap instead of the distance from the last valid row. The statement fills the first `limit` cells of a long gap.
- Treating `inf` as missing. Only `NaN` is missing here.

## Interview follow-ups
- How would you back-fill instead (use the next valid value below)? (Flip the rows, forward-fill, flip back, or use `np.minimum.accumulate` on a reversed index.)
- How does pandas' `df.ffill(limit=...)` behave on a `groupby`, and how would you reproduce it in NumPy? (Reset the scan at group boundaries, for example by treating each group's first row as a barrier.)
"""
