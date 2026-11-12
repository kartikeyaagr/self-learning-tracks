import numpy as np

META = dict(title='Rows inside a bounding box', tags=['Data Processing'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:02.03'], related=['boolean-masking', 'row-extremes'])

STATEMENT = r"""
Write `rows_in_box(X, lo, hi)` that keeps the rows of a data matrix whose every feature lies inside a per-feature range, and reports where those rows came from.

## Input
- `X`: float array of shape `(n, d)`.
- `lo`, `hi`: float arrays of shape `(d,)` with `lo <= hi`. Row `i` is **kept** when $lo_j \le X_{ij} \le hi_j$ for every column $j$ (both ends inclusive).

## Output
A tuple `(rows, idx)`:
- `rows`: the kept rows, in their original order, shape `(k, d)`.
- `idx`: an `int64` array of shape `(k,)` with the original row index of each kept row, increasing.

Use a boolean mask; do not loop over rows.

## Example
```
X  = [[1, 5], [3, 2], [0, 9]]
lo = [0, 0],  hi = [3, 6]
-> rows = [[1, 5], [3, 2]],  idx = [0, 1]
```
"""

STARTER = '''import numpy as np

def rows_in_box(X, lo, hi):
    """Return (rows (k, d), idx int64 (k,)) for rows with lo <= x <= hi in every column."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def rows_in_box(X, lo, hi):
    X = np.asarray(X, dtype=float)
    # (n, d) comparisons broadcast lo/hi across rows; .all(axis=1) requires every feature to pass.
    keep = ((X >= lo) & (X <= hi)).all(axis=1)
    return X[keep], np.flatnonzero(keep)
'''

HINTS = [
    '`X >= lo` compares each row against `lo` element-wise thanks to broadcasting, giving an `(n, d)` boolean array.',
    'Combine the two conditions with `&` (not `and`), wrapping each comparison in parentheses, then reduce each row with `.all(axis=1)`.',
    'With the row mask `keep`: `X[keep]` gives the rows and `np.flatnonzero(keep)` (or `np.nonzero(keep)[0]`) gives their indices.',
]

_r = np.random.default_rng(23)
TESTS = [
    (np.array([[1.0, 5.0], [3.0, 2.0], [0.0, 9.0]]), np.array([0.0, 0.0]), np.array([3.0, 6.0])),
    (np.array([[2.0, 2.0], [2.0, 2.0001]]), np.array([2.0, 2.0]), np.array([2.0, 2.0])),
    (np.array([[-1.0, -1.0, -1.0]]), np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])),
    (np.array([[5.0], [-5.0], [0.0], [5.0]]), np.array([-5.0]), np.array([5.0])),
    (np.array([[1e9, -1e9], [1e9, 1e9]]), np.array([0.0, -2e9]), np.array([2e9, 0.0])),
    (_r.normal(size=(20, 3)), np.array([-1.0, -1.5, -2.0]), np.array([1.0, 1.5, 2.0])),
    (_r.uniform(0, 10, size=(15, 2)), np.array([2.0, 3.0]), np.array([8.0, 7.0])),
]

EXPLANATION = r"""
## Why it works
`lo` has shape `(d,)` and `X` has shape `(n, d)`, so `X >= lo` broadcasts `lo` across every row and yields an `(n, d)` boolean matrix. `&` combines the lower and upper conditions element-wise, and `.all(axis=1)` collapses each row to one boolean: true only if every feature passed. That 1-D mask selects rows directly (`X[keep]` always returns a copy), and `np.flatnonzero` turns it into indices, which is useful for looking up labels or IDs stored elsewhere.

## Complexity
$O(nd)$ time, $O(nd)$ temporary booleans.

## Common bugs
- Writing `X >= lo and X <= hi`. Python's `and` asks for a single truth value of an array and raises `ValueError`.
- Forgetting the parentheses: `X >= lo & X <= hi` parses as `X >= (lo & X) <= hi`, because `&` binds tighter than comparisons.
- Using `.any(axis=1)`, which keeps rows where at least one feature is in range.

## Interview follow-ups
- How does this change if some features have no limit? (Use `-np.inf` / `np.inf` as bounds; the comparisons still work.)
- With NaN in `X`, what happens to such rows, and how would you keep rows whose only problem is a NaN? (Comparisons with NaN are False, so the row is dropped; OR in `np.isnan(X)` before the `all`.)
"""
