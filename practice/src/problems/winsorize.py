import numpy as np

META = dict(title='Winsorize columns with your own percentiles', tags=['Feature Engineering', 'Probability and Statistics'],
            diff='Hard', runtime='numpy', lessons=['ieng:04.05'],
            related=['normalize-columns', 'column-scaling', 'aggregation-functions'])

STATEMENT = r"""
**Winsorizing** limits outliers without deleting rows: every value below a low percentile is raised to that percentile, and every value above a high percentile is lowered to it. Implement it per column, ignoring missing values, **including the percentile computation itself**.

Write `winsorize(X, lower=5.0, upper=95.0)` that returns `(W, lo, hi, n_clipped)`.

## Input
- `X`: float array `(n, d)`, possibly containing `NaN`. Every column has at least one non-NaN value.
- `lower`, `upper`: percentiles with `0 <= lower <= upper <= 100`.

## Percentile rule (linear interpolation)
For one column, sort its $k$ non-NaN values: $v_0 \le v_1 \le \dots \le v_{k-1}$. For percentile $p$ let $h = (k-1)\,p/100$ and $f = \lfloor h \rfloor$. Then

$$q(p) = v_f + (h - f)\,(v_{f+1} - v_f)$$

with $v_{f+1}$ read as $v_f$ when $f = k-1$. (This matches `np.nanpercentile`'s default, which you can use to check your work, but write the computation yourself with sorting and indexing.)

## Output
- `lo`, `hi`: `float64` arrays `(d,)`, the `lower` and `upper` percentiles of each column.
- `W`: `float64` array `(n, d)`, `X` with every non-NaN value clipped to `[lo[j], hi[j]]`. NaN stays NaN.
- `n_clipped`: `int64` array `(d,)`, how many values in each column were strictly outside `[lo[j], hi[j]]` before clipping.

No loop over columns.
"""

STARTER = '''import numpy as np

def winsorize(X, lower=5.0, upper=95.0):
    """Return (W, lo, hi, n_clipped): per-column NaN-aware percentile clipping."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _col_percentile(S, k, p):
    """S: columns sorted ascending with NaN last; k: (d,) valid counts. Linear-interpolated percentile p."""
    h = (k - 1) * (p / 100.0)
    f = np.floor(h).astype(np.int64)
    c = np.minimum(f + 1, k - 1)                       # upper neighbour, clamped at the last valid value
    vf = np.take_along_axis(S, f[None, :], axis=0)[0]
    vc = np.take_along_axis(S, c[None, :], axis=0)[0]
    return vf + (h - f) * (vc - vf)

def winsorize(X, lower=5.0, upper=95.0):
    X = np.asarray(X, dtype=float)
    S = np.sort(X, axis=0)                             # np.sort puts NaN at the end of each column
    k = (~np.isnan(X)).sum(axis=0)                     # number of valid values per column
    lo, hi = _col_percentile(S, k, lower), _col_percentile(S, k, upper)
    n_clipped = ((X < lo) | (X > hi)).sum(axis=0)      # comparisons with NaN are False
    W = np.clip(X, lo, hi)                             # clip leaves NaN as NaN
    return W, lo, hi, n_clipped.astype(np.int64)
'''

HINTS = [
    '`np.sort(X, axis=0)` sorts every column independently and puts NaN at the **end**. So the first `k[j]` entries of sorted column `j` are its valid values in order, where `k = (~np.isnan(X)).sum(axis=0)`.',
    'Now the percentile is index arithmetic on vectors of length `d`: `h = (k - 1) * p / 100`, `f = floor(h)` and `c = min(f + 1, k - 1)`. Fetch `S[f[j], j]` for every column at once with `np.take_along_axis(S, f[None, :], axis=0)`.',
    'Interpolate with `vf + (h - f) * (vc - vf)`. Then `np.clip(X, lo, hi)` broadcasts the `(d,)` bounds down the rows, and `((X < lo) | (X > hi)).sum(axis=0)` counts what was clipped.',
]

nan = np.nan
_r = np.random.default_rng(79)
_heavy = _r.standard_t(2, size=(40, 3)) * np.array([1.0, 10.0, 0.1])
_gaps = _r.normal(size=(15, 4))
_gaps[_r.random(size=_gaps.shape) < 0.25] = nan
_gaps[1:, 0] = nan
TESTS = [
    (np.array([[1.0], [2.0], [3.0], [4.0], [100.0]]), 0.0, 75.0),
    (np.array([[10.0, 1.0], [20.0, nan], [30.0, 3.0]]), 25.0, 75.0),
    (np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]),),
    (np.array([[7.0, -1.0], [nan, 1.0]]), 10.0, 90.0),
    (np.array([[-1e12], [0.0], [1e12]]), 50.0, 50.0),
    (np.arange(21.0).reshape(21, 1), 5.0, 95.0),
    (_heavy,),
    (_heavy, 1.0, 99.0),
    (_gaps, 10.0, 80.0),
]

EXPLANATION = r"""
## Why it works
Percentiles are order statistics, so the core step is sorting. `np.sort(..., axis=0)` sorts every column independently and sends NaN to the end. That means "the $k_j$ valid values of column $j$, ascending" is simply the top $k_j$ entries of the sorted column, and no per-column filtering is needed. The interpolation formula then needs two indices per column, `f` and `c`. Because they differ between columns, `np.take_along_axis` gathers `S[f[j], j]` for all columns in one call.

Clipping uses broadcasting: `lo` and `hi` are `(d,)` and stretch down the rows. `np.clip` returns NaN for NaN inputs, and NaN comparisons are False, so missing values are neither changed nor counted.

## Complexity
$O(d\, n \log n)$ for the column sorts, $O(nd)$ for the rest. A selection algorithm (`np.partition`) can find percentiles in $O(nd)$ average time, but needs fixed indices per call.

## Common bugs
- Sorting the whole matrix flat instead of per column.
- Using `n` instead of the per-column count `k` of valid values, which shifts every percentile in columns with gaps.
- Reading index `f + 1` without clamping: for `p = 100` that is out of bounds, or it silently lands on a NaN.
- Counting values that are **equal** to a bound as clipped. They do not change.

## Interview follow-ups
- Winsorizing vs trimming: what does each do to the mean and the sample size? (Winsorizing keeps $n$ and replaces extremes; trimming drops them. Both give robust estimates of location.)
- In a train/test pipeline, which data should the bounds come from? (The training set only, then reused for validation, test and production, exactly like scaling statistics.)
"""
