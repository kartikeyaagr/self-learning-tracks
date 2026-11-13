import numpy as np

META = dict(title='Min-max scale each column', tags=['Feature Engineering', 'Data Processing'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.03'], related=['normalize-columns', 'row-scaling', 'winsorize'])

STATEMENT = r"""
Min-max scaling maps each feature (column) linearly so its smallest value lands on `new_min` and its largest on `new_max`:

$$x' = \text{new\_min} + \frac{x - \min_j}{\max_j - \min_j}\,(\text{new\_max} - \text{new\_min})$$

where $\min_j$ and $\max_j$ are taken over column $j$.

Write `minmax_scale(X, new_min=0.0, new_max=1.0)`.

## Input
- `X`: float array of shape `(n, d)` with `n, d >= 1` and no NaN.
- `new_min`, `new_max`: Python floats. `new_min > new_max` is allowed (it flips the direction).

## Output
A `float64` array of shape `(n, d)`. A **constant** column (where max equals min) maps entirely to `new_min`.

Use broadcasting with per-column statistics; no loop over columns.
"""

STARTER = '''import numpy as np

def minmax_scale(X, new_min=0.0, new_max=1.0):
    """Scale each column of X linearly to [new_min, new_max]; constant columns -> new_min."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def minmax_scale(X, new_min=0.0, new_max=1.0):
    X = np.asarray(X, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)           # (d,), broadcasts down the rows
    span = hi - lo
    unit = np.zeros_like(X)                          # constant columns stay at 0, i.e. new_min
    np.divide(X - lo, span, out=unit, where=span > 0)
    return new_min + unit * (new_max - new_min)
'''

HINTS = [
    'Column statistics come from reducing over the rows: `X.min(axis=0)` has shape `(d,)`, one value per column.',
    'A `(d,)` vector broadcasts against `(n, d)` along the rows, so `X - lo` subtracts each column\'s own minimum. No `keepdims` is needed for this axis.',
    'For constant columns, avoid `0/0`: start from zeros and use `np.divide(X - lo, span, out=unit, where=span > 0)`, then map `[0, 1]` onto `[new_min, new_max]`.',
]

_r = np.random.default_rng(43)
TESTS = [
    (np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 40.0]]),),
    (np.array([[5.0, 1.0], [5.0, 3.0]]), -1.0, 1.0),
    (np.array([[4.0, -4.0, 0.0]]),),
    (np.array([[0.0], [10.0], [5.0]]), 1.0, 0.0),
    (np.array([[1e-12, 1e12], [2e-12, -1e12], [3e-12, 0.0]]),),
    (_r.normal(size=(7, 4)) * np.array([1.0, 10.0, 100.0, 0.01]),),
    (_r.uniform(-50, 50, size=(5, 3)), 10.0, 20.0),
]

EXPLANATION = r"""
## Why it works
Reducing over `axis=0` gives one min and one max per column, as `(d,)` vectors. Broadcasting aligns trailing axes, so `X - lo` subtracts column $j$'s minimum from every entry of column $j$. Dividing by the span maps each column onto $[0, 1]$, and the final affine step stretches $[0, 1]$ onto $[\text{new\_min}, \text{new\_max}]$. With `new_min > new_max` the same formula reverses the order.

Constant columns have span 0 and no meaningful position, so the masked division leaves them at 0, which maps to `new_min`.

## Complexity
$O(nd)$ time, two passes for min and max plus one for the transform.

## Common bugs
- Computing a single global min and max with `X.min()`, which scales every column by the largest feature's range.
- Dividing by zero for constant columns and getting NaN.
- In production: computing min and max on the **test** set. Fit the statistics on training data and reuse them. Test values outside the training range will then fall outside $[0, 1]$, and that is expected.

## Interview follow-ups
- Why is min-max scaling sensitive to outliers, and what helps? (One extreme value squashes everyone else into a tiny interval. Clip or winsorize first, or use robust scaling with the median and IQR.)
- When do you prefer min-max over standardisation? (Bounded inputs such as pixel intensities, or models and activations that expect a fixed range.)
"""
