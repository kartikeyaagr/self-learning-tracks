import numpy as np

META = dict(title='Turn count rows into distributions', tags=['Data Processing', 'Probability'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.03'], related=['column-scaling', 'outer-sum', 'normalize-columns'])

STATEMENT = r"""
Each row of a count matrix records how often something happened (a user's clicks per category, a word's co-occurrence counts). Dividing each row by its total turns it into a probability distribution.

Write `row_normalize(counts)`.

## Input
`counts`: a non-negative array (ints or floats) of shape `(n, d)`, with `d >= 1`.

## Output
A `float64` array of the same shape where row $i$ is

$$p_{ij} = \frac{c_{ij}}{\sum_k c_{ik}}$$

so every row sums to 1. A row whose total is 0 (no observations) becomes the **uniform** distribution `1/d` in every cell.

Use broadcasting, without a loop over rows.

## Example
`[[1, 3], [0, 0], [2, 2]]` becomes `[[0.25, 0.75], [0.5, 0.5], [0.5, 0.5]]`.
"""

STARTER = '''import numpy as np

def row_normalize(counts):
    """Divide each row by its sum; all-zero rows become uniform 1/d. float64, same shape."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def row_normalize(counts):
    c = np.asarray(counts, dtype=float)
    totals = c.sum(axis=1, keepdims=True)                 # (n, 1) so it broadcasts across columns
    out = np.full_like(c, 1.0 / c.shape[1])                # default: uniform
    np.divide(c, totals, out=out, where=totals > 0)        # only divide rows that have counts
    return out
'''

HINTS = [
    '`c.sum(axis=1)` has shape `(n,)`, and `(n, d) / (n,)` does not line the totals up with the rows. Keep the reduced axis.',
    '`c.sum(axis=1, keepdims=True)` has shape `(n, 1)`, which broadcasts across the columns of its row.',
    'Handle zero rows without dividing by zero: start from `np.full_like(c, 1/d)` and call `np.divide(c, totals, out=out, where=totals > 0)`.',
]

_r = np.random.default_rng(37)
TESTS = [
    (np.array([[1, 3], [0, 0], [2, 2]]),),
    (np.array([[5.0, 0.0, 0.0]]),),
    (np.array([[0, 0, 0, 0]]),),
    (np.array([[7]]),),
    (np.array([[1e-300, 3e-300], [1e300, 1e300]]),),
    (np.array([[0.2, 0.3, 0.5], [0.0, 0.0, 0.0], [10.0, 0.0, 30.0]]),),
    (_r.poisson(3.0, size=(6, 5)),),
    (_r.integers(0, 2, size=(8, 3)),),
]

EXPLANATION = r"""
## Why it works
`keepdims=True` keeps the summed axis as length 1, so `totals` has shape `(n, 1)`. Broadcasting matches trailing axes: `(n, d)` against `(n, 1)` stretches each row's total across that row's `d` columns. Without it, a `(n,)` vector lines up with the **columns** instead (and only runs at all when `n == d`, giving silently wrong answers).

`np.divide(..., out=out, where=totals > 0)` computes the division only where the mask is true and leaves the pre-filled uniform values elsewhere. No `0/0` is ever evaluated, so no warning and no NaN to clean up.

## Complexity
$O(nd)$ time, one output array.

## Common bugs
- `c / c.sum(axis=1)`, the missing-`keepdims` bug above.
- Dividing first and patching NaNs afterwards with `np.nan_to_num`. It works, but emits `RuntimeWarning`s and turns NaNs that were already in the input into zeros too.
- Normalising columns (`axis=0`) by accident. Check that each **row** sums to 1.

## Interview follow-ups
- What is additive (Laplace) smoothing, and how does it change this? (Add $\alpha$ to every count before normalising: $p_{ij} = (c_{ij}+\alpha)/(\sum_k c_{ik} + \alpha d)$. Zero rows then become uniform automatically.)
- How do you normalise rows of a huge sparse matrix? (Scale by a diagonal matrix of inverse row sums, `diags(1/totals) @ C`, which never densifies.)
"""
