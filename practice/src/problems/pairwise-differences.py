import numpy as np

META = dict(title='Manhattan distance matrix', tags=['Data Processing', 'Clustering'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.04', 'air-fanout-v1:02.03'], related=['outer-sum', 'cosine-similarity-matrix', 'kmeans-step'])

STATEMENT = r"""
The Manhattan (L1) distance between two points is the sum of the absolute differences of their coordinates:

$$D_{ij} = \sum_{k=1}^{d} \lvert A_{ik} - B_{jk} \rvert$$

Write `l1_distances(A, B)` that returns the full distance matrix between two point sets.

## Input
- `A`: float array of shape `(n, d)`; `B`: float array of shape `(m, d)`, with `n, m, d >= 1`.

## Output
A `float64` array `D` of shape `(n, m)`.

Use broadcasting, with no Python loops. (The `(n, m, d)` intermediate is fine at these sizes.)

## Example
`A = [[0, 0], [1, 2]]`, `B = [[1, 1], [3, 0], [0, 2]]` gives

```
[[2., 3., 2.],
 [1., 4., 1.]]
```
"""

STARTER = '''import numpy as np

def l1_distances(A, B):
    """A: (n, d), B: (m, d). Return the (n, m) matrix of L1 distances."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def l1_distances(A, B):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    diff = A[:, None, :] - B[None, :, :]       # (n, 1, d) - (1, m, d) -> (n, m, d)
    return np.abs(diff).sum(axis=-1)
'''

HINTS = [
    'You need every row of `A` paired with every row of `B`, coordinate by coordinate. That is a 3-D array of differences, shape `(n, m, d)`.',
    'Insert a new axis in a different place on each side: `A[:, None, :]` is `(n, 1, d)` and `B[None, :, :]` is `(1, m, d)`.',
    'Subtract, take `np.abs`, and sum over the last axis (the coordinates) to get `(n, m)`.',
]

_r = np.random.default_rng(41)
TESTS = [
    (np.array([[0.0, 0.0], [1.0, 2.0]]), np.array([[1.0, 1.0], [3.0, 0.0], [0.0, 2.0]])),
    (np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 2.0, 3.0]])),
    (np.array([[5.0]]), np.array([[-5.0], [2.5], [5.0]])),
    (np.array([[-1.0, -1.0], [-3.0, 4.0]]), np.array([[2.0, -2.0]])),
    (np.array([[1e10, -1e10]]), np.array([[-1e10, 1e10], [1e10, -1e10]])),
    (_r.normal(size=(5, 3)), _r.normal(size=(4, 3))),
    (_r.integers(-10, 10, size=(6, 6)).astype(float), _r.integers(-10, 10, size=(2, 6)).astype(float)),
]

EXPLANATION = r"""
## Why it works
Broadcasting aligns shapes from the right. `(n, 1, d)` against `(1, m, d)`: the last axes agree, and each size-1 axis stretches, giving `diff[i, j, k] = A[i, k] - B[j, k]` for every pair. Taking absolute values and summing over the coordinate axis leaves the `(n, m)` table of distances.

## Complexity
$O(nmd)$ time and, because of the intermediate, $O(nmd)$ memory. For large inputs, process `A` in row chunks, or use `scipy.spatial.distance.cdist(A, B, 'cityblock')`, which never builds the 3-D array.

## Common bugs
- `A[:, None] - B` without the trailing `:` is the same thing, but `A[None] - B[:, None]` gives `(m, n, d)` and a transposed answer.
- Summing over the wrong axis (`axis=1`), which sums over points instead of coordinates.
- Squaring instead of taking the absolute value, which gives squared Euclidean distance.

## Interview follow-ups
- Squared Euclidean distances can avoid the 3-D intermediate. How? ($\lVert a-b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2a\cdot b$: one matrix product and two outer sums. Clamp at 0 for rounding error.)
- Why can't the L1 distance use the same matrix-product trick? (The absolute value does not expand into products; it is not an inner-product norm.)
"""
