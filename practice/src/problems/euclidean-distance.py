import numpy as np

META = dict(title='Pairwise Euclidean distance matrix', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:06.01'], related=['dot-product', 'mahalanobis-distance', 'rbf-kernel-matrix', 'kmeans-step'])

STATEMENT = r"""
Nearest-neighbour search, k-means and kernel methods all start from the distances between two sets of points. For rows $x_i$ of `X` and $y_j$ of `Y`:

$$D_{ij} = \lVert x_i - y_j \rVert_2 = \sqrt{\sum_{k} (x_{ik} - y_{jk})^2}$$

Write `pairwise_distances(X, Y)` that returns the full matrix $D$.

## Inputs
- `X`: float array `(n, d)`, `Y`: float array `(m, d)`, with $n, m \ge 1$.

## Output
- Float array of shape `(n, m)`. All entries are $\ge 0$; a point compared with itself gives exactly `0.0`.

## Requirements
- No Python loops: use broadcasting.
"""

STARTER = '''import numpy as np

def pairwise_distances(X, Y):
    """X: (n, d), Y: (m, d). Return the (n, m) matrix of Euclidean distances."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pairwise_distances(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    diff = X[:, None, :] - Y[None, :, :]        # (n, 1, d) - (1, m, d) -> (n, m, d)
    return np.sqrt((diff ** 2).sum(axis=-1))
'''

HINTS = [
    'You need every row of `X` paired with every row of `Y`. Can you insert axes so that subtraction broadcasts into an `(n, m, d)` block?',
    '`X[:, None, :] - Y[None, :, :]` has shape `(n, m, d)`. Square it and sum over the last axis.',
    '`np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))`.',
]

_r = np.random.default_rng(3)
TESTS = [
    (np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[0.0, 0.0], [6.0, 8.0]])),
    (np.array([[1.0, 2.0, 2.0]]), np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])),
    (np.array([[5.0]]), np.array([[-2.0], [5.0], [9.0]])),
    (np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])),
    (np.array([[1e6, -1e6], [0.0, 0.0]]), np.array([[1e6 + 3.0, -1e6 + 4.0]])),
    (_r.normal(size=(6, 4)), _r.normal(size=(3, 4))),
    (_r.normal(size=(4, 10)) * 50, _r.normal(size=(5, 10)) * 50),
]

EXPLANATION = r"""
## Why broadcasting works
`X[:, None, :]` has shape `(n, 1, d)` and `Y[None, :, :]` has shape `(1, m, d)`. NumPy stretches the size-1 axes, so entry `[i, j, :]` of the difference is $x_i - y_j$. Squaring and summing the last axis gives $\lVert x_i - y_j\rVert^2$ for every pair at once.

## The expansion trick
For large inputs the `(n, m, d)` intermediate is too big. Use

$$\lVert x - y\rVert^2 = \lVert x\rVert^2 + \lVert y\rVert^2 - 2\,x\cdot y$$

which needs one matrix product `X @ Y.T` and $O(nm)$ memory. It suffers from cancellation: for nearly identical points the result can come out as a tiny negative number, so clip at 0 before `sqrt`, and expect self-distances like `1e-8` instead of exactly 0. Test 5 shows the problem: large coordinates with a small separation lose digits in the expansion.

## Complexity
$O(nmd)$ time either way. Broadcasting uses $O(nmd)$ memory; the expansion uses $O(nm + (n+m)d)$.

## Common bugs
- Forgetting the `None` axes and subtracting `(n, d) - (m, d)`, which only works (wrongly) when $n = m$.
- `sqrt` of a slightly negative number from the expansion trick, giving `nan`.

## Interview follow-ups
- How would you find each point's nearest neighbour without materialising all of $D$? (Process `X` in chunks, take `argmin` per chunk; or use a KD-tree / approximate index.)
- Why does Euclidean distance become less informative in very high dimensions? (Distances concentrate: the ratio of the farthest to the nearest distance tends to 1.)
"""
