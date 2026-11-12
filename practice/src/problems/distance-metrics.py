import numpy as np

META = dict(title='Pairwise distance matrices', tags=['Classic ML', 'Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.04', 'air-fanout-v1:02.03'], related=['cosine-similarity-matrix', 'knn-classifier'])

STATEMENT = r"""
Many algorithms (k-NN, clustering, retrieval) start from a matrix of distances between two sets of points. Write `pairwise_distances(A, B, metric)` that returns $D_{ij} = \mathrm{dist}(a_i, b_j)$.

| `metric` | $\mathrm{dist}(a, b)$ |
| --- | --- |
| `'euclidean'` | $\sqrt{\sum_k (a_k - b_k)^2}$ |
| `'manhattan'` | $\sum_k \lvert a_k - b_k\rvert$ |
| `'chebyshev'` | $\max_k \lvert a_k - b_k\rvert$ |
| `'cosine'` | $1 - \dfrac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}$ |

## Rules
- `A`: float `(m, d)`; `B`: float `(n, d)`; $d \ge 1$. Return a float array `(m, n)`.
- Cosine: if either vector is all zeros, the distance is `1.0`.
- Identical rows must give a distance of **0** (within `1e-8`) for every metric. Careful: the expansion $\lVert a\rVert^2 + \lVert b\rVert^2 - 2a^\top b$ can come out slightly negative or leave a residue whose square root is far above `1e-8`.
- Use broadcasting or matrix products, no Python loops over pairs.
"""

STARTER = '''import numpy as np

def pairwise_distances(A, B, metric):
    """(m, n) matrix of 'euclidean', 'manhattan', 'chebyshev' or 'cosine' distances."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pairwise_distances(A, B, metric):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if metric == 'cosine':
        na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
        denom = np.outer(na, nb)
        sim = np.divide(A @ B.T, denom, out=np.zeros_like(denom), where=denom > 0)   # zero vector -> sim 0
        return 1.0 - np.clip(sim, -1.0, 1.0)
    diff = np.abs(A[:, None, :] - B[None, :, :])          # (m, n, d): exact zeros for identical rows
    if metric == 'euclidean':
        return np.sqrt((diff ** 2).sum(axis=-1))
    if metric == 'manhattan':
        return diff.sum(axis=-1)
    return diff.max(axis=-1)                                # chebyshev
'''

HINTS = [
    '`A[:, None, :] - B[None, :, :]` broadcasts to `(m, n, d)`: every pair\'s difference vector at once. Three of the four metrics are a reduction over its last axis.',
    'Cosine needs only `A @ B.T` and the row norms. Guard zero norms with `np.divide(..., where=denom > 0)` so they give similarity 0, i.e. distance 1.',
    'Clip the cosine similarity to $[-1, 1]$: rounding can push it just past 1 for parallel vectors, giving a tiny negative distance.',
]

_r = np.random.default_rng(151)
_A = _r.normal(size=(6, 4))
_B = np.vstack([_A[:2], _r.normal(size=(3, 4))])
TESTS = [
    (np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[3.0, 4.0]]), 'euclidean'),
    (np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[3.0, 4.0]]), 'manhattan'),
    (np.array([[1.0, -2.0, 3.0]]), np.array([[4.0, 0.0, 3.0], [1.0, -2.0, 3.0]]), 'chebyshev'),
    (np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 2.0]]), np.array([[0.0, 3.0], [1.0, 1.0], [-1.0, 0.0]]), 'cosine'),
    (np.array([[1e8, 1e8 + 1.0]]), np.array([[1e8, 1e8 + 1.0], [1e8 + 3.0, 1e8 - 3.0]]), 'euclidean'),
    (np.array([[5.0]]), np.array([[-2.0], [5.0]]), 'manhattan'),
    (_A, _B, 'euclidean'),
    (_A, _B, 'cosine'),
    (_A * 1e3, _B * 1e3, 'chebyshev'),
]

EXPLANATION = r"""
## How the metrics relate
Euclidean, Manhattan and Chebyshev are the $L_p$ norms of the difference vector for $p = 2, 1, \infty$. They always satisfy $L_\infty \le L_2 \le L_1$. Cosine distance ignores magnitude and only compares direction, which suits embeddings and term-frequency vectors where length reflects document size rather than meaning. It isn't a true metric (it fails the triangle inequality), but $\sqrt{2(1-\cos)}$ is the Euclidean distance between the normalised vectors.

## The expansion trick and its trap
$\lVert a - b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2a^\top b$ turns the work into one matrix product, $O(mnd)$ flops at BLAS speed and only $O(mn)$ memory. But with coordinates around $10^8$ (test 5), each term is about $10^{16}$ and the subtraction cancels catastrophically: identical rows can come out visibly nonzero, or even negative, instead of 0. Libraries clip at 0 and sometimes recompute small distances exactly. The broadcasted difference used here is exact for identical rows, at the cost of an $(m, n, d)$ temporary.

## Complexity
$O(mnd)$ time for every metric; $O(mnd)$ memory for the broadcast version, $O(mn)$ for the matrix-product version.

## Common bugs
- Dividing by a zero norm in cosine and returning `nan`.
- Forgetting `np.abs` for Manhattan/Chebyshev.
- Broadcasting `A[:, None] - B[None]` on huge inputs and running out of memory: chunk over rows of `A`.

## Interview follow-ups
- Why do all pairwise distances concentrate (look similar) in high dimensions, and what does that mean for k-NN?
- When would you choose Mahalanobis distance instead, and how does it relate to whitening the data first?
"""
