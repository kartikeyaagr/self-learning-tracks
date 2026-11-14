import numpy as np

META = dict(title='Measure over-smoothing', tags=['Neural Networks', 'Linear Algebra'], diff='Medium', runtime='numpy',
            related=['normalized-adjacency', 'gcn-layer'])

STATEMENT = r"""
Repeated averaging over neighbours makes all node representations alike. Measure it.

Write `oversmoothing(A_hat, X, ks)`. For each depth `k` in `ks`, compute $H_k = \hat A^k X$ (with $\hat A^0 = I$) and its **mean average distance**:
$$\text{MAD}(H) = \frac{1}{n(n-1)}\sum_{i \ne j}\Big(1 - \frac{h_i \cdot h_j}{\lVert h_i\rVert\,\lVert h_j\rVert + 10^{-12}}\Big).$$

Return a list of MAD values, one per entry of `ks`.
"""

STARTER = '''import numpy as np

def oversmoothing(A_hat, X, ks):
    """Mean pairwise cosine distance of A_hat^k X for each k."""
    A, X = np.asarray(A_hat, float), np.asarray(X, float)
    out = []
    for k in ks:
        H = k * (A @ X)
        D = np.linalg.norm(H[:, None] - H[None, :], axis=2)
        out.append(float(D.mean()))
    return out
'''

SOLUTION = '''import numpy as np

def _mad(H):
    nr = np.linalg.norm(H, axis=1)
    C = (H @ H.T) / (nr[:, None] * nr[None, :] + 1e-12)
    n = len(H)
    D = 1.0 - C
    return float((D.sum() - np.trace(D)) / (n * (n - 1)))

def oversmoothing(A_hat, X, ks):
    A, X = np.asarray(A_hat, float), np.asarray(X, float)
    return [_mad(np.linalg.matrix_power(A, k) @ X) for k in ks]
'''

HINTS = [
    'np.linalg.matrix_power(A, 0) is the identity, so k = 0 gives the raw features.',
    'Exclude the diagonal (i = j) from the average; it is 0 for nonzero rows.',
    'Cosine distance ignores scale, which is what you want: smoothing is about directions converging, not norms shrinking.',
]

def _ahat(edges, n):
    A = np.eye(n)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    d = A.sum(1) ** -0.5
    return d[:, None] * A * d[None, :]

_r = np.random.default_rng(7)
_ring = [(i, (i + 1) % 8) for i in range(8)]
_two = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]

TESTS = [
    (np.eye(3), [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0, 1, 5]),
    (_ahat([(0, 1)], 2), [[1.0, 0.0], [0.0, 1.0]], [0, 1, 2]),
    (_ahat(_ring, 8), _r.normal(size=(8, 4)), [0, 1, 2, 4, 8, 32]),
    (_ahat(_two, 6), _r.normal(size=(6, 3)), [0, 2, 8, 64]),
    (_ahat(_ring + [(0, 4), (2, 6)], 8), np.abs(_r.normal(size=(8, 5))), [0, 1, 3, 10]),
]

EXPLANATION = r"""
## Why representations collapse
$\hat A$ has largest eigenvalue 1 with eigenvector proportional to $\sqrt{d_i}$. Every other eigenvalue has magnitude below 1 on a connected, non-bipartite graph with self-loops, so $\hat A^k X$ converges to a matrix whose rows are all proportional to the same vector: every node's direction becomes identical and MAD goes to 0. Deep GCNs therefore lose the ability to tell nodes apart, which is one reason most GCNs have only two or three layers.

## What the tests show
- With $\hat A = I$ (the first test) nothing smooths.
- On a ring, MAD falls with depth and approaches 0 by $k = 32$.
- With two disconnected triangles (the fourth test) each component smooths separately, so MAD stops at a non-zero floor: the two components keep different directions.

## Fixes
Residual connections, normalisation layers designed for graphs (PairNorm), mixing in the initial features at every layer (APPNP, GCNII) and dropping edges during training all slow the collapse.

## Interview follow-ups
- Over-smoothing and over-squashing are often confused. What is the difference?
"""
