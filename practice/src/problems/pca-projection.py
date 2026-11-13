import numpy as np

META = dict(title='PCA from the covariance matrix', tags=['Linear Algebra', 'Feature Engineering'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:08.03', 'mlmaths:05.02'], related=['kmeans-step', 'cosine-similarity-matrix'])

STATEMENT = r"""
Principal component analysis finds the orthogonal directions of greatest variance. Implement it with an eigendecomposition:

1. Centre the data: $X_c = X - \bar{x}$.
2. Sample covariance: $\Sigma = \frac{1}{n-1} X_c^\top X_c$.
3. Eigen-decompose $\Sigma$, sort the eigenpairs by eigenvalue (largest first), and keep the top $k$ eigenvectors as rows of $W \in \mathbb{R}^{k\times d}$.
4. Project: $Z = X_c W^\top$.

Write `pca(X, k)` that returns `(Z, W, ratio)`, where `ratio[j]` is the fraction of total variance explained by component $j$.

## Sign convention
Eigenvectors are only defined up to sign. Flip each row of `W` so that **its entry with the largest absolute value is positive**. Your output must follow this to match.

## Shapes
`X`: `(n, d)` with `n >= 2`; `Z`: `(n, k)`; `W`: `(k, d)`; `ratio`: `(k,)`.
"""

STARTER = '''import numpy as np

def pca(X, k):
    """Return (Z (n, k), W (k, d), explained_variance_ratio (k,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pca(X, k):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(cov)                 # symmetric: real eigenvalues, ascending order
    order = np.argsort(vals)[::-1][:k]
    W = vecs[:, order].T                             # rows = components
    idx = np.abs(W).argmax(axis=1)
    W = W * np.sign(W[np.arange(k), idx])[:, None]   # make the largest-magnitude entry positive
    return Xc @ W.T, W, vals[order] / vals.sum()
'''

HINTS = [
    'Use `np.linalg.eigh`, not `eig`. The covariance matrix is symmetric, so `eigh` is faster and guarantees real, orthonormal eigenvectors. It returns eigenvalues in **ascending** order.',
    'Eigenvectors come back as **columns**. Reverse-sort the eigenvalue indices, take the first k columns, and transpose to get rows.',
    'For the sign rule: `idx = np.abs(W).argmax(axis=1)`, then multiply each row by `np.sign(W[row, idx])`.',
]

_r = np.random.default_rng(0)
A = _r.normal(size=(3, 3))
corr = _r.normal(size=(40, 3)) @ A + np.array([1.0, -2.0, 5.0])
TESTS = [
    (np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]), 1),
    (np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), 2),
    (corr, 2),
    (corr, 3),
    (_r.normal(size=(25, 5)) * np.array([5.0, 3.0, 1.0, 0.5, 0.1]), 3),
    (np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 3.0, 1.0], [2.0, 1.0, 0.5]]), 2),
]
META['tol'] = [1e-6, 1e-7]

EXPLANATION = r"""
## Why eigenvectors of the covariance
The variance of the data projected on a unit vector $w$ is $w^\top \Sigma w$. Maximising it subject to $\lVert w\rVert = 1$ (Lagrange multipliers) gives $\Sigma w = \lambda w$, and the variance captured is $\lambda$. So components are eigenvectors, and their eigenvalues are the variances along them.

## SVD route
In practice, use the SVD of $X_c = U S V^\top$: the rows of $V^\top$ are the components and $\lambda_j = s_j^2/(n-1)$. It never forms $X_c^\top X_c$, which squares the condition number. The two routes agree up to sign, which is why the sign convention is needed.

## Common bugs
- Forgetting to centre, so the first "component" just points at the mean.
- Using `eig` and getting complex output or unsorted pairs.
- Taking rows of the eigenvector matrix instead of columns.

## Interview follow-ups
- Should you standardise features first? (Yes, when units differ. Otherwise the largest-unit feature dominates.)
- How many components to keep? (Cumulative explained-variance threshold, or the downstream metric.)
"""
