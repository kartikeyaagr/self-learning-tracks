import numpy as np

META = dict(title='Mahalanobis distance for outlier scoring', tags=['Probability and Statistics', 'Linear Algebra'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:11.21', 'mlmaths:11.25'], related=['euclidean-distance', 'cholesky-decomposition', 'whitening-transform'])

STATEMENT = r"""
Euclidean distance treats every direction alike. When features are correlated or on different scales, a point can be "close" in Euclidean terms yet wildly unusual for the data. The Mahalanobis distance measures distance in units of the data's own spread:

$$d_M(q) = \sqrt{(q - \mu)^\top \Sigma^{-1} (q - \mu)}$$

Write `mahalanobis(X, Q)` that fits $\mu$ and $\Sigma$ on reference data `X` and scores every row of `Q`.

## Inputs
- `X`: float array `(n, d)` of reference data, with $n > d \ge 1$ and a non-singular sample covariance.
- `Q`: float array `(m, d)` of query points, $m \ge 1$.

## Definitions
- $\mu$ = column means of `X`.
- $\Sigma = \frac{1}{n-1}(X - \mu)^\top (X - \mu)$, the sample covariance (`ddof=1`).

## Output
- Float array `(m,)` of distances, one per query row.

## Requirements
- No loops over query points.
- Don't form $\Sigma^{-1}$ explicitly: use a Cholesky factor or `np.linalg.solve`.
"""

STARTER = '''import numpy as np

def mahalanobis(X, Q):
    """X: (n, d) reference data, Q: (m, d) queries. Return (m,) Mahalanobis distances (ddof=1 covariance)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def mahalanobis(X, Q):
    X = np.asarray(X, dtype=float)
    Q = np.asarray(Q, dtype=float)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = Xc.T @ Xc / (len(X) - 1)
    L = np.linalg.cholesky(cov)                 # cov = L L^T
    # (q-mu)^T cov^{-1} (q-mu) = ||L^{-1}(q-mu)||^2: one triangular-style solve for all queries at once.
    Z = np.linalg.solve(L, (Q - mu).T)          # (d, m)
    return np.sqrt((Z ** 2).sum(axis=0))
'''

HINTS = [
    'Fit first: mean and `ddof=1` covariance of `X`. Then centre the queries with the **reference** mean, not their own.',
    'If $\\Sigma = LL^\\top$, then $(q-\\mu)^\\top\\Sigma^{-1}(q-\\mu) = \\lVert L^{-1}(q-\\mu)\\rVert^2$. Solving $Lz = q - \\mu$ for all queries at once means passing a `(d, m)` right-hand side.',
    '`L = np.linalg.cholesky(cov)`; `Z = np.linalg.solve(L, (Q - mu).T)`; return `np.sqrt((Z**2).sum(axis=0))`.',
]

_r = np.random.default_rng(79)
_C = np.array([[1.0, 0.0, 0.0], [1.5, 0.5, 0.0], [-0.5, 0.2, 2.0]])
_X = _r.normal(size=(200, 3)) @ _C.T + np.array([1.0, -1.0, 0.0])
TESTS = [
    (np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [2.0, 4.0]]), np.array([[1.0, 2.0], [2.0, 2.0], [1.0, 4.0]])),
    (np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), np.array([[3.0], [0.0], [8.0]])),
    (np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.1], [3.0, 2.9], [4.0, 4.0]]), np.array([[2.0, 2.0], [2.0, 0.0], [0.0, 2.0]])),
    (np.array([[1e6, 0.0], [1e6 + 2.0, 1.0], [1e6 + 1.0, 3.0], [1e6 + 3.0, 2.0]]), np.array([[1e6 + 1.5, 1.5]])),
    (_X, _X[:5]),
    (_X, np.array([[1.0, -1.0, 0.0], [3.0, 3.0, 3.0], [-4.0, 2.0, 1.0]])),
    (_r.normal(size=(30, 4)) * np.array([1.0, 10.0, 0.1, 100.0]), _r.normal(size=(6, 4))),
]

EXPLANATION = r"""
## Why it works
Write $\Sigma = LL^\top$ and $z = L^{-1}(q - \mu)$. Then $z^\top z = (q-\mu)^\top L^{-\top}L^{-1}(q-\mu) = (q-\mu)^\top\Sigma^{-1}(q-\mu)$. The map $x \mapsto L^{-1}(x - \mu)$ **whitens** the data: it has identity covariance afterwards, so Mahalanobis distance is plain Euclidean distance in whitened coordinates.

Test 3 makes the point. The data lies along the diagonal; $(2, 0)$ and $(0, 2)$ are the same Euclidean distance from the mean $(2, 2)$ as each other, but both are far off the data's line, so their Mahalanobis distances are large, while $(2, 2)$ scores 0.

## Complexity
$O(nd^2)$ for the covariance, $O(d^3)$ for Cholesky, $O(md^2)$ for the solve.

## Common bugs
- Centring queries with their own mean.
- `ddof=0` (`np.cov(..., bias=True)`), which scales distances by $\sqrt{(n-1)/n}$.
- `np.cov(X)` without `rowvar=False`: it treats rows as variables and returns an $n\times n$ matrix.
- `np.linalg.inv(cov)`: works, but is slower and less accurate than a factorisation.

## Interview follow-ups
- Under a Gaussian model, what distribution does $d_M^2$ follow, and how would you set an outlier threshold? ($\chi^2_d$; flag points above its 99th percentile.)
- What if the covariance is singular (e.g. $n \le d$)? (Shrink it, $\Sigma + \lambda I$ or Ledoit–Wolf, or use the pseudoinverse within the data's subspace.)
"""
