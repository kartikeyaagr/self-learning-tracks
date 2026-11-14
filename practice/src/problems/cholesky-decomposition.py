import numpy as np

META = dict(title='Cholesky factorisation, solve and log-determinant', tags=['Linear Algebra'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:07.01', 'mlmaths:03.28'], related=['lu-decomposition', 'mahalanobis-distance', 'solve-linear-system'])

STATEMENT = r"""
A symmetric positive definite (SPD) matrix factors as $A = LL^\top$ with $L$ lower-triangular and $L_{ii} > 0$. It is half the cost of LU, needs no pivoting, and it doubles as a test for positive definiteness: the factorisation succeeds exactly when $A$ is SPD. Gaussian processes and Gaussian log-likelihoods use it for both the solve and the log-determinant.

Write `cholesky_solve(A, b)` **from scratch** (no `np.linalg.cholesky`, `solve`, `inv`, `det` or SciPy).

## Algorithm
1. For $j = 0, \dots, n-1$: compute the pivot $d = A_{jj} - \sum_{k<j} L_{jk}^2$. If $d \le 0$, $A$ is not positive definite: **return `None`**. Otherwise $L_{jj} = \sqrt d$, and for $i > j$: $L_{ij} = \big(A_{ij} - \sum_{k<j} L_{ik}L_{jk}\big)/L_{jj}$.
2. Forward substitution: solve $Ly = b$.
3. Back substitution: solve $L^\top x = y$.
4. $\log\det A = 2\sum_i \log L_{ii}$.

## Inputs
- `A`: symmetric float array `(n, n)`, $n \ge 1$ (only read the lower triangle).
- `b`: float array `(n,)` or `(n, k)`.

## Output
- `None` if a pivot is $\le 0$.
- Otherwise a tuple `(L, x, logdet)`: `L` is `(n, n)` with exact zeros above the diagonal, `x` has the shape of `b` and solves $Ax = b$, and `logdet` is a Python float.
"""

STARTER = '''import numpy as np

def cholesky_solve(A, b):
    """A: SPD (n, n), b: (n,) or (n, k). Return (L, x, logdet) with A = L L^T and A x = b, or None if not PD."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def cholesky_solve(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        d = A[j, j] - L[j, :j] @ L[j, :j]
        if d <= 0:
            return None                                       # not positive definite
        L[j, j] = np.sqrt(d)
        # Whole column below the diagonal at once: L[i, j] for all i > j.
        L[j + 1:, j] = (A[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]

    y = np.zeros_like(b)
    for i in range(n):                                        # L y = b, top to bottom
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):                            # L^T x = y, bottom to top; row i of L^T is column i of L
        x[i] = (y[i] - L[i + 1:, i] @ x[i + 1:]) / L[i, i]
    return L, x, float(2.0 * np.log(np.diag(L)).sum())
'''

HINTS = [
    'Build `L` one column at a time. The pivot for column $j$ only needs row $j$ of `L` to the left of the diagonal; check it before taking the square root.',
    'Forward substitution: `y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]`. Back substitution with $L^\\top$ uses **column** $i$ of `L` below the diagonal: `x[i] = (y[i] - L[i+1:, i] @ x[i+1:]) / L[i, i]`. Both lines work unchanged when `b` is `(n, k)`.',
    'The determinant of a triangular matrix is the product of its diagonal, and $\\det A = (\\det L)^2$, so `logdet = 2 * np.log(np.diag(L)).sum()`. Summing logs never overflows, unlike the product.',
]

_r = np.random.default_rng(71)
_M = _r.normal(size=(5, 5))
_S = _M @ _M.T + 0.5 * np.eye(5)
_N = _r.normal(size=(30, 30))
TESTS = [
    (np.array([[4.0, 2.0], [2.0, 3.0]]), np.array([2.0, 1.0])),
    (np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([1.0, 1.0])),
    (np.array([[9.0]]), np.array([3.0])),
    (np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([1.0, 0.0])),
    (np.array([[-1.0]]), np.array([1.0])),
    (np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]), np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])),
    (_S, _r.normal(size=5)),
    (_N @ _N.T + 30 * np.eye(30), _r.normal(size=30)),
]

EXPLANATION = r"""
## Why it works
Write $A = LL^\top$ entrywise: $A_{ij} = \sum_{k \le j} L_{ik}L_{jk}$ for $i \ge j$. The $k = j$ term is the only unknown in each equation, so solving for it column by column gives the recursion. The pivot $d$ is the Schur complement of the leading block: it is positive at every step if and only if every leading principal minor is positive, which (Sylvester's criterion) is exactly positive definiteness.

Tests 2, 4 and 5 are not PD: $[[1,2],[2,1]]$ has eigenvalue $-1$, $[[1,1],[1,1]]$ is only semi-definite (pivot exactly 0), and $[-1]$ is negative. Test 6 is a classic SPD matrix with integer factor $L = [[2,0,0],[6,1,0],[-8,5,3]]$.

## The log-determinant
$\det A = \prod_i L_{ii}^2$. For the $30\times 30$ test, $\det A$ is around $10^{50}$; at a few hundred dimensions the product would overflow. Summing logs is what Gaussian likelihoods use.

## Complexity
$\tfrac13 n^3$ flops for the factorisation (half of LU), $O(n^2 k)$ for the two triangular solves.

## Common bugs
- Taking `np.sqrt` of a negative pivot, which gives `nan` with a warning instead of a clean "not PD".
- Back substitution with `L[i, i+1:]` (row of $L$, all zeros) instead of `L[i+1:, i]` (row of $L^\top$).
- Computing `np.log(np.linalg.det(A))`, which overflows to `inf` for large matrices.

## Interview follow-ups
- How do you sample from $\mathcal{N}(\mu, \Sigma)$? ($x = \mu + Lz$ with $z \sim \mathcal{N}(0, I)$ and $\Sigma = LL^\top$.)
- A covariance matrix fails Cholesky by a hair because of round-off. What do you do? (Add jitter $\epsilon I$ to the diagonal, the standard fix in GP code.)
"""
