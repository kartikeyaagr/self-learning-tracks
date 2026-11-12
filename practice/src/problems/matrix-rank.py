import numpy as np

META = dict(title='Numerical rank from singular values', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:04.04', 'mlmaths:07.02'], related=['svd-components', 'matrix-determinant', 'low-rank-approximation'])

STATEMENT = r"""
The rank of a matrix is the number of linearly independent columns (equivalently rows). In floating point, an exactly dependent column usually comes out *almost* dependent, so rank is computed by counting singular values that are clearly non-zero.

Write `numerical_rank(A, tol=None)`.

## Inputs
- `A`: float array of shape `(m, n)`, $m, n \ge 1$.
- `tol`: a non-negative float, or `None`.

## Rule
Let $\sigma_1 \ge \sigma_2 \ge \dots$ be the singular values of $A$.
- If `tol` is `None`, use $\text{tol} = \max(m, n)\cdot \varepsilon \cdot \sigma_1$, where $\varepsilon$ is `np.finfo(float).eps` ($\approx 2.2\times 10^{-16}$).
- The rank is the number of $\sigma_i$ **strictly greater** than `tol`.

## Output
- A Python `int`. The zero matrix has rank 0.
"""

STARTER = '''import numpy as np

def numerical_rank(A, tol=None):
    """A: (m, n). Count singular values > tol (default max(m, n) * eps * largest singular value)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def numerical_rank(A, tol=None):
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)        # singular values only, sorted descending
    if tol is None:
        tol = max(A.shape) * np.finfo(float).eps * s[0]
    return int((s > tol).sum())
'''

HINTS = [
    'Rank = number of non-zero singular values. In floating point "non-zero" needs a threshold.',
    '`np.linalg.svd(A, compute_uv=False)` returns just the singular values, largest first, so `s[0]` is $\\sigma_1$.',
    'Default `tol = max(A.shape) * np.finfo(float).eps * s[0]`, then `int((s > tol).sum())`. For the zero matrix `tol` is 0 and nothing is strictly greater, giving 0.',
]

_r = np.random.default_rng(29)
_U, _V = _r.normal(size=(8, 3)), _r.normal(size=(3, 6))
TESTS = [
    (np.array([[1.0, 2.0], [2.0, 4.0]]),),
    (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),),
    (np.zeros((3, 4)),),
    (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),),
    (np.array([[3.0, -1.0, 2.0, 0.5]]),),
    (_U @ _V,),
    (np.diag([1.0, 1e-3, 1e-9]),),
    (np.diag([1.0, 1e-3, 1e-9]), 1e-6),
    (_r.normal(size=(5, 7)),),
]

EXPLANATION = r"""
## Why singular values
The SVD $A = U\Sigma V^\top$ writes $A$ as a sum of $r$ rank-1 pieces $\sigma_i u_i v_i^\top$ with orthonormal $u_i$, $v_i$; the rank is the number of non-zero $\sigma_i$. Rounding error perturbs a matrix by about $\varepsilon\,\lVert A\rVert_2 = \varepsilon\,\sigma_1$ per operation, so singular values below a small multiple of that are indistinguishable from 0. The default tolerance is the one NumPy's `matrix_rank` uses.

Test 4 is the classic singular $3\times 3$ (its third singular value is around $10^{-16}$, not 0). Tests 7 and 8 show that rank depends on the tolerance: $10^{-9}$ is far above round-off, but below a tolerance chosen for noisy data.

## Complexity
$O(mn\min(m,n))$ for the singular values. Gaussian elimination is cheaper but less reliable for near-dependent columns.

## Common bugs
- Counting `s > 0` or `s != 0`: test 4 then reports rank 3.
- An absolute tolerance like `1e-10` regardless of scale: it calls `1e-12 * I` rank 0.
- Using `>=` instead of `>`, which makes the zero matrix rank $\min(m, n)$.

## Interview follow-ups
- What does the rank tell you about $Ax = b$? (Rank-nullity: the solution set, if non-empty, has dimension $n - r$.)
- How is rank used in LoRA? (The weight update is constrained to rank $r$ by writing it as $BA$ with inner dimension $r$.)
"""
