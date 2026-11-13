import numpy as np

META = dict(title='Pseudoinverse and minimum-norm solutions', tags=['Linear Algebra'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:08.01', 'mlmaths:07.02'], related=['least-squares-solution', 'svd-components', 'matrix-rank'])

STATEMENT = r"""
The Moore–Penrose pseudoinverse $A^+$ generalises the inverse to any matrix, square or not, full rank or not. From the SVD $A = U\Sigma V^\top$:

$$A^+ = V\,\Sigma^+\,U^\top, \qquad \Sigma^+_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > \text{cutoff} \\ 0 & \text{otherwise} \end{cases}$$

$x^\star = A^+ b$ is the **minimum-norm least-squares** solution: among all $x$ that minimise $\lVert Ax - b\rVert_2$, it has the smallest $\lVert x\rVert_2$.

Write `pinv_solve(A, b, rtol=1e-10)` without `np.linalg.pinv` or `np.linalg.lstsq`.

## Inputs
- `A`: float array `(m, n)`, $m, n \ge 1$, any rank (it may be all zeros).
- `b`: float array `(m,)`.
- `rtol`: singular values with $\sigma_i \le \texttt{rtol}\cdot\sigma_1$ are treated as zero. If $\sigma_1 = 0$, all are.

## Output
A tuple `(A_pinv, x, rank, residual)`:
- `A_pinv`: float array `(n, m)`.
- `x`: float array `(n,)`, equal to `A_pinv @ b`.
- `rank`: int, the number of singular values kept.
- `residual`: float, $\lVert Ax - b\rVert_2$.
"""

STARTER = '''import numpy as np

def pinv_solve(A, b, rtol=1e-10):
    """A: (m, n), b: (m,). Return (A_pinv (n, m), x = A_pinv @ b, rank, ||A x - b||)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pinv_solve(A, b, rtol=1e-10):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)    # U (m, p), s (p,), Vt (p, n), p = min(m, n)
    keep = s > rtol * s[0]                              # all False for the zero matrix (0 > 0 is False)
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]                         # never divide by the discarded (tiny) values
    A_pinv = (Vt.T * s_inv) @ U.T                       # V diag(s_inv) U^T
    x = A_pinv @ b
    return A_pinv, x, int(keep.sum()), float(np.linalg.norm(A @ x - b))
'''

HINTS = [
    'Take the reduced SVD. The pseudoinverse inverts the singular values you trust and zeros the rest, then swaps the roles of $U$ and $V$.',
    'Build `s_inv` with zeros, fill `1/s` only where `s > rtol * s[0]`, then `A_pinv = (Vt.T * s_inv) @ U.T`. Multiplying by a row vector scales columns, so no diagonal matrix is needed.',
    'Watch the zero matrix: `s[0] == 0` makes the cutoff 0 and `s > 0` is all False, so `A_pinv` is all zeros and `x = 0`. The residual is then `norm(b)`.',
]

_r = np.random.default_rng(73)
_B = _r.normal(size=(6, 2)) @ _r.normal(size=(2, 4))
TESTS = [
    (np.array([[2.0, 0.0], [0.0, 0.0]]), np.array([4.0, 3.0])),
    (np.array([[1.0, 1.0]]), np.array([2.0])),
    (np.array([[1.0], [1.0], [1.0]]), np.array([1.0, 2.0, 6.0])),
    (np.zeros((2, 3)), np.array([1.0, -1.0])),
    (np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), np.array([1.0, 0.0, 1.0])),
    (np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([5.0, 5.0])),
    (np.array([[1.0, 0.0], [0.0, 1e-12]]), np.array([1.0, 1.0])),
    (_B, _r.normal(size=6)),
    (_r.normal(size=(3, 5)), _r.normal(size=3)),
]

EXPLANATION = r"""
## Why it is the minimum-norm least-squares solution
In the rotated coordinates $c = U^\top b$ and $z = V^\top x$, the residual is $\sum_i (\sigma_i z_i - c_i)^2$ plus a part that no $x$ can touch. For $\sigma_i > 0$ the best choice is $z_i = c_i/\sigma_i$. For $\sigma_i = 0$, $z_i$ doesn't affect the residual at all, so choosing $z_i = 0$ minimises $\lVert x\rVert = \lVert z\rVert$. That is exactly $x = V\Sigma^+U^\top b$.

## The four cases in the tests
- Square invertible (test 6): $A^+ = A^{-1}$ and the residual is 0.
- Wide (tests 2, 9): infinitely many exact solutions; $A^+b$ is the shortest one. For $[1, 1]x = 2$ it's $(1, 1)$, not $(2, 0)$.
- Tall (test 3): no exact solution; $A^+b$ is the least-squares fit, here the mean of $b$.
- Rank-deficient (tests 1, 5, 8): both effects at once. Test 7 shows why the cutoff matters: $\sigma_2 = 10^{-12}$ is below `rtol`, so it is treated as zero instead of producing a $10^{12}$ entry.

## Complexity
$O(mn\min(m,n))$ for the SVD, $O(mn\min(m,n))$ to form $A^+$. For a single solve, `Vt.T @ (s_inv * (U.T @ b))` skips building $A^+$.

## Common bugs
- `1 / s` on every singular value, giving `inf` or huge entries for rank-deficient input.
- An absolute cutoff: the right threshold scales with $\sigma_1$.
- Getting the shape backwards: $A^+$ is $n \times m$.
- `np.linalg.inv(A.T @ A) @ A.T`: only valid for full column rank, and it squares the condition number.

## Interview follow-ups
- Which four Penrose conditions define $A^+$ uniquely? ($AA^+A = A$, $A^+AA^+ = A^+$, and both $AA^+$ and $A^+A$ symmetric.)
- How does ridge regression relate? ($(A^\top A + \lambda I)^{-1}A^\top$ replaces $1/\sigma$ by $\sigma/(\sigma^2 + \lambda)$, a smooth version of the hard cutoff, and tends to $A^+$ as $\lambda \to 0$.)
"""
