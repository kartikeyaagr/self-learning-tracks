import numpy as np

META = dict(title='Solve Ax = b or report singular', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:03.03', 'mlmaths:03.04'], related=['lu-decomposition', 'matrix-rank', 'least-squares-solution'])

STATEMENT = r"""
A square system $Ax = b$ has exactly one solution when $A$ is invertible. Solvers like `np.linalg.solve` factorise $A$ (LU with pivoting) instead of forming $A^{-1}$, which is faster and more accurate. But `solve` only raises an error for *exactly* singular matrices; a nearly singular one silently returns garbage.

Write `solve_system(A, b)`.

## Inputs
- `A`: float array `(n, n)`, $n \ge 1$.
- `b`: float array `(n,)` (one right-hand side) or `(n, k)` (k right-hand sides, solved together).

## Output
- If `np.linalg.matrix_rank(A)` (default tolerance) is less than $n$, return `None`.
- Otherwise return `x` with the same shape as `b` such that $Ax = b$.

## Requirements
Do not compute `np.linalg.inv(A)`.
"""

STARTER = '''import numpy as np

def solve_system(A, b):
    """A: (n, n), b: (n,) or (n, k). Return x with A x = b, or None if A is (numerically) singular."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def solve_system(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.linalg.matrix_rank(A) < A.shape[0]:
        return None                          # singular or numerically singular: no unique solution
    return np.linalg.solve(A, b)             # LU with partial pivoting; handles (n,) and (n, k) b
'''

HINTS = [
    'Decide singularity before solving. The statement tells you exactly which test to use.',
    '`np.linalg.matrix_rank(A) < n` means no unique solution. Otherwise `np.linalg.solve` does the rest and accepts a matrix of right-hand sides.',
    '`if np.linalg.matrix_rank(A) < A.shape[0]: return None`, then `return np.linalg.solve(A, b)`.',
]

_r = np.random.default_rng(41)
TESTS = [
    (np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([3.0, 5.0])),
    (np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0])),
    (np.array([[4.0]]), np.array([2.0])),
    (np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[1.0, 2.0], [3.0, 4.0]])),
    (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), np.array([1.0, 1.0, 1.0])),
    (np.array([[1e-3, 1.0], [1.0, 1.0]]), np.array([1.0, 2.0])),
    (_r.normal(size=(5, 5)) + 5 * np.eye(5), _r.normal(size=5)),
    (_r.normal(size=(4, 4)) + 4 * np.eye(4), _r.normal(size=(4, 3))),
]

EXPLANATION = r"""
## Why rank first
`np.linalg.solve` raises `LinAlgError` only when a pivot is exactly zero. The $3\times 3$ matrix in test 5 is singular in exact arithmetic, but in floating point its LU pivots come out as tiny non-zero numbers, so `solve` returns a vector with entries around $10^{15}$ instead of raising. Counting singular values above a scaled tolerance catches it.

## Why not the inverse
`inv(A) @ b` does roughly 3 times the work of one LU solve and is less accurate, because it multiplies the rounding errors of the inverse by $b$. For many right-hand sides, pass them all as columns of `b`: the factorisation is done once.

Test 6 needs pivoting: eliminating with the $10^{-3}$ pivot is fine in double precision, but with a pivot like $10^{-17}$ it would destroy the answer. `solve` swaps rows to use the largest pivot.

## Complexity
$O(n^3)$ for the SVD in `matrix_rank` and $O(n^3 + n^2 k)$ for the solve.

## Common bugs
- `try: solve(...) except LinAlgError: return None`, which misses numerically singular matrices.
- `det(A) == 0` as the test: determinants underflow or overflow for perfectly fine matrices.
- Flattening a `(n, k)` right-hand side.

## Interview follow-ups
- What is the condition number and what does it tell you? ($\kappa = \sigma_{\max}/\sigma_{\min}$; you can lose about $\log_{10}\kappa$ digits of accuracy in $x$.)
- What should you do when $A$ is singular but you still need an answer? (Least squares / pseudoinverse for the minimum-norm solution, or regularise with $A + \lambda I$.)
"""
