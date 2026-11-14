import numpy as np

META = dict(title='LU decomposition with partial pivoting', tags=['Linear Algebra'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:03.16'], related=['solve-linear-system', 'matrix-determinant', 'cholesky-decomposition'])

STATEMENT = r"""
Gaussian elimination, recorded as matrices, is the factorisation

$$PA = LU$$

where $P$ is a permutation matrix, $L$ is unit lower-triangular (ones on the diagonal) holding the elimination multipliers, and $U$ is upper-triangular. Once you have it, solving $Ax = b$ is two cheap triangular solves.

Write `lu(A)` for a square matrix using **partial pivoting**.

## Algorithm (follow it exactly)
For each column $k = 0, \dots, n-1$:
1. Among rows $k, \dots, n-1$, pick the row $p$ whose entry in column $k$ has the **largest absolute value**. On ties take the **smallest** row index.
2. Swap rows $k$ and $p$ in the working matrix, in $P$, and in the already-computed part of $L$ (columns $0..k-1$).
3. If the pivot is exactly `0.0`, the column is already eliminated: skip to the next $k$.
4. Otherwise, for each row $i > k$: $L_{ik} = U_{ik}/U_{kk}$, then subtract $L_{ik}$ times row $k$ from row $i$.

## Input / output
- `A`: float array `(n, n)`, $n \ge 1$. It may be singular.
- Return a tuple `(P, L, U)` of float `(n, n)` arrays with `P @ A == L @ U` (up to rounding).
"""

STARTER = '''import numpy as np

def lu(A):
    """A: (n, n). Return (P, L, U) with P @ A = L @ U, partial pivoting, L unit lower-triangular."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def lu(A):
    U = np.array(A, dtype=float)          # working copy; becomes U in place
    n = U.shape[0]
    P, L = np.eye(n), np.eye(n)
    for k in range(n):
        p = k + int(np.argmax(np.abs(U[k:, k])))    # argmax returns the first max: smallest index on ties
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]           # only the multipliers already computed move
        if U[k, k] == 0.0:
            continue                                # nothing to eliminate below a zero column
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
        U[k + 1:, k] = 0.0                          # exact zeros below the pivot
    return P, L, U
'''

HINTS = [
    'Work on a float copy of `A` that you turn into `U` in place. Start `P` and `L` as identity matrices.',
    'When you swap rows $k$ and $p$, swap them in `U` and `P`, but in `L` only swap the first $k$ columns. The diagonal ones must stay put.',
    'Vectorise the inner loop: `L[k+1:, k] = U[k+1:, k] / U[k, k]` and `U[k+1:, k:] -= np.outer(L[k+1:, k], U[k, k:])`. Use `np.argmax(np.abs(U[k:, k]))` for the pivot.',
]

_r = np.random.default_rng(43)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 4.0]]),),
    (np.array([[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]]),),
    (np.array([[5.0]]),),
    (np.array([[0.0, 1.0], [1.0, 0.0]]),),
    (np.array([[1.0, 2.0], [-1.0, 3.0]]),),
    (np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]]),),
    (np.zeros((2, 2)),),
    (np.array([[1e-20, 1.0], [1.0, 1.0]]),),
    (_r.normal(size=(5, 5)),),
]

EXPLANATION = r"""
## Why it works
Each elimination step left-multiplies by a matrix $I - \ell_k e_k^\top$ that subtracts multiples of row $k$. Their inverses are $I + \ell_k e_k^\top$, and multiplying those inverses together just places each multiplier $\ell_{ik}$ into position $(i, k)$: that is $L$. Row swaps are collected in $P$. When a later swap happens, the multipliers already stored in $L$ belong to the swapped rows, so they move too, but the unit diagonal doesn't.

## Why pivot
Test 8 has pivot $10^{-20}$. Without swapping, the multiplier is $10^{20}$ and $U_{22} = 1 - 10^{20}$ rounds to $-10^{20}$, losing the original $1$ entirely: $LU$ no longer reproduces $A$. Picking the largest pivot keeps every $|L_{ik}| \le 1$.

## Complexity
$\tfrac{2}{3}n^3$ flops for the factorisation; each subsequent solve is $O(n^2)$. $\det A = \det(P)\prod_i U_{ii}$, with $\det(P) = \pm 1$ from the number of swaps.

## Common bugs
- Swapping whole rows of `L`, which moves the diagonal ones off the diagonal.
- Returning the convention $A = PLU$ (SciPy's) instead of $PA = LU$; they differ by a transpose of $P$.
- Dividing by a zero pivot in a singular matrix (tests 6 and 7) instead of skipping the column.

## Interview follow-ups
- How do you solve $Ax = b$ once you have $PA = LU$? (Forward-substitute $Ly = Pb$, then back-substitute $Ux = y$.)
- When is pivoting unnecessary? (For symmetric positive definite or diagonally dominant matrices; for SPD use Cholesky, which is twice as cheap.)
"""
