import numpy as np

META = dict(title='Matrix inverse by Gauss-Jordan elimination with partial pivoting', tags=['Linear Algebra'], diff='Medium',
            runtime='numpy', related=['solve-linear-system', 'lu-decomposition', 'moore-penrose-pseudoinverse', 'matrix-determinant'])

STATEMENT = r"""
Write `inverse(A, tol=1e-12)` that inverts a square matrix by Gauss-Jordan elimination on the augmented matrix $[A \mid I]$.

## Algorithm
For each column $k = 0, \dots, n-1$:
1. **Pivot:** among rows $k, \dots, n-1$, pick the row with the largest $|a_{ik}|$ (the first such row on ties) and swap it into row $k$.
2. If that largest value is below `tol`, the matrix is singular: return `None`.
3. Divide row $k$ by the pivot, then subtract multiples of row $k$ from every other row so that column $k$ is zero except for the 1 on the diagonal.

When all columns are done, the right half of the augmented matrix is $A^{-1}$.

Do not call `np.linalg.inv`, `solve` or `pinv`.
"""

STARTER = '''import numpy as np

def inverse(A, tol=1e-12):
    """A: (n, n). Return A^{-1}, or None if A is singular (pivot below tol)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def inverse(A, tol=1e-12):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    M = np.hstack([A.copy(), np.eye(n)])
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[p, k]) < tol:
            return None
        if p != k:
            M[[k, p]] = M[[p, k]]
        M[k] /= M[k, k]
        for i in range(n):
            if i != k:
                M[i] -= M[i, k] * M[k]
    return M[:, n:]
'''

HINTS = [
    'Build the augmented matrix with `np.hstack([A, np.eye(n)])`. Every row operation you apply to A you also apply to the identity half.',
    'Partial pivoting: `p = k + argmax(abs(M[k:, k]))`. Swapping in the largest available pivot keeps the division stable.',
    'Unlike Gaussian elimination, eliminate column k in rows above the pivot too, so the left half becomes the identity rather than upper-triangular.',
]

_r = np.random.default_rng(5)
TESTS = [
    (np.array([[2.0, 1.0], [1.0, 3.0]]),),
    (np.array([[0.0, 1.0], [1.0, 0.0]]),),
    (np.array([[1.0, 2.0], [2.0, 4.0]]),),
    (np.array([[4.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 4.0]]),),
    (_r.normal(size=(5, 5)) + 5 * np.eye(5),),
    (np.array([[1e-14, 1.0], [1.0, 1.0]]),),
    (np.array([[3.0]]),),
]

EXPLANATION = r"""
## Why it works
Each row operation is multiplication on the left by an elementary matrix $E_j$. If the operations turn $A$ into $I$, then $E_m \cdots E_1 A = I$, so $E_m \cdots E_1 = A^{-1}$. Applying the same operations to the identity half records exactly that product.

## Why pivot
Without pivoting, the test with $10^{-14}$ in the corner divides by a tiny number and the result is dominated by rounding error, even though the matrix is perfectly well conditioned. Choosing the largest pivot in the column keeps the multipliers at most 1 in size.

## Complexity
$n$ columns, each touching $n$ rows of length $2n$: about $2n^3$ floating-point operations, roughly three times the cost of an LU factorisation.

## In practice
You rarely need an explicit inverse. To solve $Ax = b$, factorise once and back-substitute; it is faster and more accurate than forming $A^{-1}b$.

## Interview follow-ups
- How would you compute the determinant during the same elimination? (Product of pivots, times $-1$ per swap.)
- What does the condition number tell you about the accuracy of the inverse?
"""
