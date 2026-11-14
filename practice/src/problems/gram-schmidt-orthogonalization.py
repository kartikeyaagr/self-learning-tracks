import numpy as np

META = dict(title='Modified Gram-Schmidt with rank detection', tags=['Linear Algebra'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:06.13', 'mlmaths:06.07'], related=['vector-projection', 'qr-decomposition', 'matrix-rank'])

STATEMENT = r"""
Gram–Schmidt turns a list of vectors into an orthonormal basis for their span. Process the columns $a_1, a_2, \dots$ of $A$ in order; for each one, subtract its components along the basis vectors found so far, and keep what's left (normalised) if it is not negligibly small.

Write `gram_schmidt(A, tol=1e-10)` using the **modified** variant: subtract the projections one at a time from the running residual.

```
for each column a_j, in order:
    v = a_j
    for each q already in the basis, in order:
        v = v - (q . v) q
    if ||v|| > tol * max(1, ||a_j||):
        append v / ||v|| to the basis
```

## Inputs
- `A`: float array `(n, k)`, $n, k \ge 1$; the vectors are its **columns**.
- `tol`: float, relative threshold for dropping dependent columns.

## Output
- `Q`: float array `(n, r)` whose columns are the orthonormal basis vectors in the order they were found. $r$ is the number of kept columns ($0 \le r \le \min(n, k)$); for $r = 0$ return an array of shape `(n, 0)`.

The result is unique: no sign or ordering choices are left open.
"""

STARTER = '''import numpy as np

def gram_schmidt(A, tol=1e-10):
    """A: (n, k), vectors as columns. Return Q (n, r) with orthonormal columns spanning them (modified GS)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gram_schmidt(A, tol=1e-10):
    A = np.asarray(A, dtype=float)
    basis = []
    for j in range(A.shape[1]):
        a = A[:, j]
        v = a.copy()
        for q in basis:
            v -= (q @ v) * q                    # project the *current* residual: that's the "modified" part
        nv = np.linalg.norm(v)
        if nv > tol * max(1.0, np.linalg.norm(a)):
            basis.append(v / nv)
    if not basis:
        return np.zeros((A.shape[0], 0))
    return np.column_stack(basis)
'''

HINTS = [
    'Keep a Python list of the orthonormal vectors found so far. Each new column only needs to be compared against that list.',
    'Inside the loop, use `q @ v` with the **current** `v`, not with the original column. After subtracting, compare `norm(v)` against `tol * max(1, norm(a))` before normalising.',
    'Collect kept vectors in a list and return `np.column_stack(basis)`, or `np.zeros((n, 0))` if nothing was kept.',
]

_r = np.random.default_rng(37)
_B = _r.normal(size=(6, 3))
TESTS = [
    (np.array([[3.0, 1.0], [4.0, 2.0]]),),
    (np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 0.0]]),),
    (np.zeros((3, 2)),),
    (np.array([[2.0], [0.0], [0.0]]),),
    (np.array([[1.0, 0.0, 1.0, 5.0], [0.0, 1.0, 1.0, 7.0]]),),
    (np.array([[1.0, 1.0, 1.0], [1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]]),),
    (np.column_stack([_B, _B @ np.array([1.0, -2.0, 0.5])]),),
    (_r.normal(size=(5, 4)),),
]

EXPLANATION = r"""
## Why it works
After processing column $j$, the kept vectors are orthonormal and span the same space as $a_1, \dots, a_j$. Subtracting $(q\cdot v)\,q$ removes the component of $v$ along $q$ (a projection onto a unit vector), so the residual is orthogonal to every earlier $q$. If the residual is tiny relative to the column, $a_j$ was already in the span and adding it would inject pure round-off noise.

## Classical vs modified
Classical GS computes every coefficient against the **original** column, $q_i\cdot a_j$. In exact arithmetic it's identical. In floating point, rounding errors in earlier $q$'s are never corrected, and orthogonality decays with the condition number squared. Modified GS projects the running residual, so each step also removes the error left by the previous one. Test 6 is the Läuchli matrix: nearly parallel columns, where classical GS returns vectors far from orthogonal.

## Complexity
$O(nk\,r)$ time, $O(nr)$ memory.

## Common bugs
- Normalising before checking the norm, so a dependent column becomes a random unit vector of noise.
- Absolute tolerance only: a column of size $10^{12}$ leaves a residual of ~$10^{-4}$ after round-off, which would wrongly be kept.
- Returning rows instead of columns.

## Interview follow-ups
- How does Gram–Schmidt give a QR decomposition? (The coefficients $q_i\cdot v$ and norms form the upper-triangular $R$ with $A = QR$.)
- Why do libraries prefer Householder reflections? (They're backward stable: $Q$ is orthogonal to machine precision regardless of conditioning.)
"""
