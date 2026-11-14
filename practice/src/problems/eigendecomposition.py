import numpy as np

META = dict(title='Symmetric eigendecomposition and matrix powers', tags=['Linear Algebra'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:05.01', 'mlmaths:05.07'], related=['whitening-transform', 'pca', 'matrix-trace'])

STATEMENT = r"""
A real symmetric matrix has real eigenvalues and an orthonormal basis of eigenvectors:

$$S = V \Lambda V^\top, \qquad V^\top V = I$$

That makes functions of $S$ easy: $S^p = V \Lambda^p V^\top$, because the inner $V^\top V$ factors cancel. Repeated multiplication becomes one decomposition plus a scalar power per eigenvalue.

Write `sym_eig(S, p)`.

## Inputs
- `S`: symmetric float array `(n, n)`, $n \ge 1$, with **distinct** eigenvalues.
- `p`: int. If `p < 0`, you may assume no eigenvalue is zero.

## Output
A tuple `(vals, V, S_p)`:
- `vals`: float array `(n,)`, eigenvalues in **descending** order.
- `V`: float array `(n, n)` whose column $i$ is a unit eigenvector for `vals[i]`.
- `S_p`: float array `(n, n)`, $S^p$ computed from the decomposition. $S^0 = I$.

## Sign convention
Each eigenvector is only defined up to sign. Flip column $i$ so that its entry with the largest absolute value is positive. The tests avoid eigenvectors whose two largest entries tie in absolute value.
"""

STARTER = '''import numpy as np

def sym_eig(S, p):
    """S: symmetric (n, n). Return (vals descending, V with sign-fixed unit columns, S**p via V diag(vals**p) V^T)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sym_eig(S, p):
    S = np.asarray(S, dtype=float)
    vals, V = np.linalg.eigh(S)                # ascending eigenvalues, orthonormal eigenvector columns
    vals, V = vals[::-1], V[:, ::-1]
    idx = np.abs(V).argmax(axis=0)             # first max per column
    V = V * np.sign(V[idx, np.arange(len(vals))])
    S_p = (V * vals ** float(p)) @ V.T         # V diag(vals^p) V^T without building the diagonal
    return vals, V, S_p
'''

HINTS = [
    'Use `np.linalg.eigh` for symmetric matrices. It returns eigenvalues in **ascending** order and eigenvectors as **columns**.',
    'Reverse both the values and the columns. For the sign rule, find `np.abs(V).argmax(axis=0)` and multiply each column by the sign of that entry.',
    'Power: `(V * vals**float(p)) @ V.T`. Multiplying `V` by a row vector scales its columns, which is the same as `V @ np.diag(vals**p)`. Cast `p` to float so negative powers of integer-valued floats work.',
]

_r = np.random.default_rng(59)
_M = _r.normal(size=(4, 4))
TESTS = [
    (np.array([[2.0, 0.0], [0.0, 5.0]]), 2),
    (np.array([[2.0, 1.0], [1.0, 3.0]]), 3),
    (np.array([[4.0, 1.0], [1.0, 3.0]]), -1),
    (np.array([[7.0]]), 0),
    (np.array([[1.0, 2.0, 0.0], [2.0, -1.0, 1.0], [0.0, 1.0, 3.0]]), 1),
    (np.array([[0.0, 3.0], [3.0, 1.0]]), 4),
    (_M + _M.T, 3),
    (_M @ _M.T + np.eye(4), -2),
]

EXPLANATION = r"""
## Why it works
For symmetric $S$, the spectral theorem gives $S = V\Lambda V^\top$ with orthogonal $V$. Then $S^2 = V\Lambda V^\top V\Lambda V^\top = V\Lambda^2 V^\top$, and by induction $S^p = V\Lambda^p V^\top$; for $p = -1$ it is the inverse, as long as no $\lambda_i = 0$. The same recipe gives $\exp(S)$, $\sqrt{S}$ for PSD $S$, and so on: apply the function to the eigenvalues.

Test 3 uses $p = -1$, so `S_p` is the inverse; check it against `np.linalg.inv`. Test 4 uses $p = 0$, which must give the identity even for a $1\times 1$ matrix.

## Why the sign rule avoids ties
A matrix like $\begin{pmatrix}2&1\\1&2\end{pmatrix}$ has eigenvectors $(1, \pm 1)/\sqrt2$, whose entries are equal in absolute value. In floating point one of them comes out an ulp larger, and which one depends on the LAPACK build, so "largest entry positive" is not well defined there. Any sign rule has some ambiguous inputs; the fix is to report sign-free quantities (like `S_p`, or $v v^\top$) when you can.

## Complexity
$O(n^3)$ for the decomposition and $O(n^3)$ to recombine, independent of $p$. Repeated squaring would be $O(n^3 \log p)$ and doesn't handle negative or fractional powers.

## Common bugs
- `np.linalg.eig` on a symmetric matrix: unsorted output and possibly complex dtype from round-off.
- Taking rows of `V` instead of columns as the eigenvectors.
- Sorting `vals` but not reordering `V` the same way.
- `vals ** p` with an integer array and negative `p` raises `ValueError`.

## Interview follow-ups
- How does power iteration find the top eigenvector, and what controls its speed? (Repeatedly apply $S$ and normalise; convergence rate $|\lambda_2/\lambda_1|$.)
- What do the eigenvalues of a Hessian tell you at a critical point? (All positive: minimum; all negative: maximum; mixed signs: saddle.)
"""
