import numpy as np

META = dict(title='Projection matrix onto a column space', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:06.13', 'mlmaths:08.01'], related=['vector-projection', 'least-squares-solution', 'qr-decomposition'])

STATEMENT = r"""
The orthogonal projection onto the column space of a full-column-rank matrix $A \in \mathbb{R}^{m\times n}$ is

$$P = A\,(A^\top A)^{-1} A^\top$$

$Pb$ is the closest point to $b$ that can be written as $Ax$. In linear regression $P$ is the "hat matrix": it maps targets $y$ to fitted values $\hat y$.

Write `projection_matrix(A)`.

## Input
- `A`: float array `(m, n)` with $m \ge n \ge 1$ and linearly independent columns.

## Output
- A tuple `(P, rank)`: `P` is the float `(m, m)` projection matrix and `rank` is the Python int $\operatorname{tr}(P)$ rounded to the nearest integer. (The trace of a projection equals the dimension of the subspace it projects onto.)

## Requirements
Do not call `np.linalg.inv`. Use `np.linalg.solve`, or orthonormalise the columns first.
"""

STARTER = '''import numpy as np

def projection_matrix(A):
    """A: (m, n), full column rank. Return (P (m, m) projecting onto col(A), round(trace(P)))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def projection_matrix(A):
    A = np.asarray(A, dtype=float)
    # Q has orthonormal columns spanning col(A), so P = A (A^T A)^{-1} A^T simplifies to Q Q^T.
    Q, _ = np.linalg.qr(A)
    P = Q @ Q.T
    return P, int(round(np.trace(P)))
'''

HINTS = [
    'If the columns were orthonormal, $A^\\top A = I$ and the formula collapses. Can you get orthonormal columns with the same span?',
    '`Q, _ = np.linalg.qr(A)` gives $Q$ with orthonormal columns and $\\operatorname{col}(Q) = \\operatorname{col}(A)$. Then $P = QQ^\\top$, and the sign choices inside `qr` cancel.',
    'Alternatively `P = A @ np.linalg.solve(A.T @ A, A.T)`. Either way, `rank = int(round(np.trace(P)))`.',
]

_r = np.random.default_rng(61)
TESTS = [
    (np.array([[1.0], [0.0]]),),
    (np.array([[1.0], [1.0]]),),
    (np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),),
    (np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]),),
    (np.array([[2.0, 0.0], [0.0, 3.0]]),),
    (np.array([[1e5], [1e5], [0.0]]),),
    (_r.normal(size=(6, 3)),),
    (_r.normal(size=(5, 1)),),
]

EXPLANATION = r"""
## Why it works
$Pb$ must lie in the column space, $Pb = A\hat x$, with the residual orthogonal to every column: $A^\top(b - A\hat x) = 0$. Solving gives $\hat x = (A^\top A)^{-1}A^\top b$, so $P = A(A^\top A)^{-1}A^\top$. With $A = QR$ and $R$ invertible, the $R$ factors cancel and $P = QQ^\top$.

## Properties to check
- Idempotent: $P^2 = P$ (projecting twice changes nothing).
- Symmetric: $P^\top = P$ (that's what makes it an *orthogonal* projection).
- Eigenvalues are 0 or 1, so $\operatorname{tr}(P) = n$, and $I - P$ projects onto the orthogonal complement.

Test 4 is the design matrix of a straight-line fit (intercept plus slope): $P$ is the hat matrix, and $\operatorname{tr}(P) = 2$ is the model's degrees of freedom. Test 5 spans all of $\mathbb{R}^2$, so $P = I$.

## Complexity
$O(mn^2)$ for the QR, $O(m^2 n)$ to form $QQ^\top$. Storing $P$ costs $O(m^2)$: for large $m$, keep $Q$ and apply $Q(Q^\top b)$ instead.

## Common bugs
- `np.linalg.inv(A.T @ A)`: forming $A^\top A$ squares the condition number, so nearly collinear features lose half their digits.
- `np.linalg.qr(A, mode='complete')`: the full $m\times m$ $Q$ gives $QQ^\top = I$.

## Interview follow-ups
- What are the diagonal entries of the hat matrix used for? (Leverage: $P_{ii}$ near 1 means point $i$ pulls the fit strongly towards itself.)
- How does ridge regression change $P$? ($A(A^\top A + \lambda I)^{-1}A^\top$ is no longer a projection: its eigenvalues shrink into $(0, 1)$ and the trace becomes the effective degrees of freedom.)
"""
