import numpy as np

META = dict(title='Householder QR from scratch', tags=['Linear Algebra'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:06.03', 'mlmaths:06.02'], related=['gram-schmidt-orthogonalization', 'orthogonal-projection-matrix', 'least-squares-solution'])

STATEMENT = r"""
Every matrix $A \in \mathbb{R}^{m\times n}$ with $m \ge n$ factors as $A = QR$, with $Q \in \mathbb{R}^{m\times n}$ having orthonormal columns and $R \in \mathbb{R}^{n\times n}$ upper-triangular. Libraries compute it with **Householder reflections**, which stay orthogonal to machine precision even when Gram–Schmidt doesn't.

A Householder reflector $H = I - 2\,vv^\top/(v^\top v)$ mirrors space across the hyperplane orthogonal to $v$. Choosing $v = x + \operatorname{sign}(x_1)\lVert x\rVert e_1$ sends $x$ to $-\operatorname{sign}(x_1)\lVert x\rVert e_1$, zeroing everything below the first entry.

Write `householder_qr(A)`.

## Algorithm
1. Copy $A$ into $R$. For $k = 0, \dots, n-1$, let $x = R_{k:, k}$. Take $\operatorname{sign}(0) = +1$. If $\lVert x\rVert = 0$ skip this column; otherwise build $v$ as above and apply $H$ to rows $k{:}$ of $R$ (and accumulate it into $Q$).
2. Return the **reduced** factors: the first $n$ columns of $Q$ and the top $n\times n$ block of $R$.

## Uniqueness convention
For a full-column-rank $A$ the factorisation is unique once the diagonal of $R$ is positive. After step 2, for every $i$ with $R_{ii} < 0$, negate row $i$ of $R$ and column $i$ of $Q$.

## Input / output
- `A`: float array `(m, n)`, $m \ge n \ge 1$, full column rank.
- Return `(Q, R)`: float arrays `(m, n)` and `(n, n)`, with $R$ exactly zero below the diagonal and $R_{ii} > 0$.
"""

STARTER = '''import numpy as np

def householder_qr(A):
    """A: (m, n), m >= n, full column rank. Return reduced (Q (m, n), R (n, n)) with diag(R) > 0."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def householder_qr(A):
    R = np.array(A, dtype=float)
    m, n = R.shape
    Q = np.eye(m)
    for k in range(n):
        x = R[k:, k]
        nx = np.linalg.norm(x)
        if nx == 0:
            continue
        v = x.copy()
        v[0] += (1.0 if x[0] >= 0 else -1.0) * nx      # same sign as x[0]: avoids cancellation
        v /= np.linalg.norm(v)
        # Apply H = I - 2 v v^T to the trailing rows of R, and accumulate Q = H_0 H_1 ... (right-multiply).
        R[k:, :] -= 2.0 * np.outer(v, v @ R[k:, :])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    Q, R = Q[:, :n], np.triu(R[:n])                     # reduced factors; clean round-off below the diagonal
    signs = np.where(np.diag(R) < 0, -1.0, 1.0)
    return Q * signs, R * signs[:, None]                # flip column i of Q and row i of R together
'''

HINTS = [
    'Keep two arrays: `R` (starts as a copy of `A`) and `Q` (starts as the $m\\times m$ identity). Each step builds one unit vector `v` from the current column below the diagonal.',
    'You never need the matrix $H$: $HR = R - 2v(v^\\top R)$ and $QH = Q - 2(Qv)v^\\top$, using only rows/columns $k$ onward. Since each $H$ is symmetric and its own inverse, $A = H_0 H_1\\cdots R$, so $Q$ accumulates by right-multiplication.',
    'At the end take `Q[:, :n]`, `np.triu(R[:n])`, then multiply column $i$ of `Q` and row $i$ of `R` by $-1$ wherever `R[i, i] < 0`. `Q @ R` is unchanged because the two flips cancel.',
]

_r = np.random.default_rng(67)
TESTS = [
    (np.array([[3.0, 1.0], [4.0, 2.0]]),),
    (np.array([[1.0], [1.0], [1.0]]),),
    (np.array([[-2.0, 1.0], [0.0, -3.0]]),),
    (np.array([[5.0]]),),
    (np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),),
    (np.array([[1.0, 1.0, 1.0], [1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]]),),
    (_r.normal(size=(6, 4)),),
    (_r.normal(size=(5, 5)) * 1e4,),
]

EXPLANATION = r"""
## Why it works
Each reflector $H_k$ is symmetric and orthogonal ($H_k^2 = I$) and only touches rows $k$ onward, so it zeros column $k$ below the diagonal without disturbing the zeros created earlier. After $n$ steps, $H_{n-1}\cdots H_0 A = R_{\text{full}}$ is upper-triangular, so $A = (H_0\cdots H_{n-1})R_{\text{full}} = Q_{\text{full}}R_{\text{full}}$. The bottom $m - n$ rows of $R_{\text{full}}$ are zero, so only the first $n$ columns of $Q_{\text{full}}$ matter.

The sign choice $v = x + \operatorname{sign}(x_1)\lVert x\rVert e_1$ makes $v_1$ the *sum* of two same-sign numbers. The other choice subtracts nearly equal numbers when $x$ is close to $e_1$, and $v$ becomes mostly round-off.

## Why fix the signs
If $A = QR$, then also $A = (QD)(DR)$ for any diagonal $D$ of $\pm 1$s. Requiring $R_{ii} > 0$ picks one. `np.linalg.qr` doesn't promise it, so its output can differ in sign from yours.

Test 6 is the Läuchli matrix, where classical Gram–Schmidt loses orthogonality: Householder's $Q$ is orthonormal to about $10^{-16}$.

## Complexity
$O(mn^2)$ to compute $R$; accumulating the explicit $m\times m$ $Q$ costs $O(m^2 n)$. Libraries store the $v$'s instead and apply $Q$ implicitly.

## Common bugs
- Updating all rows of `R` instead of rows `k:` (or `Q`'s columns `k:`), which mixes in the already-finished part.
- Left-multiplying `Q` by `H` instead of right-multiplying.
- Using `np.sign(x[0])`, which is 0 when `x[0] == 0` and leaves `v = x`, a valid but cancellation-prone choice that differs from the statement.

## Interview follow-ups
- How do you solve least squares with QR? ($\min\lVert Ax - b\rVert$: solve $Rx = Q^\top b$ by back-substitution. No normal equations, so no squared condition number.)
- What's the alternative for sparse matrices? (Givens rotations: they zero one entry at a time and preserve sparsity.)
"""
