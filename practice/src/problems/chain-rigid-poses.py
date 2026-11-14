import numpy as np

META = dict(title='Chain and invert rigid transforms', tags=['3D Geometry', 'Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:03.33'], related=['rodrigues-rotation', 'pinhole-projection', 'matrix-multiply'])

STATEMENT = r"""
A rigid transform in 3D is a rotation $R$ (orthonormal, $\det R = 1$) followed by a translation $t$, stored as a $4\times4$ homogeneous matrix:

$$T = \begin{pmatrix} R & t \\ 0 & 1 \end{pmatrix}, \qquad T \begin{pmatrix} p \\ 1 \end{pmatrix} = \begin{pmatrix} Rp + t \\ 1 \end{pmatrix}$$

A robot arm or a camera rig is a chain of frames $0, 1, \dots, K$. You are given `Ts`, shape `(K, 4, 4)`, where `Ts[i]` maps coordinates in frame `i+1` to coordinates in frame `i` (it is $T_{i \leftarrow i+1}$). You also get points `P`, shape `(N, 3)`, expressed in the **last** frame $K$.

Write `chain_poses(Ts, P)` and return a tuple `(T_0K, T_K0, P0)`:

| Output | Shape | Meaning |
|---|---|---|
| `T_0K` | `(4, 4)` | maps frame $K$ to frame $0$ |
| `T_K0` | `(4, 4)` | the inverse: maps frame $0$ to frame $K$ |
| `P0` | `(N, 3)` | the points `P` expressed in frame 0 |

## Requirements
- $K \ge 1$; $N$ may be 0 (return a `(0, 3)` array).
- Compute `T_K0` in closed form from the rotation and translation of `T_0K` (no general matrix inverse). Its last row must be exactly `[0, 0, 0, 1]`.
- All outputs are float arrays.
"""

STARTER = '''import numpy as np

def chain_poses(Ts, P):
    """Ts: (K, 4, 4) with Ts[i] = T_{i <- i+1}. P: (N, 3) in frame K.
    Return (T_0K, T_K0, P0)."""
    T = np.eye(4)
    for Ti in Ts:
        T = Ti @ T
    return T, np.linalg.inv(T), P
'''

SOLUTION = '''import numpy as np

def chain_poses(Ts, P):
    Ts, P = np.asarray(Ts, dtype=float), np.asarray(P, dtype=float).reshape(-1, 3)
    T = np.eye(4)
    for Ti in Ts:
        T = T @ Ti                       # T_{0<-i+1} = T_{0<-i} @ T_{i<-i+1}: new links go on the right

    R, t = T[:3, :3], T[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T                    # R is orthonormal, so R^-1 = R^T
    inv[:3, 3] = -R.T @ t                # undo the translation, then the rotation

    P0 = P @ R.T + t                     # row-vector form of R p + t
    return T, inv, P0
'''

HINTS = [
    'Read the subscripts like fractions that cancel: $T_{0\\leftarrow 2} = T_{0\\leftarrow 1}\\,T_{1\\leftarrow 2}$. Each new link multiplies on the **right**.',
    'If $q = Rp + t$, then $p = R^\\top (q - t) = R^\\top q - R^\\top t$. That gives the rotation and translation blocks of the inverse.',
    'Points are rows, so apply the transform as `P @ R.T + t`, not `R @ P`.',
]


def _T(rv, t):
    th = np.linalg.norm(rv)
    K = np.array([[0, -rv[2], rv[1]], [rv[2], 0, -rv[0]], [-rv[1], rv[0], 0]]) / (th if th else 1)
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    T[:3, 3] = t
    return T


_r = np.random.default_rng(5)
_chain = np.stack([_T(_r.normal(size=3), _r.normal(size=3) * 2) for _ in range(4)])
TESTS = [
    (np.stack([_T([0, 0, 0], [1, 2, 3])]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
    (np.stack([_T([0, 0, np.pi / 2], [1, 0, 0]), _T([0, 0, 0], [1, 0, 0])]), np.array([[0.0, 0.0, 0.0]])),
    (np.stack([_T([np.pi / 2, 0, 0], [0, 0, 0]), _T([0, np.pi / 2, 0], [0, 0, 0])]), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
    (np.stack([_T(np.full(3, 2 * np.pi / 3 / np.sqrt(3)), [1, 0, 0])] * 3), np.array([[0.0, 0.0, 1.0]])),
    (np.stack([_T([0, 0, 1.0], [1, 1, 0]), _T([0, 0, -1.0], [-1, -1, 0])]), np.array([[2.0, 3.0, 4.0]])),
    (_chain[:2], _r.normal(size=(3, 3))),
    (_chain, _r.normal(size=(6, 3)) * 5),
]

EXPLANATION = r"""
## Why it works
$T_{i\leftarrow i+1}$ takes a point written in frame $i+1$ and rewrites it in frame $i$. To go from frame $K$ to frame $0$ you apply $T_{K-1\leftarrow K}$ first and $T_{0\leftarrow 1}$ last, so the product is $T_{0\leftarrow 1} T_{1\leftarrow 2} \cdots T_{K-1\leftarrow K}$. Accumulating left to right means each new factor goes on the right. Test 2 (rotate, then translate along the new $x$) gives a different answer under the wrong order.

The inverse follows from solving $q = Rp + t$ for $p$. Orthonormality makes $R^{-1} = R^\top$ exact and cheap, and keeps the last row exactly `[0, 0, 0, 1]`.

## Complexity
$O(K)$ for the $4\times4$ products and $O(N)$ to transform the points.

## Common bugs
- Multiplying in the wrong order, which gives $T_{K-1\leftarrow K}\cdots T_{0\leftarrow 1}$.
- Inverting as `[R.T | -t]`. The translation must be rotated too: $-R^\top t$.
- `np.linalg.inv` is correct here, but it is slower and, after many compositions, slowly drifts away from a true rigid transform.
- Treating points as columns while storing them as rows, which applies $R^\top$ instead of $R$.

## Interview follow-ups
- After thousands of compositions $R$ is no longer orthonormal. How do you re-project it onto SO(3)? (SVD: replace the singular values with 1, keeping $\det = +1$.)
- How do you transform a **direction** (a surface normal or a velocity) differently from a point? (Use $w = 0$ in homogeneous coordinates, so the translation drops out.)
"""
