import numpy as np

META = dict(title='Rotation vectors to matrices (Rodrigues)', tags=['3D Geometry', 'Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:03.33'], related=['chain-rigid-poses', 'quaternion-slerp', 'outer-product'])

STATEMENT = r"""
A **rotation vector** (axis-angle) $v \in \mathbb{R}^3$ encodes a rotation by angle $\theta = \lVert v \rVert$ (radians, right-hand rule) about the unit axis $n = v / \theta$. Rodrigues' formula turns it into a rotation matrix:

$$R = I + \frac{\sin\theta}{\theta}\,[v]_\times + \frac{1 - \cos\theta}{\theta^2}\,[v]_\times^2, \qquad [v]_\times = \begin{pmatrix} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{pmatrix}$$

$[v]_\times$ is the cross-product matrix: $[v]_\times w = v \times w$.

Write `rotvec_to_matrix(V)` that converts a batch of rotation vectors.

## Input
- `V`: float array of shape `(N, 3)`, one rotation vector per row. `N` may be 0.

## Output
- Float array of shape `(N, 3, 3)`. `R[i] @ p` rotates a column vector `p` by `V[i]`.

## Requirements
- Vectorised over the batch, no Python loop over rows.
- The two coefficients are $0/0$ at $\theta = 0$. When $\theta < 10^{-6}$, use their Taylor series instead: $\frac{\sin\theta}{\theta} \approx 1 - \frac{\theta^2}{6}$ and $\frac{1-\cos\theta}{\theta^2} \approx \frac12 - \frac{\theta^2}{24}$. The zero vector must give exactly the identity.
"""

STARTER = '''import numpy as np

def rotvec_to_matrix(V):
    """V: (N, 3) rotation vectors. Return (N, 3, 3) rotation matrices."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def rotvec_to_matrix(V):
    V = np.asarray(V, dtype=float).reshape(-1, 3)
    th = np.linalg.norm(V, axis=1)
    small = th < 1e-6
    safe = np.where(small, 1.0, th)                       # avoid 0/0 in the branch np.where discards
    A = np.where(small, 1 - th**2 / 6, np.sin(safe) / safe)
    B = np.where(small, 0.5 - th**2 / 24, (1 - np.cos(safe)) / safe**2)

    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    K = np.zeros((len(V), 3, 3))                          # [v]_x for every row
    K[:, 0, 1], K[:, 0, 2] = -z, y
    K[:, 1, 0], K[:, 1, 2] = z, -x
    K[:, 2, 0], K[:, 2, 1] = -y, x

    return np.eye(3) + A[:, None, None] * K + B[:, None, None] * (K @ K)
'''

HINTS = [
    'Build the skew matrix $[v]_\\times$ for every row at once: allocate `(N, 3, 3)` zeros and fill the six off-diagonal slots from the columns of `V`.',
    'Compute the two scalar coefficients as `(N,)` arrays, then broadcast them with `[:, None, None]`. `K @ K` is a batched matmul.',
    'Use `np.where(small, series, exact)`, but feed the exact formula a safe angle (e.g. 1.0) where `small` is true, or NumPy warns about `0/0`.',
]

_r = np.random.default_rng(11)
_axes = _r.normal(size=(5, 3))
_axes /= np.linalg.norm(_axes, axis=1, keepdims=True)
TESTS = [
    (np.array([[0.0, 0.0, np.pi / 2]]),),
    (np.array([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]]),),
    (np.array([[1e-9, -2e-9, 5e-10]]),),
    (np.array([[0.3, -0.4, 1.2], [2.0, 2.0, 2.0]]),),
    (np.array([[2e-6, 0.0, 0.0], [0.0, 9e-7, 0.0]]),),
    (np.array([[0.0, 2 * np.pi, 0.0]]),),
    (_axes * _r.uniform(0, np.pi, size=(5, 1)),),
    (_r.normal(size=(4, 3)) * 3,),
]

EXPLANATION = r"""
## Why it works
Split a vector $p$ into its part along the axis $n$ and its part perpendicular to it. The parallel part does not move. The perpendicular part rotates by $\theta$ in the plane spanned by $p_\perp$ and $n \times p$. Writing that out gives $R = I + \sin\theta\,[n]_\times + (1-\cos\theta)\,[n]_\times^2$, since $[n]_\times^2 p = -p_\perp$. Substituting $n = v/\theta$ moves the $\theta$s into the coefficients and lets you skip normalising the axis.

## Complexity
$O(N)$ time and memory, with a constant number of $3\times3$ operations per row.

## Common bugs
- Normalising $v$ before checking $\theta$, which divides by zero for the identity rotation.
- Swapping the signs in $[v]_\times$. That gives the transpose, a rotation by $-\theta$. Test 1 (90° about $z$ must send $x$ to $y$) catches it.
- Letting `np.where` evaluate `sin(th)/th` at `th = 0`. The result is discarded, but NumPy still warns, and the warnings hide real problems.

## Interview follow-ups
- Invert the map: recover $v$ from $R$ using $\cos\theta = (\mathrm{tr}R - 1)/2$ and the skew part $R - R^\top$. What breaks near $\theta = \pi$?
- Why do optimisers (bundle adjustment, pose graph SLAM) update rotations through a rotation vector instead of the nine matrix entries?
"""
