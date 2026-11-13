import numpy as np

META = dict(title='Quaternion slerp along the short arc', tags=['3D Geometry'], diff='Medium', runtime='numpy',
            lessons=[], related=['rodrigues-rotation', 'cosine-similarity'])

STATEMENT = r"""
A unit quaternion $q = (w, x, y, z)$ represents a 3D rotation by angle $\theta$ about unit axis $n$ as $q = (\cos\frac\theta2,\ \sin\frac\theta2\, n)$. Both $q$ and $-q$ represent the **same** rotation.

Spherical linear interpolation moves along the great circle between two unit quaternions at constant angular speed:

$$\mathrm{slerp}(q_0, q_1; s) = \frac{\sin\big((1-s)\Omega\big)}{\sin\Omega}\, q_0 + \frac{\sin(s\Omega)}{\sin\Omega}\, q_1, \qquad \cos\Omega = q_0 \cdot q_1$$

Write `slerp(q0, q1, ts)` that returns the interpolated quaternions.

## Input
- `q0`, `q1`: float arrays of shape `(4,)` in `(w, x, y, z)` order. They are non-zero but **not necessarily unit length**.
- `ts`: float array of shape `(T,)`, each value in $[0, 1]$.

## Steps
1. Normalise `q0` and `q1` to unit length.
2. **Short arc:** if $q_0 \cdot q_1 < 0$, replace $q_1$ by $-q_1$ (same rotation, the other way round is shorter). If the dot is exactly 0, leave it.
3. If the (possibly flipped) dot is greater than $0.9995$, use normalised linear interpolation: $\frac{(1-s) q_0 + s q_1}{\lVert (1-s) q_0 + s q_1 \rVert}$. Otherwise use the slerp formula above, with $\Omega = \arccos(\text{dot})$.

## Output
- Float array of shape `(T, 4)`. Row $i$ is the result for `ts[i]`, so row 0 equals the normalised `q0` when `ts[0] == 0`. Do not change the sign of the result.
"""

STARTER = '''import numpy as np

def slerp(q0, q1, ts):
    """q0, q1: (4,) quaternions (w, x, y, z). ts: (T,). Return (T, 4)."""
    q0, q1, ts = np.asarray(q0, float), np.asarray(q1, float), np.asarray(ts, float)
    out = (1 - ts)[:, None] * q0 + ts[:, None] * q1
    return out / np.linalg.norm(out, axis=1, keepdims=True)
'''

SOLUTION = '''import numpy as np

def slerp(q0, q1, ts):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    ts = np.asarray(ts, float)[:, None]                 # (T, 1) so it broadcasts against (4,)
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)

    dot = float(q0 @ q1)
    if dot < 0:                                         # q and -q are the same rotation: take the short way
        q1, dot = -q1, -dot

    if dot > 0.9995:                                    # nearly parallel: sin(Omega) ~ 0, lerp is accurate
        out = (1 - ts) * q0 + ts * q1
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    omega = np.arccos(dot)
    s = np.sin(omega)
    return (np.sin((1 - ts) * omega) / s) * q0 + (np.sin(ts * omega) / s) * q1
'''

HINTS = [
    'Normalise first: the dot product is only $\\cos\\Omega$ for unit quaternions.',
    'If the dot is negative, the two quaternions are more than 90° apart on the 4D sphere, but $-q_1$ is the same rotation and closer. Flip $q_1$ and the dot.',
    'When the quaternions are almost equal, $\\sin\\Omega$ is tiny and the formula divides by almost 0. That is the reason for the `0.9995` lerp fallback, and why the lerp result is renormalised.',
]


def _q(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    return np.r_[np.cos(ang / 2), np.sin(ang / 2) * a]


_r = np.random.default_rng(21)
TESTS = [
    (np.array([1.0, 0.0, 0.0, 0.0]), _q([0, 0, 1], np.pi / 2), np.array([0.0, 0.5, 1.0])),
    (np.array([2.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 3.0]), np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
    (np.array([1.0, 0.0, 0.0, 0.0]), -_q([1, 0, 0], 0.6), np.array([0.0, 0.5, 1.0])),
    (_q([0, 1, 0], 0.3), _q([0, 1, 0], 0.3 + 1e-4), np.array([0.0, 0.3, 1.0])),
    (_q([1, 1, 0], 1.0), -_q([1, 1, 0], 1.0), np.array([0.0, 0.5, 1.0])),
    (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.5])),
    (_r.normal(size=4), _r.normal(size=4), np.linspace(0, 1, 7)),
    (_r.normal(size=4) * 5, _r.normal(size=4) * 0.1, _r.uniform(0, 1, size=5)),
]

EXPLANATION = r"""
## Why it works
Unit quaternions lie on the 3-sphere in $\mathbb{R}^4$. The slerp weights are chosen so the result stays on that sphere and the angle from $q_0$ grows linearly in $s$: the output makes angle $s\Omega$ with $q_0$ and $(1-s)\Omega$ with $q_1$. Since a quaternion angle of $\Omega$ is a rotation angle of $2\Omega$, the interpolated rotation turns about a fixed axis at constant angular speed, which plain matrix or Euler-angle interpolation cannot do.

The quaternion sphere covers every rotation twice ($q$ and $-q$). Without the sign flip, slerp between two rotations 10° apart can take the 350° route instead (test 3). If $q_1 = -q_0$ (test 5), the two are the same rotation, and after the flip the output is constant.

## Complexity
$O(T)$: one dot product, then a weighted sum of two 4-vectors per time step.

## Common bugs
- Skipping normalisation, so `arccos` gets a value outside $[-1, 1]$ and returns `nan`.
- Forgetting the short-arc flip.
- Dividing by $\sin\Omega \approx 0$ for nearly equal inputs (test 4) and getting noisy or infinite weights.
- Not renormalising the lerp branch, which slowly shrinks quaternions over long animations.

## Interview follow-ups
- Convert the slerped quaternion to a rotation matrix. How many multiplies does it cost compared with Rodrigues?
- For a smooth path through more than two keyframes, why isn't piecewise slerp enough, and what does SQUAD (or a spline in the tangent space) add?
"""
