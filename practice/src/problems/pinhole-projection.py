import numpy as np

META = dict(title='Project 3D points through a pinhole camera', tags=['3D Geometry', 'Computer Vision'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:03.33'], related=['chain-rigid-poses', 'iou-matrix'])

STATEMENT = r"""
A calibrated pinhole camera maps a world point $X_w$ to a pixel in three steps.

1. **Extrinsics.** Move the point into the camera frame: $X_c = R X_w + t$. The camera looks down $+z$, with $x$ to the right and $y$ **down** the image.
2. **Normalise and distort.** $x = X_c^{(x)} / X_c^{(z)}$, $y = X_c^{(y)} / X_c^{(z)}$, $r^2 = x^2 + y^2$, then apply radial distortion: $x_d = x\,(1 + k_1 r^2 + k_2 r^4)$ and the same for $y_d$.
3. **Intrinsics.** $\begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = K \begin{pmatrix} x_d \\ y_d \\ 1 \end{pmatrix}$, with $K = \begin{pmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$.

Write `project_points(Xw, K, R, t, image_size, dist=(0.0, 0.0))`.

## Inputs
- `Xw`: `(N, 3)` world points. `N` may be 0.
- `K`: `(3, 3)` intrinsics as above (use all of it, including the skew `s`). `R`: `(3, 3)` rotation. `t`: `(3,)` translation.
- `image_size`: `(W, H)` in pixels. `dist`: `(k1, k2)`.

## Output
A dict with three entries:

| Key | Shape, dtype | Value |
|---|---|---|
| `"uv"` | `(N, 2)` float | `[u, v]`, or `[nan, nan]` for a point with $X_c^{(z)} \le 0$ |
| `"depth"` | `(N,)` float | $X_c^{(z)}$ for every point, including those behind the camera |
| `"visible"` | `(N,)` bool | `True` iff $X_c^{(z)} > 0$ and $0 \le u < W$ and $0 \le v < H$ |

No point in the tests has depth exactly 0 or lands exactly on an image border.
"""

STARTER = '''import numpy as np

def project_points(Xw, K, R, t, image_size, dist=(0.0, 0.0)):
    """Return {"uv": (N, 2), "depth": (N,), "visible": (N,) bool}."""
    Xc = Xw @ R + t
    uv = K[:2, :2] @ (Xc[:, :2] / Xc[:, 2:]).T
    return {"uv": uv.T + K[:2, 2], "depth": Xc[:, 2], "visible": Xc[:, 2] > 0}
'''

SOLUTION = '''import numpy as np

def project_points(Xw, K, R, t, image_size, dist=(0.0, 0.0)):
    Xw = np.asarray(Xw, float).reshape(-1, 3)
    K, R, t = np.asarray(K, float), np.asarray(R, float), np.asarray(t, float)
    W, H = image_size
    k1, k2 = dist

    Xc = Xw @ R.T + t                                  # rows: R @ x + t for each point
    z = Xc[:, 2]
    front = z > 0
    zs = np.where(front, z, 1.0)                       # don't divide by depths we will discard
    xy = Xc[:, :2] / zs[:, None]

    r2 = (xy ** 2).sum(axis=1, keepdims=True)
    xy_d = xy * (1 + k1 * r2 + k2 * r2 ** 2)           # radial distortion scales x and y alike

    hom = np.hstack([xy_d, np.ones((len(xy_d), 1))])   # (N, 3) homogeneous normalised coords
    uv = (hom @ K.T)[:, :2]
    uv[~front] = np.nan

    u, v = uv[:, 0], uv[:, 1]
    with np.errstate(invalid="ignore"):                # comparisons with nan are simply False
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return {"uv": uv, "depth": z, "visible": front & inside}
'''

HINTS = [
    'Points are rows, so the extrinsic step is `Xw @ R.T + t`. Getting $R$ vs $R^\\top$ wrong is the most common projection bug.',
    'Divide by depth only where it is positive: compute a safe depth with `np.where`, then overwrite the rows behind the camera with `nan` afterwards.',
    'Distortion is applied to the **normalised** coordinates, before $K$. Append a column of ones and multiply by `K.T` to get `[u, v, 1]`, which keeps the skew term.',
]

_K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
_I = np.eye(3)
_c, _s = np.cos(0.4), np.sin(0.4)
_Ry = np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])
_r = np.random.default_rng(8)
TESTS = [
    (np.array([[0.0, 0.0, 5.0], [1.0, 0.5, 5.0]]), _K, _I, np.zeros(3), (640, 480)),
    (np.array([[0.0, 0.0, -2.0], [2.0, 0.0, 2.0]]), _K, _I, np.zeros(3), (640, 480)),
    (np.array([[0.2, -0.1, 0.0], [-0.5, 0.3, 1.0]]), _K, _Ry, np.array([0.1, -0.2, 3.0]), (640, 480)),
    (np.array([[0.3, 0.2, 1.0], [-0.4, 0.1, 2.0]]), _K, _I, np.zeros(3), (640, 480), (-0.2, 0.05)),
    (np.array([[0.5, 0.5, 4.0]]), np.array([[600.0, 2.0, 300.0], [0.0, 580.0, 250.0], [0.0, 0.0, 1.0]]), _I, np.zeros(3), (640, 480)),
    (np.array([[0.0, -1.0, 1.0], [0.0, 0.0, 100.0]]), _K, _I, np.zeros(3), (640, 480)),
    (np.array([[0.0, 0.0, 1.0]]), _K, _I, np.array([0.0, 0.0, -1.5]), (640, 480)),
    (_r.uniform(-3, 3, size=(8, 3)) + [0, 0, 4], _K, _Ry, np.array([0.3, 0.0, 1.0]), (640, 480), (0.1, -0.01)),
]

EXPLANATION = r"""
## Why it works
The extrinsics $(R, t)$ re-express the point in the camera's own frame. Dividing by depth is the actual perspective step: everything on one ray through the optical centre lands on the same normalised point $(x, y)$. Lens distortion is a function of the angle from the optical axis, so it acts on those normalised coordinates. $K$ only rescales them to pixels (focal lengths), shifts the origin to the image corner (principal point) and optionally shears (skew).

Points with $z \le 0$ would still produce a finite-looking pixel after the division (a point behind the camera flips through the centre), so they must be masked out before the image-bounds test (test 2).

## Complexity
$O(N)$ time and memory.

## Common bugs
- Using `Xw @ R` instead of `Xw @ R.T`, which applies the inverse rotation.
- Applying distortion after $K$, i.e. in pixels instead of normalised coordinates.
- Dropping the skew entry $K_{01}$ by treating $K$ as two independent focal lengths (test 5).
- Using `u <= W`: pixel centres run from 0 to $W-1$, so a valid continuous coordinate is in $[0, W)$.

## Interview follow-ups
- Given the pixel and its depth, back-project it to a 3D point. Why does undistortion need an iterative solve?
- Your extrinsics are camera-to-world instead of world-to-camera. What changes? (Invert them: $R^\top$ and $-R^\top t$.)
"""
