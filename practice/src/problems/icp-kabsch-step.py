import numpy as np

META = dict(title='One ICP step with Kabsch alignment', tags=['3D Geometry', 'Linear Algebra'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:01.13'], related=['svd-components', 'euclidean-distance', 'chain-rigid-poses'])

STATEMENT = r"""
Iterative Closest Point aligns a source point cloud to a target cloud by repeating two steps: match each source point to its nearest target point, then solve for the rigid transform that best aligns the matched pairs. Implement **one** step.

Write `icp_step(src, dst, max_dist=np.inf)`.

## Inputs
- `src`: `(N, 3)` source points. `dst`: `(M, 3)` target points, $M \ge 1$.
- `max_dist`: matches with Euclidean distance **greater** than this are rejected (distance equal to `max_dist` is kept).

## Steps
1. **Match.** For each source point $i$, find the nearest target point (Euclidean distance). On exactly equal distances, take the **lowest** target index.
2. **Reject** matches farther than `max_dist`.
3. **Kabsch.** Over the kept pairs $(p_j, q_j)$, find the proper rotation $R$ ($\det R = +1$) and translation $t$ minimising $\sum_j \lVert R p_j + t - q_j \rVert^2$:
   - centroids $\bar p, \bar q$; cross-covariance $H = \sum_j (p_j - \bar p)(q_j - \bar q)^\top$ (a $3\times3$ matrix);
   - SVD $H = U \Sigma V^\top$; $d = \operatorname{sign}\det(V U^\top)$; $R = V\,\mathrm{diag}(1, 1, d)\,U^\top$; $t = \bar q - R \bar p$.
4. **Score.** RMSE of the kept pairs **after** applying $(R, t)$: $\sqrt{\frac1n \sum_j \lVert R p_j + t - q_j\rVert^2}$.

## Output
A dict:

| Key | Value |
|---|---|
| `"R"` | `(3, 3)` float |
| `"t"` | `(3,)` float |
| `"matches"` | `(N,)` int array: the matched target index, or `-1` if rejected |
| `"rmse"` | Python float |

If fewer than 3 pairs are kept, skip the alignment and return `R = I`, `t = 0` and `rmse = nan`, with `matches` filled in as usual. Whenever 3 or more pairs are kept, the tests guarantee those source points are not collinear, so $R$ is unique.
"""

STARTER = '''import numpy as np

def icp_step(src, dst, max_dist=np.inf):
    """Return {"R": (3,3), "t": (3,), "matches": (N,) int, "rmse": float}."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    match = np.argmin(((src[:, None] - dst[None]) ** 2).sum(-1), axis=1)
    P, Q = src, dst[match]
    U, S, Vt = np.linalg.svd((P - P.mean(0)).T @ (Q - Q.mean(0)))
    R = Vt.T @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    rmse = float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(1).mean()))
    return {"R": R, "t": t, "matches": match, "rmse": rmse}
'''

SOLUTION = '''import numpy as np

def icp_step(src, dst, max_dist=np.inf):
    src = np.asarray(src, float).reshape(-1, 3)
    dst = np.asarray(dst, float).reshape(-1, 3)

    # 1. brute-force nearest neighbours; argmin returns the first (lowest) index on ties
    d2 = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(-1)       # (N, M)
    nn = np.argmin(d2, axis=1) if len(src) else np.zeros(0, int)
    dist = np.sqrt(d2[np.arange(len(src)), nn])

    # 2. reject far matches (compare distances, not squared distances, against max_dist)
    keep = dist <= max_dist
    matches = np.where(keep, nn, -1).astype(int)
    if keep.sum() < 3:
        return {"R": np.eye(3), "t": np.zeros(3), "matches": matches, "rmse": float("nan")}

    # 3. Kabsch on the kept pairs
    P, Q = src[keep], dst[nn[keep]]
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)                                     # sum of outer products (p - pc)(q - qc)^T
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))                        # -1 means the best orthogonal fit is a reflection
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = qc - R @ pc

    # 4. residual after alignment
    res = P @ R.T + t - Q
    return {"R": R, "t": t, "matches": matches, "rmse": float(np.sqrt((res ** 2).sum(1).mean()))}
'''

HINTS = [
    'Matching is a full `(N, M)` squared-distance matrix plus `argmin` along the target axis. `argmin` already breaks ties toward the lowest index.',
    'Centre both matched sets before building $H$. Translation then drops out, and once $R$ is known, $t = \\bar q - R\\bar p$.',
    'Without the $\\mathrm{diag}(1, 1, d)$ correction, SVD can return a reflection ($\\det = -1$), especially for flat or noisy clouds. $d = \\operatorname{sign}\\det(VU^\\top)$ flips the axis with the smallest singular value.',
]


def _rot(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    K = np.array([[0, -rv[2], rv[1]], [rv[2], 0, -rv[0]], [-rv[1], rv[0], 0]]) / th
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


_r = np.random.default_rng(13)
_cloud = _r.normal(size=(12, 3))
_Rt = _rot([0.1, -0.05, 0.08])
_moved = _cloud @ _Rt.T + [0.05, -0.02, 0.03]
_thin = np.array([[-0.0414, -0.0838, -1.361], [-0.0263, 0.9383, -1.5453], [0.0301, -0.4351, 0.067],
                  [0.0082, -0.2775, 0.3472], [-0.0406, 0.9514, 1.8251], [-0.0067, -0.8632, 0.5942]])   # nearly flat in x
_sq = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
TESTS = [
    (_sq, _sq + [0.0, 0.0, 0.5]),
    (_sq, _sq @ _rot([0, 0, 0.2]).T),
    (np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
     np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.1], [0.0, 0.1, 2.0]])),
    (np.r_[_sq, [[9.0, 9.0, 9.0]]], _sq + 0.1, 1.0),
    (_sq, _sq + [5.0, 0.0, 0.0], 1.0),
    (_thin, _thin * [-1.0, 1.0, 1.0]),
    (_cloud, _moved),
    (_cloud + _r.normal(scale=0.01, size=_cloud.shape), _moved, 0.5),
]

EXPLANATION = r"""
## Why it works
With correspondences fixed, the optimal translation always maps centroid to centroid, so the problem reduces to finding the rotation that maximises $\sum_j (q_j - \bar q)^\top R (p_j - \bar p) = \mathrm{tr}(R H)$. Writing $H = U\Sigma V^\top$, the trace is largest when $R$ lines up $U$ with $V$: $R = VU^\top$. If that matrix has $\det = -1$ it is a reflection. The best **proper** rotation then flips the direction with the smallest singular value, because that costs the least trace. That is the $\mathrm{diag}(1,1,d)$ term. Test 6 matches a nearly flat cloud to its mirror image across $x = 0$: the best orthogonal fit is that mirror, so without the correction you return a reflection.

ICP then alternates: better alignment gives better nearest neighbours, which give a better alignment. Each step does not increase the matched error, so it converges, but only to a local minimum. It needs a decent initial pose.

## Complexity
Brute-force matching is $O(NM)$ time and memory. The Kabsch step is $O(n)$ plus a constant $3\times3$ SVD.

## Common bugs
- Building $H$ as $\sum (q - \bar q)(p - \bar p)^\top$, which returns $R^\top$.
- Skipping the determinant check, so a mirrored scene comes back as a "rotation" with $\det = -1$.
- Comparing squared distances against `max_dist` without squaring it.
- Computing $t$ from un-centred means or before $R$ is final.
- Using `np.sign(det)` on $H$ itself instead of on $VU^\top$.

## Interview follow-ups
- Replace brute force with a k-d tree. What does matching cost now, and when does building the tree not pay off?
- Point-to-plane ICP minimises $\big((Rp + t - q)\cdot n_q\big)^2$. Why does it converge faster on smooth surfaces, and why can't it be solved in closed form like Kabsch?
"""
