import numpy as np

META = dict(title='Oriented surface normals by local PCA', tags=['3D Geometry', 'Linear Algebra'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:08.06'], related=['icp-kabsch-step', 'eigendecomposition', 'pca'])

STATEMENT = r"""
A point cloud from a depth sensor has positions but no surface normals. A standard way to estimate them: take each point's neighbourhood, fit a plane by PCA, and use the direction of **least** variance as the normal.

Write `estimate_normals(P, k, viewpoint)`.

## Inputs
- `P`: `(N, 3)` float points, $N \ge k$.
- `k`: neighbourhood size, $3 \le k \le N$.
- `viewpoint`: `(3,)` position of the sensor.

## Steps, for every point $p_i$
1. **Neighbours.** Sort all $N$ points by Euclidean distance to $p_i$, breaking exactly equal distances by **lower index**, and keep the first $k$. The point itself (distance 0) is included.
2. **Covariance.** With $\mu$ the mean of the $k$ neighbours, $C = \frac1k \sum_q (q - \mu)(q - \mu)^\top$.
3. **Normal.** The unit eigenvector $n$ of $C$ for its smallest eigenvalue $\lambda_0$ (eigenvalues $\lambda_0 \le \lambda_1 \le \lambda_2$).
4. **Orientation.** An eigenvector's sign is arbitrary. Choose it so the normal faces the sensor: $n \cdot (\text{viewpoint} - p_i) > 0$.
5. **Surface variation.** $\sigma = \lambda_0 / (\lambda_0 + \lambda_1 + \lambda_2)$: 0 on a perfect plane, up to $1/3$ for an isotropic blob.

## Output
A tuple `(normals, variation)`: a float array of shape `(N, 3)` with unit rows, and a float array of shape `(N,)`.

The tests guarantee that every neighbourhood has a strictly smallest eigenvalue (well separated from $\lambda_1$) and that no normal is perpendicular to its view direction, so the answer is unique.
"""

STARTER = '''import numpy as np

def estimate_normals(P, k, viewpoint):
    """P: (N, 3). Return (normals (N, 3), variation (N,))."""
    P = np.asarray(P, float)
    C = np.cov(P.T)
    w, V = np.linalg.eigh(C)
    n = np.tile(V[:, 0], (len(P), 1))
    return n, np.full(len(P), w[0] / w.sum())
'''

SOLUTION = '''import numpy as np

def estimate_normals(P, k, viewpoint):
    P = np.asarray(P, float)
    vp = np.asarray(viewpoint, float)

    # 1. k nearest neighbours; a stable sort keeps lower indices first on exact ties
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)          # (N, N)
    idx = np.argsort(d2, axis=1, kind="stable")[:, :k]           # (N, k)
    nb = P[idx]                                                  # (N, k, 3)

    # 2. biased covariance of each neighbourhood (divide by k, not k - 1)
    D = nb - nb.mean(axis=1, keepdims=True)
    C = np.einsum("nki,nkj->nij", D, D) / k                      # (N, 3, 3)

    # 3. eigh sorts eigenvalues ascending, so column 0 is the least-variance direction
    w, V = np.linalg.eigh(C)
    n = V[:, :, 0]

    # 4. flip normals that point away from the sensor
    s = np.sign(((vp - P) * n).sum(axis=1))
    n = n * s[:, None]

    return n, w[:, 0] / w.sum(axis=1)
'''

HINTS = [
    'Compute the full `(N, N)` squared-distance matrix and take `np.argsort(..., kind="stable")[:, :k]`. A stable sort keeps lower indices first among equal distances; the default quicksort does not promise that.',
    'Gather neighbourhoods with fancy indexing, `P[idx]` of shape `(N, k, 3)`, centre them, and build all covariances at once with `np.einsum("nki,nkj->nij", D, D) / k`.',
    '`np.linalg.eigh` works on a stack of symmetric matrices and returns eigenvalues in ascending order, so the normal is `V[:, :, 0]` (a column, not a row). Then multiply each normal by the sign of its dot product with `viewpoint - p`.',
]

_r = np.random.default_rng(17)
_grid = np.array([[x, y, 0.0] for x in range(3) for y in range(3)], float)
_tilt = np.c_[_r.uniform(-1, 1, size=(15, 2)), np.zeros(15)]
_tilt[:, 2] = 0.5 * _tilt[:, 0] - 0.3 * _tilt[:, 1] + _r.normal(scale=0.01, size=15)
_sph = _r.normal(size=(30, 3))
_sph /= np.linalg.norm(_sph, axis=1, keepdims=True)
_wave = np.c_[_r.uniform(-2, 2, size=(25, 2)), np.zeros(25)]
_wave[:, 2] = 0.2 * np.sin(_wave[:, 0])
TESTS = [
    (_grid, 4, np.array([1.0, 1.0, 10.0])),
    (_grid[:, [0, 2, 1]], 5, np.array([1.0, 3.0, 1.0])),
    (_grid, 5, np.array([1.0, 1.0, -5.0])),
    (_tilt, 6, np.array([0.0, 0.0, 10.0])),
    (_tilt, 6, np.array([0.0, 0.0, -10.0])),
    (_sph * 2.0, 6, np.zeros(3)),
    (_wave, 7, np.array([0.0, 0.0, 5.0])),
    (np.r_[_grid, _r.normal(scale=0.3, size=(6, 3)) + [1.0, 1.0, 2.0]], 5, np.array([1.0, 1.0, 10.0]))
]

EXPLANATION = r"""
## Why it works
Near any point a smooth surface looks like its tangent plane. The covariance of a small neighbourhood has large variance along the two tangent directions and almost none across the surface, so the eigenvector of the smallest eigenvalue estimates the normal. PCA is the least-squares plane fit: that eigenvector minimises $\sum_q ((q - \mu)\cdot n)^2$ over unit $n$, and $\lambda_0$ is the mean squared distance from the plane.

The eigen-solver returns $n$ or $-n$ unpredictably, and different platforms can disagree. Pointing every normal toward the sensor fixes the sign deterministically and matches what renderers and meshing need. On a closed surface viewed from inside (test 6, sensor at the centre of a sphere) all normals point inward.

$\sigma$ measures how far the neighbourhood is from flat. It is used to detect edges and corners and to choose keypoints.

## Complexity
$O(N^2)$ memory and $O(N^2 \log N)$ time for the brute-force neighbour sort, plus $O(Nk)$ for the covariances and a constant-cost $3\times3$ eigendecomposition per point. Real pipelines use a k-d tree and `argpartition` to get about $O(N \log N + Nk)$.

## Common bugs
- Taking a **row** of the eigenvector matrix (`V[:, 0, :]`) instead of a column.
- Using the largest eigenvector, which is a tangent direction.
- Leaving the sign to the solver, which makes outputs flip between runs or machines.
- Using the default sort, which does not guarantee the lower-index tie rule on a regular grid (tests 1-3 are full of exact ties).

## Interview follow-ups
- Orienting toward a viewpoint fails for merged scans from many viewpoints. How does propagating orientation along a minimum spanning tree of neighbours fix that?
- How does the choice of $k$ trade off noise robustness against blurring sharp edges? How would you pick it per point?
"""
