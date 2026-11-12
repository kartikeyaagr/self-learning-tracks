import numpy as np

META = dict(title='One step of k-means (Lloyd\'s algorithm)', tags=['Clustering', 'Classic ML'], diff='Medium', runtime='numpy',
            lessons=[], related=['pca-projection', 'cosine-similarity-matrix'])

STATEMENT = r"""
Lloyd's algorithm alternates two moves until the clusters stop changing:

1. **Assign**: each point goes to its nearest centroid (squared Euclidean distance).
2. **Update**: each centroid moves to the mean of the points assigned to it.

Write `kmeans_step(X, C)` that performs **one** assign + update round and returns `(labels, C_new)`.

## Requirements
- `X`: `(n, d)` points, `C`: `(k, d)` current centroids.
- `labels`: int array `(n,)`. Ties go to the **lowest** centroid index.
- An empty cluster keeps its old centroid.
- Compute all distances at once, with no loop over points. A loop over the `k` clusters for the update is fine.
"""

STARTER = '''import numpy as np

def kmeans_step(X, C):
    """X: (n, d), C: (k, d). Return (labels (n,), C_new (k, d))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def kmeans_step(X, C):
    X, C = np.asarray(X, dtype=float), np.asarray(C, dtype=float)
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)   # (n, k) squared distances
    labels = d2.argmin(axis=1)                                  # argmin returns the first minimum: lowest index wins ties
    C_new = C.copy()
    for j in range(len(C)):
        pts = X[labels == j]
        if len(pts):
            C_new[j] = pts.mean(axis=0)
    return labels, C_new
'''

HINTS = [
    '`X[:, None, :] - C[None, :, :]` broadcasts to `(n, k, d)`. Sum the squares over the last axis to get an `(n, k)` distance matrix.',
    '`argmin(axis=1)` returns the **first** index of the minimum, which is exactly the tie rule asked for.',
    'Start from `C.copy()` and only overwrite centroids that received points. That handles empty clusters.',
]

_r = np.random.default_rng(1)
blobs = np.vstack([_r.normal(loc=c, scale=0.3, size=(10, 2)) for c in ((0, 0), (5, 5), (0, 5))])
TESTS = [
    (np.array([[0.0], [1.0], [10.0], [11.0]]), np.array([[0.0], [10.0]])),
    (np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([[1.0, 1.0], [1.0, -1.0]])),
    (np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 0.0], [100.0, 100.0], [50.0, 50.0]])),
    (blobs, np.array([[1.0, 1.0], [4.0, 4.0], [1.0, 4.0]])),
    (_r.normal(size=(20, 3)), _r.normal(size=(4, 3))),
    (np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]])),
]

EXPLANATION = r"""
## Why it converges
Each move can only lower the within-cluster sum of squares $\sum_i \lVert x_i - c_{\ell_i}\rVert^2$. Assignment picks the best centroid for fixed centroids, and the mean is the best centroid for fixed assignments. There are finitely many assignments, so the algorithm stops. It finds a **local** optimum, which is why initialisation (k-means++) matters.

## Memory note
The broadcast builds an `(n, k, d)` tensor. For large $n$, use $\lVert x-c\rVert^2 = \lVert x\rVert^2 - 2x\cdot c + \lVert c\rVert^2$, which needs only an $(n,k)$ matrix and one matrix product.

## Empty clusters
They happen with bad initialisation or outliers. Keeping the old centroid is the simplest policy. Libraries often re-seed it at the point farthest from its centroid instead.

## Interview follow-ups
- How do you choose $k$? (Elbow on inertia, silhouette score, or the downstream task.)
- When does k-means fail? (Non-spherical clusters, very different cluster sizes, unscaled features.)
"""
