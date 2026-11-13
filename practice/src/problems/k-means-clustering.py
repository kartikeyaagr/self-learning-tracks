import numpy as np

META = dict(title="K-means to convergence", tags=['Clustering', 'Classic ML'], diff='Medium', runtime='numpy',
            lessons=[], related=['kmeans-step', 'k-means-initialization', 'gaussian-mixture-models-em'])

STATEMENT = r"""
Lloyd's algorithm alternates an **assign** step (each point to its nearest centroid by squared Euclidean distance) and an **update** step (each centroid to the mean of its points) until the assignments stop changing.

Write `kmeans(X, C_init, max_iter)` that runs it from the given starting centroids and returns `(labels, centroids, inertia, n_iter)`.

## Loop (follow exactly)
For `it = 1, 2, ..., max_iter`:
1. Assign every point. Ties go to the **lowest** centroid index.
2. Update every centroid to the mean of its assigned points. A centroid with no points keeps its previous position.
3. If this iteration's labels are identical to the previous iteration's labels, stop. (Iteration 1 never stops this way, since there are no previous labels.)

## Output
- `labels`: int array `(n,)` from the last assign step.
- `centroids`: float array `(k, d)` after the last update step.
- `inertia`: Python `float`, $\sum_i \lVert x_i - c_{\text{labels}_i}\rVert^2$ using the returned labels and centroids.
- `n_iter`: Python `int`, the number of iterations performed (including the final one that detected convergence).

## Input
`X`: float `(n, d)`; `C_init`: float `(k, d)`; `max_iter >= 1`. Compute distances without a Python loop over points; a loop over the $k$ clusters is fine.
"""

STARTER = '''import numpy as np

def kmeans(X, C_init, max_iter):
    """Lloyd's algorithm from C_init. Return (labels, centroids, inertia, n_iter)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def kmeans(X, C_init, max_iter):
    X, C = np.asarray(X, dtype=float), np.array(C_init, dtype=float)
    prev = None
    for it in range(1, max_iter + 1):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)   # (n, k)
        labels = d2.argmin(axis=1)                                  # first minimum = lowest index
        for j in range(len(C)):
            members = X[labels == j]
            if len(members):
                C[j] = members.mean(axis=0)                         # empty clusters keep their centroid
        if prev is not None and np.array_equal(labels, prev):
            break
        prev = labels
    inertia = float(((X - C[labels]) ** 2).sum())
    return labels, C, inertia, it
'''

HINTS = [
    'Reuse the single-step logic (assign with `argmin` over an `(n, k)` distance matrix, update with a loop over clusters) and wrap it in a loop that remembers the previous labels.',
    'The convergence check comes **after** the update in each iteration, and compares the labels just computed with the labels from the iteration before.',
    'Return the loop counter of the iteration you stopped in. Compute inertia at the very end with `((X - C[labels]) ** 2).sum()`.',
]

_r = np.random.default_rng(79)
_X = np.vstack([_r.normal((0, 0), 0.6, size=(20, 2)), _r.normal((4, 0), 0.6, size=(20, 2)), _r.normal((2, 3.5), 0.6, size=(20, 2))])
TESTS = [
    (np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0]]), np.array([[0.0, 0.0], [1.0, 1.0]]), 10),
    (np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]]), np.array([[1.0], [2.0]]), 10),
    (np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]]), np.array([[1.0], [2.0]]), 1),
    (np.array([[0.0], [2.0], [4.0]]), np.array([[1.0], [3.0]]), 5),
    (np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.5, 0.0], [100.0, 100.0]]), 5),
    (np.array([[2.0, 2.0]]), np.array([[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]]), 3),
    (_X, _X[[0, 1, 2]], 100),
    (_X, np.array([[-1.0, -1.0], [5.0, 0.0], [2.0, 4.0]]), 100),
]

EXPLANATION = r"""
## Why it converges
Inertia never increases. The assign step picks, for each point, the centroid that minimises its own term; the update step picks, for each cluster, the point (the mean) that minimises the sum of squared distances to its members. There are finitely many possible assignments and inertia can't revisit a higher value, so the labels must eventually repeat. Convergence is to a **local** minimum: test 7 starts all three centroids inside one blob, and the result depends on that start.

## Complexity
$O(nkd)$ per iteration for the distance matrix. The broadcasted `(n, k, d)` difference uses $O(nkd)$ memory; for large data compute $\lVert x\rVert^2 - 2x^\top c + \lVert c\rVert^2$ instead, at the cost of some rounding.

## Common bugs
- Checking convergence before the update, so the returned centroids are one step stale.
- Comparing centroids with `==` instead of labels. Floating-point means can wobble in the last bit and never compare equal.
- Overwriting `C_init` in place (the caller's array changes). `np.array(C_init, dtype=float)` makes a copy.
- Re-initialising an empty cluster randomly, which the statement doesn't allow.

## Interview follow-ups
- How do you choose $k$? (Elbow on inertia, silhouette score, or a downstream metric; gap statistic.)
- Why is k-means a bad fit for elongated or differently sized clusters, and what would you use instead? (GMMs with full covariances, or DBSCAN for arbitrary shapes.)
"""
