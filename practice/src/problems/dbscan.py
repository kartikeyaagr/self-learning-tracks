import numpy as np

META = dict(title='DBSCAN density clustering', tags=['Clustering', 'Classic ML'], diff='Medium', runtime='numpy',
            lessons=[], related=['k-means-clustering', 'agglomerative-clustering', 'distance-metrics'])

STATEMENT = r"""
DBSCAN grows clusters from dense regions and labels isolated points as noise. With radius $\varepsilon$ and threshold `min_samples`:

- The **neighbourhood** of point $i$ is every point $j$ (including $i$ itself) with Euclidean distance $\lVert x_i - x_j\rVert \le \varepsilon$.
- $i$ is a **core point** if its neighbourhood has at least `min_samples` points.
- A cluster is a maximal set of core points connected through each other's neighbourhoods, plus every non-core point that lies in the neighbourhood of one of those core points (**border points**).
- Everything else is **noise**, label `-1`.

Write `dbscan(X, eps, min_samples)` that returns an int array of labels `(n,)`.

## Numbering (so the answer is unique)
- Clusters are numbered `0, 1, 2, ...` in increasing order of their **lowest-index core point**.
- A border point within $\varepsilon$ of core points from several clusters gets the **smallest** cluster number among them.

## Input
`X`: float `(n, d)`; `eps > 0`; `min_samples >= 1`. Build the neighbourhoods with one vectorised distance computation.
"""

STARTER = '''import numpy as np

def dbscan(X, eps, min_samples):
    """Return DBSCAN labels (n,): clusters 0, 1, ... by lowest core index, noise -1."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from collections import deque

def dbscan(X, eps, min_samples):
    X = np.asarray(X, dtype=float)
    n = len(X)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    nbrs = d2 <= eps * eps                                   # (n, n), diagonal True: a point counts itself
    core = nbrs.sum(axis=1) >= min_samples
    labels = np.full(n, -1)
    cid = 0
    for i in range(n):                                       # seeds in index order give the required numbering
        if not core[i] or labels[i] != -1:
            continue
        labels[i] = cid
        queue = deque([i])
        while queue:                                         # expand through core points only
            p = queue.popleft()
            for q in np.flatnonzero(nbrs[p]):
                if labels[q] == -1:                          # first cluster to reach a border point keeps it
                    labels[q] = cid
                    if core[q]:
                        queue.append(q)
        cid += 1
    return labels
'''

HINTS = [
    'Compute the full boolean neighbourhood matrix `d2 <= eps**2` once. The core mask is then a row sum.',
    'Scan points in index order. An unlabelled core point starts a new cluster; expand it with a queue, enqueueing only **core** neighbours but labelling every unlabelled neighbour.',
    'Because each cluster is expanded completely before the next one starts, "only label points that are still -1" automatically gives border points the smallest cluster number.',
]

_r = np.random.default_rng(89)
_X = np.vstack([_r.normal((0, 0), 0.3, size=(25, 2)), _r.normal((3, 3), 0.3, size=(25, 2)), _r.uniform(-2, 5, size=(8, 2))])
TESTS = [
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [10.0, 10.0]]), 1.0, 3),
    (np.array([[0.0], [1.0], [2.0], [5.0], [6.0], [7.0], [20.0]]), 1.0, 2),
    (np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), 1.0, 3),
    (np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), 0.5, 2),
    (np.array([[0.0], [0.3], [0.6], [1.0], [2.0], [3.0], [3.4], [3.7], [4.0]]), 1.0, 4),
    (np.array([[5.0, 5.0]]), 1.0, 1),
    (_X, 0.5, 4),
    (_X * 100.0, 50.0, 4),
]

EXPLANATION = r"""
## Why it works
Core points are where the estimated density (count within $\varepsilon$) is high. Two core points within $\varepsilon$ of each other are "density-connected", and the connected components of that graph are the clusters. A breadth-first search from any core point visits exactly its component. Border points are attached to a cluster but never expand it, which stops one cluster bleeding into another through a thin bridge of low-density points.

## The border-point ambiguity
In test 5 the point at 2.0 has only 3 neighbours, so it isn't core, but is within $\varepsilon$ of core points from both sides. The original algorithm gives it to whichever cluster reaches it first, which depends on the visiting order. Fixing the order (seeds by index, full expansion before the next seed) makes the answer unique: the smallest cluster number wins.

## Complexity
The dense distance matrix is $O(n^2 d)$ time and $O(n^2)$ memory; the search is $O(n^2)$ in the worst case. A spatial index (KD-tree, ball tree, grid) makes neighbourhood queries roughly $O(\log n)$ in low dimensions.

## Common bugs
- Not counting the point itself in its neighbourhood (`min_samples` is off by one).
- Enqueueing border points, which lets clusters grow through non-core points.
- Comparing `sqrt(d2) <= eps` and `d2 <= eps**2` inconsistently. Both are fine mathematically; pick one.

## Interview follow-ups
- How would you choose $\varepsilon$? (Plot each point's distance to its `min_samples`-th neighbour, sorted, and look for the knee.)
- What breaks when clusters have very different densities, and how do OPTICS or HDBSCAN fix it?
"""
