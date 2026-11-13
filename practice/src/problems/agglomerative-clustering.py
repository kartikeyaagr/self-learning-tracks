import numpy as np

META = dict(title='Agglomerative clustering with linkages', tags=['Clustering', 'Classic ML'], diff='Medium', runtime='numpy',
            lessons=[], related=['dbscan', 'k-means-clustering', 'distance-metrics'])

STATEMENT = r"""
Bottom-up hierarchical clustering starts with every point in its own cluster and repeatedly merges the two **closest** clusters. "Closest" depends on the linkage, defined from Euclidean point distances $d(a, b)$:

| linkage | distance between clusters $A$ and $B$ |
| --- | --- |
| `single` | $\min_{a\in A,\, b\in B} d(a, b)$ |
| `complete` | $\max_{a\in A,\, b\in B} d(a, b)$ |
| `average` | mean of $d(a, b)$ over all $\lvert A\rvert\,\lvert B\rvert$ pairs |

Write `agglomerative(X, n_clusters, linkage)` that merges until `n_clusters` clusters remain and returns `(labels, heights)`.

## Tie-breaking
Identify each cluster by its smallest point index. Let $\delta$ be the smallest linkage distance among all pairs. Every pair within $10^{-9}(1 + \delta)$ of $\delta$ counts as tied; among tied pairs, merge the one whose (smaller id, larger id) is lexicographically smallest.

## Output
- `labels`: int array `(n,)`. Number the final clusters `0, 1, ...` in increasing order of their smallest point index (so point 0 is always in cluster 0).
- `heights`: float array `(n - n_clusters,)`, the linkage distance of each merge in the order performed.

## Input
`X`: float `(n, d)`; `1 <= n_clusters <= n`; `linkage` is one of the strings above.
"""

STARTER = '''import numpy as np

def agglomerative(X, n_clusters, linkage):
    """Merge closest clusters until n_clusters remain. Return (labels (n,), merge heights)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def agglomerative(X, n_clusters, linkage):
    X = np.asarray(X, dtype=float)
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
    reduce = {'single': np.min, 'complete': np.max, 'average': np.mean}[linkage]
    clusters = [[i] for i in range(n)]                       # kept sorted by smallest member index
    heights = []
    while len(clusters) > n_clusters:
        pairs = [(reduce(D[np.ix_(clusters[a], clusters[b])]), a, b)
                 for a in range(len(clusters)) for b in range(a + 1, len(clusters))]
        delta = min(p[0] for p in pairs)
        # pairs are generated in (a, b) order and clusters are ordered by min index, so the first tie wins
        h, a, b = next(p for p in pairs if p[0] <= delta + 1e-9 * (1 + delta))
        heights.append(h)
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]                                      # a < b, and merged min index stays clusters[a][0]
    labels = np.empty(n, dtype=int)
    for c, members in enumerate(clusters):
        labels[members] = c
    return labels, np.array(heights, dtype=float)
'''

HINTS = [
    'Keep a list of clusters (lists of point indices) ordered by their smallest member. Merging $A$ into the earlier of the two keeps that order, so list position equals the required numbering at the end.',
    'Precompute the full point distance matrix once. The linkage distance of two clusters is `min`, `max` or `mean` of the sub-block `D[np.ix_(A, B)]`.',
    'Generate candidate pairs in `(a, b)` order with `a < b`, find the minimum distance $\\delta$, then take the **first** pair within `1e-9 * (1 + delta)` of it.',
]

_r = np.random.default_rng(97)
_X = np.vstack([_r.normal((0, 0), 0.5, size=(8, 2)), _r.normal((4, 0), 0.5, size=(8, 2)), _r.normal((2, 3), 0.5, size=(6, 2))])
TESTS = [
    (np.array([[0.0], [1.0], [5.0], [6.0], [20.0]]), 2, 'single'),
    (np.array([[0.0, 0.0], [0.0, 1.0], [4.0, 0.0], [4.0, 1.0]]), 2, 'complete'),
    (np.array([[0.0], [1.0], [2.0], [3.0]]), 1, 'single'),
    (np.array([[0.0], [1.0], [2.0], [3.0]]), 2, 'single'),
    (np.array([[0.0], [1.0], [2.0], [3.0], [4.5]]), 2, 'complete'),
    (np.array([[0.0], [1.0], [2.0], [3.0], [4.5]]), 2, 'average'),
    (np.array([[7.0, 7.0]]), 1, 'average'),
    (_X, 3, 'average'),
    (_X, 4, 'single'),
    (_X * 1000.0, 3, 'complete'),
]

EXPLANATION = r"""
## How the linkages differ
- **Single** linkage merges on the closest pair, so it follows chains: test 4 is a line of equally spaced points, and the tie rule decides the order of merges. It's equivalent to cutting a minimum spanning tree.
- **Complete** linkage merges on the farthest pair, preferring compact clusters of similar diameter.
- **Average** linkage (UPGMA) sits in between and is less sensitive to single outlying points.
Tests 5 and 6 use the same points: the far point at 4.5 changes which merges happen under complete vs average linkage.

## Complexity
This direct version recomputes every pair each round: $O(n^2)$ pairs, each costing up to $O(n^2)$ to reduce, over $n$ rounds. Real implementations keep a cluster distance matrix and update it in $O(n)$ per merge with the Lance-Williams formula (for average: $d(A\cup B, C) = \frac{|A|d(A,C) + |B|d(B,C)}{|A|+|B|}$), giving $O(n^2\log n)$ with a heap, or $O(n^2)$ for single linkage via the MST.

## Common bugs
- Deleting the earlier cluster instead of the later one, which shifts the numbering.
- Using squared distances with average linkage: that changes which clusters are closest (the mean of squares isn't the square of the mean).
- Numbering clusters by merge order instead of smallest member.

## Interview follow-ups
- How do you choose the number of clusters from a dendrogram? (Cut where the merge heights jump.)
- Why is Ward linkage often the best default, and what does it minimise? (The increase in total within-cluster variance, the same objective as k-means.)
"""
