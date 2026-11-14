import numpy as np

META = dict(title='Silhouette score for a clustering', tags=['Clustering', 'Metrics & Evaluation'], diff='Medium', runtime='numpy',
            related=['k-means-clustering', 'dbscan', 'pairwise-distances-with-nested-vmap'])

STATEMENT = r"""
Write `silhouette(X, labels)` returning `(per_point, mean)` with Euclidean distances.

For point $i$ in cluster $C$:
- $a_i$: mean distance from $i$ to the **other** points of $C$;
- $b_i$: the smallest, over other clusters $C'$, of the mean distance from $i$ to the points of $C'$;
- $s_i = \dfrac{b_i - a_i}{\max(a_i, b_i)}$, with $s_i = 0$ if $C$ has a single point.

Points with label `-1` are noise: they get $s_i = $ NaN, are excluded from every $a$ and $b$, and are left out of the mean. If fewer than two non-noise clusters exist, return all-NaN `per_point` and NaN `mean`.
"""

STARTER = '''import numpy as np

def silhouette(X, labels):
    """Return (per_point_scores, mean_score)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def silhouette(X, labels):
    X, lab = np.asarray(X, dtype=float), np.asarray(labels)
    n = len(lab)
    s = np.full(n, np.nan)
    ok = lab != -1
    cl = np.unique(lab[ok])
    if len(cl) < 2:
        return s, float("nan")
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    for i in np.where(ok)[0]:
        same = ok & (lab == lab[i]); same[i] = False
        if same.sum() == 0:
            s[i] = 0.0; continue
        a = D[i, same].mean()
        b = min(D[i, ok & (lab == c)].mean() for c in cl if c != lab[i])
        s[i] = (b - a) / max(a, b)
    return s, float(np.nanmean(s[ok]))
'''

HINTS = [
    'Compute the full pairwise distance matrix once with broadcasting; every a and b is then an average over a mask.',
    'For a, exclude the point itself from its own cluster before averaging.',
    'b is a minimum over the other clusters of the mean distance, not the distance to the nearest single point.',
]

_r = np.random.default_rng(9)
_X = np.vstack([_r.normal(0, 0.3, (6, 2)), _r.normal(3, 0.3, (6, 2))])
TESTS = [
    (np.array([[0.0], [1.0], [10.0], [11.0]]), np.array([0, 0, 1, 1])),
    (_X, np.array([0] * 6 + [1] * 6)),
    (_X, np.array([0, 1] * 6)),
    (np.array([[0.0], [1.0], [5.0]]), np.array([0, 0, 1])),
    (np.array([[0.0], [0.5], [9.0], [9.5], [50.0]]), np.array([0, 0, 1, 1, -1])),
    (np.array([[0.0], [1.0]]), np.array([0, 0])),
]

EXPLANATION = r"""
## Interpreting the score
$s_i$ near 1 means point $i$ is much closer to its own cluster than to any other; near 0 means it sits on a boundary; negative means it is probably in the wrong cluster. The mean over points summarises a clustering, and comparing it across choices of $k$ is a common way to choose the number of clusters.

## Limitations
Silhouette favours convex, well-separated clusters, so it can rate a correct ring-shaped clustering poorly. It needs all pairwise distances, $O(n^2)$ time and memory; for large data, compute it on a sample.

## Noise points
Density-based methods such as DBSCAN mark outliers as noise. Excluding them, as here, measures the quality of the clusters found; some tools instead treat noise as its own cluster, which changes the score.

## Interview follow-ups
- Why can k-means with a large k have a lower silhouette than a smaller k, even though its inertia is lower?
"""
