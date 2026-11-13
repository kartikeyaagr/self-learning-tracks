import numpy as np

META = dict(title='K-means++ seeding', tags=['Clustering', 'Sampling'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:10.05'], related=['k-means-clustering', 'kmeans-step'])

STATEMENT = r"""
K-means is sensitive to where it starts. K-means++ picks starting centroids from the data so that they tend to be spread out: each new centre is sampled with probability proportional to its **squared distance to the nearest centre chosen so far**.

Write `kmeans_pp(X, k, seed)` that returns the **row indices** of the chosen centres, in the order they were chosen.

## Procedure (follow exactly)
1. `rng = np.random.default_rng(seed)`.
2. First centre: `rng.integers(n)`.
3. Keep $D_i^2$, the squared Euclidean distance from row $i$ to its nearest chosen centre.
4. Each later centre: `rng.choice(n, p=D2 / D2.sum())`. Then update $D^2$ with the new centre.

## Input / output
- `X`: float `(n, d)` with at least `k` distinct rows; `1 <= k <= n`; `seed`: int.
- Return an int array `(k,)`. No index repeats (a chosen row has $D^2 = 0$, so it can't be drawn again).
- Updating $D^2$ must be $O(nd)$ per new centre: don't recompute distances to every earlier centre.
"""

STARTER = '''import numpy as np

def kmeans_pp(X, k, seed):
    """Indices of k-means++ seeds in the order chosen. Returns an int array (k,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def kmeans_pp(X, k, seed):
    X = np.asarray(X, dtype=float)
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(n))]
    d2 = ((X - X[idx[0]]) ** 2).sum(axis=1)
    for _ in range(1, k):
        i = int(rng.choice(n, p=d2 / d2.sum()))            # far points are proportionally more likely
        idx.append(i)
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(axis=1)) # nearest-centre distance only ever shrinks
    return np.array(idx)
'''

HINTS = [
    'You only need one vector of length $n$: each point\'s squared distance to its **nearest** chosen centre.',
    'After choosing centre $i$, the nearest-centre distance becomes `np.minimum(d2, dist2_to(X[i]))`. Nothing else needs recomputing.',
    'Use the generator calls exactly as stated: `rng.integers(n)` once, then `rng.choice(n, p=d2 / d2.sum())` for each further centre. Any other call order draws different numbers.',
]

_r = np.random.default_rng(83)
_X = np.vstack([_r.normal((0, 0), 0.5, size=(30, 2)), _r.normal((6, 0), 0.5, size=(30, 2)), _r.normal((3, 5), 0.5, size=(30, 2))])
TESTS = [
    (np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.0, 10.1]]), 2, 0),
    (np.array([[0.0], [1.0], [2.0], [50.0]]), 3, 1),
    (np.array([[3.0, 4.0]]), 1, 2),
    (np.array([[0.0], [0.0], [0.0], [5.0]]), 2, 3),
    (np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [9.0, 8.0]]), 3, 4),
    (_X, 3, 5),
    (_X, 6, 6),
    (_X * 1e4, 4, 7),
]

EXPLANATION = r"""
## Why squared distance
Sampling proportionally to $D^2$ (rather than $D$, or picking the farthest point) gives the $O(\log k)$-competitive guarantee of Arthur and Vassilvitskii (2007): the expected inertia of the seeding alone is within $8(\ln k + 2)$ of the optimum. The farthest-point rule has no such guarantee and is easily hijacked by outliers; $D^2$ sampling still favours far regions but outliers are just one point among many.

## Complexity
$O(nkd)$ total: each new centre costs one pass to compute distances to it and one `minimum`. `rng.choice` with `p` is $O(n)$.

## Common bugs
- Recomputing the distance to all previous centres each round, making it $O(nk^2d)$.
- Using `D` instead of `D**2` as weights.
- Sampling with a different generator call (for example `rng.random()` plus `searchsorted`). Mathematically equivalent, but it consumes the stream differently, so the indices differ.
- Duplicate rows: test 4 has three identical rows; once one is chosen the others have $D^2 = 0$ too.

## Interview follow-ups
- K-means++ needs $k$ sequential passes. How does k-means|| (scalable k-means++) parallelise it? (Oversample about $\ell = O(k)$ points per round for a few rounds, then recluster the candidates.)
- Even with good seeding, why do libraries run k-means several times (`n_init`) and keep the best inertia?
"""
