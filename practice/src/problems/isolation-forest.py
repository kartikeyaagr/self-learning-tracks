import numpy as np

META = dict(title='Isolation forest anomaly scores', tags=['Classic ML', 'Sampling'], diff='Medium', runtime='numpy',
            lessons=[], related=['random-forest', 'dbscan'])

STATEMENT = r"""
Anomalies are few and different, so random axis-aligned cuts isolate them in fewer steps than normal points. An isolation forest builds random trees and scores each point by its average **path length**: short paths mean anomalous.

Write `isolation_scores(X, n_trees, psi, seed)` that returns an anomaly score for every row of `X`.

## The normaliser
Let $H_m = \sum_{k=1}^{m} 1/k$ (exact harmonic number), and
$$c(m) = 2H_{m-1} - \frac{2(m-1)}{m} \ \text{ for } m \ge 2, \qquad c(1) = c(0) = 0.$$
$c(m)$ is the average path length of an unsuccessful search in a random binary tree with $m$ points.

## Building (all randomness from `rng = np.random.default_rng(seed)`, in this order)
For each tree:
1. Subsample: `rows = rng.choice(n, size=psi, replace=False)`. Depth limit $\ell = \lceil \log_2 \psi \rceil$.
2. Grow from the root (depth 0), depth-first, **left subtree before right**. A node holding points $S$ is a leaf if its depth is at least $\ell$ or $|S| \le 1$. Otherwise:
   - `q = rng.integers(d)`; let `lo, hi` be the min and max of feature `q` over $S$.
   - If `lo == hi`, the node is a leaf (no further draw).
   - Else `p = rng.uniform(lo, hi)`; points with $x_q < p$ go left, the rest go right.

## Scoring
For a point $x$ and one tree, follow the splits to a leaf at depth $e$ holding $s$ training points; its path length is $h = e + c(s)$. With $\bar h$ the mean over trees, the score is
$$\text{score}(x) = 2^{-\bar h / c(\psi)} .$$

## Input / output
`X`: float `(n, d)`; `n_trees >= 1`; `2 <= psi <= n`; `seed`: int. Score every row of `X` (including rows not in a tree's subsample). Return a float array `(n,)`.
"""

STARTER = '''import numpy as np

def isolation_scores(X, n_trees, psi, seed):
    """Isolation forest anomaly scores for every row of X. Returns float array (n,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from math import ceil, log2

def c(m):
    if m < 2:
        return 0.0
    H = sum(1.0 / k for k in range(1, m))                  # H_{m-1}
    return 2.0 * H - 2.0 * (m - 1) / m

def grow(S, depth, limit, rng):
    if depth >= limit or len(S) <= 1:
        return len(S)                                       # leaf: remember its size
    q = int(rng.integers(S.shape[1]))
    lo, hi = S[:, q].min(), S[:, q].max()
    if lo == hi:
        return len(S)
    p = rng.uniform(lo, hi)
    left = S[:, q] < p
    return (q, p, grow(S[left], depth + 1, limit, rng), grow(S[~left], depth + 1, limit, rng))

def path_length(x, node):
    e = 0
    while isinstance(node, tuple):
        q, p, l, r = node
        node = l if x[q] < p else r
        e += 1
    return e + c(node)

def isolation_scores(X, n_trees, psi, seed):
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    limit = ceil(log2(psi))
    trees = []
    for _ in range(n_trees):
        rows = rng.choice(len(X), size=psi, replace=False)
        trees.append(grow(X[rows], 0, limit, rng))
    h = np.array([[path_length(x, t) for t in trees] for x in X]).mean(axis=1)
    return 2.0 ** (-h / c(psi))
'''

HINTS = [
    'Represent a leaf by the number of training points it holds and a split by `(feature, cut, left, right)`. The leaf size is needed for the $c(s)$ correction.',
    'Check both leaf conditions (depth limit, at most one point) **before** drawing the feature, and check `lo == hi` before drawing the cut. Recurse left before right so the random stream matches.',
    'Path length = number of edges walked + `c(leaf_size)`. Average over trees, then `2 ** (-h_mean / c(psi))`. Compute $H_{m-1}$ as an exact sum, not with the $\\ln$ approximation.',
]

_r = np.random.default_rng(103)
_X = np.vstack([_r.normal(0, 1, size=(60, 2)), np.array([[6.0, 6.0], [-5.0, 4.0], [0.0, -7.0]])])
TESTS = [
    (np.array([[0.0], [0.1], [0.2], [0.3], [10.0]]), 3, 4, 0),
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [8.0, 8.0], [0.5, 0.5]]), 5, 6, 1),
    (np.array([[2.0], [2.0], [2.0]]), 2, 3, 2),
    (np.array([[0.0], [1.0]]), 4, 2, 3),
    (np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [100.0, 5.0]]), 10, 5, 4),
    (_X, 50, 32, 5),
    (_X * 1e3, 20, 48, 6),
]

EXPLANATION = r"""
## Why short paths mean anomalies
A uniform random cut between a feature's min and max lands in an empty stretch of space with probability proportional to that stretch. An isolated point is separated by most cuts that fall in the gap around it, so it reaches a singleton leaf after a few levels. A point inside a dense cluster needs many cuts before it is alone. Averaging over many random trees turns that intuition into a stable score.

## Why the normaliser
$c(\psi)$ is the expected path length for a point in a tree of $\psi$ points, so $\bar h / c(\psi)$ is roughly 1 for typical points, giving scores near $0.5$. Scores near 1 mean much shorter than typical paths (anomalies); well below 0.5 means deeply embedded points. The $c(s)$ term at a leaf estimates the depth the tree *would* have needed to isolate the point if growth hadn't stopped at the depth limit.

## Why subsample
Small subsamples ($\psi = 256$ by default in the original paper) reduce **swamping** (normal points near anomalies getting high scores) and **masking** (a dense group of anomalies hiding each other). They also keep each tree tiny: depth is capped at $\lceil\log_2\psi\rceil$.

## Complexity
Building: $O(t\,\psi\log\psi)$. Scoring $n$ points: $O(n\,t\log\psi)$. Linear in $n$ with small constants, which is why it scales to large datasets.

## Common bugs
- Drawing the feature before checking the leaf conditions, which desynchronises the random stream.
- Using `<=` for the left branch: with `uniform(lo, hi)` the cut is never `hi`, but the convention must match the statement.
- Using $\ln(m) + 0.5772$ for the harmonic number. It's the paper's approximation, but not what's asked here.

## Interview follow-ups
- Why do axis-parallel cuts create artefacts in the score map, and how does Extended Isolation Forest fix them? (Random hyperplane cuts.)
- How would you pick a decision threshold without labels? (A contamination rate: flag the top x% of scores.)
"""
