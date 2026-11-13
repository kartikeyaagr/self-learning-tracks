import numpy as np

META = dict(title='Random forest classifier from scratch', tags=['Classic ML', 'Classification'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:04.07'], related=['bagging-classifier', 'decision-tree-classifier-cart', 'permutation-feature-importance'])

STATEMENT = r"""
A random forest is bagged decision trees with one extra source of randomness: each split only considers a random subset of the features. Write `random_forest(X_train, y_train, X_test, n_trees, max_depth, max_features, seed)` that grows the forest and returns its predictions.

Every random draw comes from **one** generator, `rng = np.random.default_rng(seed)`, in exactly the order below.

## The forest
For each tree in turn:
1. Bootstrap: `idx = rng.integers(0, n, size=n)`. The tree is grown on `X_train[idx], y_train[idx]`.
2. Grow the tree recursively from the root (depth 0), **depth-first**: build a node's left subtree completely before its right subtree.

## Growing a node
1. The node is a **leaf** if its depth equals `max_depth`, it has fewer than 2 samples, or its Gini impurity $1 - \sum_c p_c^2$ is exactly 0. A leaf makes **no** random draw.
2. Otherwise draw the features to consider: `feats = np.sort(rng.choice(d, size=max_features, replace=False))`.
3. For each $j$ in `feats` (ascending), take the sorted distinct values $v_1<\dots<v_m$ of feature $j$ in the node and thresholds $t = (v_k+v_{k+1})/2$ ascending. Left is $x_j \le t$. Score: $(n_L G_L + n_R G_R)/n$.
4. Scanning in that order, a candidate replaces the best only if its score is lower by more than `1e-12`.
5. If there's no candidate or the best score isn't below the node's Gini by more than `1e-12`, the node becomes a leaf. Otherwise split and recurse (left, then right).

## Prediction
A leaf predicts its majority label. Each test row gets the label predicted by the most trees. Every tie (in a leaf or in the vote) goes to the **smallest** label.

## Input / output
`X_train`: float `(n, d)`; `y_train`: int `(n,)`; `X_test`: float `(m, d)`; `1 <= max_features <= d`; `max_depth >= 1`. Return an int array `(m,)`.
"""

STARTER = '''import numpy as np

def random_forest(X_train, y_train, X_test, n_trees, max_depth, max_features, seed):
    """Bootstrap + per-node feature subsampling + Gini trees, majority vote. Return labels (m,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gini(y):
    _, c = np.unique(y, return_counts=True)
    p = c / len(y)
    return 1.0 - np.sum(p * p)

def majority(y):
    vals, counts = np.unique(y, return_counts=True)
    return vals[counts.argmax()]                      # first max = smallest label

def grow(X, y, depth, max_depth, max_features, rng):
    g = gini(y)
    if depth == max_depth or len(y) < 2 or g == 0.0:
        return majority(y)                            # leaves consume no randomness
    feats = np.sort(rng.choice(X.shape[1], size=max_features, replace=False))
    best = None
    for j in feats:
        v = np.unique(X[:, j])
        for t in (v[:-1] + v[1:]) / 2:
            left = X[:, j] <= t
            s = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / len(y)
            if best is None or s < best[0] - 1e-12:
                best = (s, j, t)
    if best is None or best[0] >= g - 1e-12:
        return majority(y)
    _, j, t = best
    left = X[:, j] <= t
    l = grow(X[left], y[left], depth + 1, max_depth, max_features, rng)    # left first: fixes the draw order
    r = grow(X[~left], y[~left], depth + 1, max_depth, max_features, rng)
    return (j, t, l, r)

def predict(x, node):
    while isinstance(node, tuple):
        j, t, l, r = node
        node = l if x[j] <= t else r
    return node

def random_forest(X_train, y_train, X_test, n_trees, max_depth, max_features, seed):
    X, y, Xt = np.asarray(X_train, float), np.asarray(y_train), np.asarray(X_test, float)
    rng = np.random.default_rng(seed)
    votes = []
    for _ in range(n_trees):
        idx = rng.integers(0, len(y), size=len(y))
        tree = grow(X[idx], y[idx], 0, max_depth, max_features, rng)
        votes.append([predict(x, tree) for x in Xt])
    votes = np.array(votes)                           # (n_trees, m)
    return np.array([majority(col) for col in votes.T], dtype=y.dtype)
'''

HINTS = [
    'Start from a working Gini tree. The forest adds two things: a bootstrap per tree and a feature subset per **splittable** node, both drawn from the same generator you pass down the recursion.',
    'The draw order is the whole difficulty. Check the leaf conditions **before** calling `rng.choice`, and recurse into the left child completely before starting the right one, so your stream of random numbers lines up with the reference.',
    'Inside a node: `feats = np.sort(rng.choice(d, size=max_features, replace=False))`, then the usual threshold scan over those features only. Collect every tree\'s predictions into an `(n_trees, m)` array and vote per column.',
]

_r = np.random.default_rng(61)
_X = _r.integers(0, 10, size=(60, 4)).astype(float)
_y = ((_X[:, 0] + _X[:, 2] > 9).astype(int) + (_X[:, 1] > 6)).astype(int)
_Xt = _r.integers(0, 10, size=(15, 4)).astype(float)
TESTS = [
    (np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [6.0, 1.0], [7.0, 6.0], [8.0, 2.0]]), np.array([0, 0, 0, 1, 1, 1]),
     np.array([[1.5, 4.0], [7.5, 4.0]]), 3, 2, 1, 0),
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), np.array([0, 1, 1, 2, 2]),
     np.array([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]]), 5, 3, 2, 1),
    (np.array([[1.0], [2.0], [3.0]]), np.array([4, 4, 4]), np.array([[0.0]]), 3, 3, 1, 2),
    (np.array([[1.0, 9.0], [2.0, 8.0], [3.0, 7.0], [4.0, 6.0]]), np.array([1, 0, 1, 0]), np.array([[2.5, 7.5]]), 1, 1, 1, 3),
    (_X, _y, _Xt, 10, 4, 2, 4),
    (_X, _y, _Xt, 15, 6, 1, 5),
    (_X, _y, _X[:10], 7, 2, 4, 6),
]

EXPLANATION = r"""
## Why feature subsampling
Bagging alone leaves the trees correlated: if one feature is very strong, almost every bootstrap tree splits on it at the root. The variance of an average of $B$ trees with pairwise correlation $\rho$ is $\rho\sigma^2 + (1-\rho)\sigma^2/B$, so beyond a modest $B$ only lowering $\rho$ helps. Restricting each split to `max_features` random features forces trees to explore other structure. Common defaults are $\sqrt d$ for classification and $d/3$ for regression.

## Why the draw order matters here
Real forests are only reproducible given a seed *and* an implementation. Specifying "bootstrap, then depth-first, left before right, draw only at splittable nodes" makes the answer unique. It's also a good habit when debugging any randomized algorithm: know exactly what consumes your random stream.

## Complexity
Each tree costs roughly $O(\text{max\_features}\cdot n\log n \cdot \text{depth})$ with sorted sweeps (more with the brute-force scan here). Trees are independent, so training parallelises perfectly across cores.

## Common bugs
- Drawing features before checking whether the node is a leaf, which shifts every later draw.
- Sampling features once per tree instead of once per node (that's the "random subspace" method, a different algorithm).
- Forgetting `np.sort`, so features are scanned in draw order and tie-breaking changes.

## Interview follow-ups
- Why do random forests rarely overfit as you add trees, while boosting can?
- How do you compute out-of-bag error and impurity-based feature importance, and what bias does impurity importance have?
"""
