import numpy as np

META = dict(title='CART decision tree classifier (Gini)', tags=['Classification', 'Classic ML'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:01.11'], related=['decision-tree-regressor', 'random-forest', 'bagging-classifier'])

STATEMENT = r"""
Grow a binary classification tree greedily: at each node, try every axis-aligned split and keep the one that makes the children purest. Impurity is Gini, $G = 1 - \sum_c p_c^2$, where $p_c$ is the fraction of the node's samples in class $c$.

Write `cart_classifier(X_train, y_train, X_test, max_depth, min_samples_split)` that grows the tree and returns `(predictions, n_leaves)`.

## Growing a node (root has depth 0)
1. The node is a **leaf** if its depth equals `max_depth`, it has fewer than `min_samples_split` samples, or its Gini is exactly 0.
2. Otherwise, enumerate candidates: for each feature $j$ in ascending order, take the sorted distinct values $v_1 < \dots < v_m$ of feature $j$ **in this node**, and the thresholds $t = (v_k + v_{k+1})/2$ in ascending order. A sample goes left if $x_j \le t$, right otherwise.
3. Score a candidate by the weighted child impurity $\frac{n_L G_L + n_R G_R}{n}$.
4. Scan candidates in the order above. A candidate replaces the current best only if its score is lower **by more than `1e-12`** (so near-ties keep the earlier candidate).
5. If there are no candidates, or the best score is not below the node's own Gini by more than `1e-12`, make the node a leaf. Otherwise split and grow both children.

## Leaves and output
- A leaf predicts the majority class of its training samples; ties go to the **smallest** label.
- `predictions`: int array `(m,)`, the leaf label reached by each row of `X_test`.
- `n_leaves`: Python `int`, the number of leaves in the tree.

## Input
`X_train`: float `(n, d)`; `y_train`: int `(n,)`; `X_test`: float `(m, d)`; `max_depth >= 0`; `min_samples_split >= 2`.
"""

STARTER = '''import numpy as np

def cart_classifier(X_train, y_train, X_test, max_depth, min_samples_split):
    """Grow a Gini CART tree with the stated rules. Return (predictions (m,), n_leaves)."""
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

def grow(X, y, depth, max_depth, min_split):
    g = gini(y)
    if depth == max_depth or len(y) < min_split or g == 0.0:
        return majority(y)
    best = None                                       # (score, feature, threshold)
    for j in range(X.shape[1]):
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
    return (j, t, grow(X[left], y[left], depth + 1, max_depth, min_split),
                  grow(X[~left], y[~left], depth + 1, max_depth, min_split))

def cart_classifier(X_train, y_train, X_test, max_depth, min_samples_split):
    X, y = np.asarray(X_train, float), np.asarray(y_train)
    tree = grow(X, y, 0, max_depth, min_samples_split)    # a leaf is a label, a split is a 4-tuple

    def predict(x, node):
        while isinstance(node, tuple):
            j, t, l, r = node
            node = l if x[j] <= t else r
        return node

    def leaves(node):
        return leaves(node[2]) + leaves(node[3]) if isinstance(node, tuple) else 1

    preds = np.array([predict(x, tree) for x in np.asarray(X_test, float)], dtype=y.dtype)
    return preds, leaves(tree)
'''

HINTS = [
    'Write three small helpers first: Gini of a label array, majority label with the tie rule, and a recursive `grow(X, y, depth)` that returns either a leaf label or a `(feature, threshold, left, right)` tuple.',
    'Candidate thresholds come from `v = np.unique(X[:, j])` restricted to the node\'s rows: `(v[:-1] + v[1:]) / 2`. A feature with one distinct value has no candidates.',
    'Track `best = (score, j, t)` and replace it only when `score < best_score - 1e-12`. After the scan, split only if `best_score < gini(node) - 1e-12`; otherwise return a leaf.',
]

_r = np.random.default_rng(47)
_X = _r.integers(0, 10, size=(60, 3)).astype(float)
_y = ((_X[:, 0] > 4) & (_X[:, 1] < 7)).astype(int) + 2 * (_X[:, 2] > 7)
_Xt = _r.integers(0, 10, size=(15, 3)).astype(float)
TESTS = [
    (np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1]), np.array([[0.0], [2.4], [2.6], [9.0]]), 3, 2),
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([0, 1, 1, 0]), np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.9]]), 2, 2),
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([0, 1, 1, 0]), np.array([[0.0, 0.0], [1.0, 0.0]]), 0, 2),
    (np.array([[1.0], [1.0], [2.0], [2.0]]), np.array([3, 5, 3, 5]), np.array([[1.0], [2.0]]), 4, 2),
    (np.array([[5.0], [5.0], [5.0]]), np.array([2, 1, 1]), np.array([[0.0]]), 3, 2),
    (np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]), np.array([0, 0, 1, 1, 0, 0]), np.array([[3.5], [5.5]]), 3, 3),
    (_X, _y, _Xt, 3, 2),
    (_X, _y, _Xt, 10, 5),
]

EXPLANATION = r"""
## Why greedy splitting works (well enough)
Finding the optimal tree is NP-hard, so CART picks the locally best split and recurses. Gini measures the probability that two samples drawn from the node with replacement disagree; minimising the weighted child Gini is equivalent to maximising the impurity decrease. Test 2 (XOR) shows the greedy weakness: no single split reduces impurity, so the root stays a leaf even though depth 2 could fit it perfectly.

## Determinism
Two different splits often have exactly the same score, and floating-point rounding can make mathematically equal scores differ in the last bit. Comparing with a `1e-12` margin makes "first candidate wins" well defined regardless of how you compute the Gini.

## Complexity
Brute force here is $O(d \cdot n^2)$ per node (each of up to $n$ thresholds scans the node). Sorting each feature once and sweeping thresholds with running class counts brings it to $O(d\, n \log n)$ per node, which is what real implementations do.

## Common bugs
- Computing thresholds from the whole training set instead of the node's samples.
- Splitting when the gain is zero. With `min_samples_split` and `max_depth` large, the tree should still stop on test 4, where no feature separates the labels.
- Using `<` for the left branch while thresholds are midpoints. It doesn't matter for training rows but does for test rows sitting exactly on a threshold.

## Interview follow-ups
- Gini vs entropy: when do they choose different splits, and does it matter in practice?
- How does cost-complexity pruning choose a subtree, and why is pruning better than early stopping on test 2-style interactions?
"""
