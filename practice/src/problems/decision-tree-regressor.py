import numpy as np

META = dict(title='Regression tree with squared-error splits', tags=['Regression', 'Classic ML'], diff='Medium', runtime='numpy',
            lessons=[], related=['decision-tree-classifier-cart', 'gradient-boosted-regressor'])

STATEMENT = r"""
A regression tree predicts a constant in each leaf (the mean of its training targets) and chooses splits that reduce squared error the most. For a set of targets $S$, let $\mathrm{SSE}(S) = \sum_{i\in S}(y_i - \bar y_S)^2$.

Write `regression_tree(X_train, y_train, X_test, max_depth, min_samples_split)` that grows the tree and returns `(predictions, n_leaves)`.

## Growing a node (root has depth 0)
1. The node is a **leaf** if its depth equals `max_depth` or it has fewer than `min_samples_split` samples.
2. Otherwise, enumerate candidates: for each feature $j$ in ascending order, take the sorted distinct values $v_1 < \dots < v_m$ of feature $j$ **in this node**, and the thresholds $t = (v_k + v_{k+1})/2$ in ascending order. A sample goes left if $x_j \le t$.
3. Score a candidate by $\mathrm{SSE}(\text{left}) + \mathrm{SSE}(\text{right})$.
4. Let the node's tolerance be $\tau = 10^{-9}\,(1 + \mathrm{SSE}(\text{node}))$. Scan candidates in that order. A candidate replaces the current best only if its score is lower **by more than $\tau$**.
5. If there are no candidates, or the best score is not below the node's own SSE by more than $\tau$, the node is a leaf. Otherwise split and grow both children.

## Output
- `predictions`: float array `(m,)`, the mean training target of the leaf each test row lands in.
- `n_leaves`: Python `int`.

## Input
`X_train`: float `(n, d)`; `y_train`: float `(n,)`; `X_test`: float `(m, d)`; `max_depth >= 0`; `min_samples_split >= 2`.
"""

STARTER = '''import numpy as np

def regression_tree(X_train, y_train, X_test, max_depth, min_samples_split):
    """Grow a squared-error regression tree with the stated rules. Return (predictions (m,), n_leaves)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sse(y):
    return float(np.sum((y - y.mean()) ** 2))

def grow(X, y, depth, max_depth, min_split):
    if depth == max_depth or len(y) < min_split:
        return float(y.mean())
    parent, best = sse(y), None                       # best = (score, feature, threshold)
    tol = 1e-9 * (1.0 + parent)                       # relative, so large targets don't make ties fragile
    for j in range(X.shape[1]):
        v = np.unique(X[:, j])
        for t in (v[:-1] + v[1:]) / 2:
            left = X[:, j] <= t
            s = sse(y[left]) + sse(y[~left])
            if best is None or s < best[0] - tol:
                best = (s, j, t)
    if best is None or best[0] >= parent - tol:
        return float(y.mean())
    _, j, t = best
    left = X[:, j] <= t
    return (j, t, grow(X[left], y[left], depth + 1, max_depth, min_split),
                  grow(X[~left], y[~left], depth + 1, max_depth, min_split))

def regression_tree(X_train, y_train, X_test, max_depth, min_samples_split):
    X, y = np.asarray(X_train, float), np.asarray(y_train, float)
    tree = grow(X, y, 0, max_depth, min_samples_split)   # leaf = float, split = 4-tuple

    def predict(x, node):
        while isinstance(node, tuple):
            j, t, l, r = node
            node = l if x[j] <= t else r
        return node

    def leaves(node):
        return leaves(node[2]) + leaves(node[3]) if isinstance(node, tuple) else 1

    return np.array([predict(x, tree) for x in np.asarray(X_test, float)]), leaves(tree)
'''

HINTS = [
    'Same skeleton as a classification tree: a recursive `grow` that returns a leaf value or a `(feature, threshold, left, right)` tuple. Only the impurity and the leaf value change.',
    'Minimising $\\mathrm{SSE}_L + \\mathrm{SSE}_R$ is the same as maximising the variance reduction. A node with constant targets has SSE 0, so no split can beat it by the tolerance and it becomes a leaf.',
    'Thresholds: `v = np.unique(X[:, j])` within the node, then `(v[:-1] + v[1:]) / 2`. Keep the first best unless a later candidate is lower by more than `tol = 1e-9 * (1 + sse(node))`.',
]

_r = np.random.default_rng(53)
_X = _r.uniform(0, 10, size=(50, 2))
_y = np.sin(_X[:, 0]) * 3 + (_X[:, 1] > 5) * 2 + 0.2 * _r.normal(size=50)
_Xt = _r.uniform(0, 10, size=(12, 2))
TESTS = [
    (np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([1.0, 1.2, 5.0, 5.4]), np.array([[0.0], [2.6], [10.0]]), 1, 2),
    (np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([1.0, 1.2, 5.0, 5.4]), np.array([[0.0], [1.6], [3.4]]), 5, 2),
    (np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]), np.array([7.0, 7.0, 7.0]), np.array([[2.0, 0.0]]), 4, 2),
    (np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0.0, 10.0, 0.0, 10.0]), np.array([[0.0], [3.0]]), 0, 2),
    (np.array([[0.0, 5.0], [0.0, 1.0], [1.0, 5.0], [1.0, 1.0]]), np.array([1.0, 2.0, 1.0, 2.0]), np.array([[0.3, 4.0], [0.7, 2.0]]), 3, 2),
    (np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), np.array([-1e4, -1e4, 0.0, 1e4, 2e4]), np.array([[2.5], [3.5]]), 2, 3),
    (_X, _y, _Xt, 3, 2),
    (_X, _y, _Xt, 8, 6),
]

EXPLANATION = r"""
## Why the leaf value is the mean
Inside a leaf the prediction is a constant $c$, and $\sum_i (y_i - c)^2$ is minimised at $c = \bar y$. So the SSE of a node is exactly the training loss of making it a leaf, and the split score is the loss after splitting. Greedy growth picks the largest one-step reduction.

## Computing SSE fast
$\mathrm{SSE} = \sum y_i^2 - (\sum y_i)^2/n$. Sorting a feature once and sweeping thresholds with running sums of $y$ and $y^2$ evaluates every split in $O(n)$ after an $O(n\log n)$ sort, instead of $O(n^2)$ here. The running-sum formula suffers from cancellation when $|y|$ is large relative to its spread, which is one reason the comparison uses a tolerance that scales with the node's SSE.

## Common bugs
- Forgetting the "no improvement" stop. Test 3 has constant targets; without it you'd split forever until `max_depth`.
- Using the mean squared error of each child (divided by its size) instead of the sum: that ignores how many samples each child holds.
- Predicting the median, which is the optimum for absolute error, not squared error.

## Interview follow-ups
- Why are single regression trees high-variance, and how do bagging and boosting each address that?
- How would you make the tree handle missing values without imputing them? (Learn a default direction per split, as XGBoost does.)
"""
