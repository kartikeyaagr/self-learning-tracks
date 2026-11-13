import numpy as np

META = dict(title='k-nearest-neighbours classifier', tags=['Classification', 'Classic ML'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.03'], related=['distance-metrics', 'cosine-similarity-matrix'])

STATEMENT = r"""
A k-NN classifier has no training step: to label a query point, find the $k$ training points closest to it and take a majority vote of their labels.

Write `knn_predict(X_train, y_train, X_test, k)` that returns one predicted label per test point.

## Rules
- Distance is Euclidean.
- **Distance ties**: if several training points are equally far, the one with the **lower training index** counts as nearer.
- **Vote ties**: if two or more labels share the highest vote count among the $k$ neighbours, predict the **smallest** of those labels.
- Compute all test-to-train distances at once (broadcasting), with no Python loop over pairs of points.

## Input / output
- `X_train`: float `(n, d)`; `y_train`: int `(n,)`; `X_test`: float `(m, d)`; `1 <= k <= n`.
- Return an int array `(m,)`.
"""

STARTER = '''import numpy as np

def knn_predict(X_train, y_train, X_test, k):
    """Majority vote of the k nearest training points (ties: lower index, then smaller label)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def knn_predict(X_train, y_train, X_test, k):
    X_train, X_test, y_train = np.asarray(X_train, float), np.asarray(X_test, float), np.asarray(y_train)
    d2 = ((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=-1)   # (m, n); sqrt doesn't change the order
    nn = np.argsort(d2, axis=1, kind='stable')[:, :k]                    # stable: equal distances keep index order
    out = []
    for labels in y_train[nn]:
        vals, counts = np.unique(labels, return_counts=True)            # vals ascending
        out.append(vals[counts.argmax()])                               # argmax takes the first max: smallest label
    return np.array(out, dtype=y_train.dtype)
'''

HINTS = [
    'You never need the square root: squared distances sort in the same order.',
    "`X_test[:, None, :] - X_train[None, :, :]` has shape `(m, n, d)`. Sum the squares over the last axis, then `np.argsort(..., kind='stable')` keeps equal distances in index order.",
    'For each row of `y_train[nn]`, `np.unique(row, return_counts=True)` returns labels sorted ascending, so `vals[counts.argmax()]` picks the smallest label among the tied winners.',
]

_r = np.random.default_rng(17)
_Xa = np.vstack([_r.normal(loc=0.0, size=(15, 2)), _r.normal(loc=2.5, size=(15, 2)), _r.normal(loc=(0.0, 4.0), size=(15, 2))])
_ya = np.repeat([0, 1, 2], 15)
TESTS = [
    (np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]]), np.array([0, 0, 1, 1]), np.array([[0.5, 0.0], [5.0, 4.0]]), 1),
    (np.array([[0.0], [1.0], [2.0], [10.0], [11.0]]), np.array([3, 3, 7, 7, 7]), np.array([[1.5], [9.0]]), 3),
    (np.array([[0.0], [2.0]]), np.array([5, 1]), np.array([[1.0]]), 1),
    (np.array([[0.0], [2.0]]), np.array([5, 1]), np.array([[1.0]]), 2),
    (np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]]), np.array([4, 2, 2, 4]), np.array([[1.0, 1.0], [0.0, 0.0]]), 3),
    (np.array([[-100.0, 3.0], [100.0, 3.0], [0.0, -50.0]]), np.array([0, 1, 2]), np.array([[0.0, 0.0], [99.0, 0.0]]), 3),
    (_Xa, _ya, _r.normal(loc=1.5, scale=2.0, size=(20, 2)), 5),
    (_Xa, _ya, _Xa[::4], 1),
]

EXPLANATION = r"""
## Why it works
k-NN estimates $P(y = c \mid x)$ by the fraction of the $k$ nearest training points with label $c$. As $n \to \infty$ with $k \to \infty$ and $k/n \to 0$, that estimate converges to the true posterior, so the classifier approaches the Bayes-optimal one. For $k = 1$ the asymptotic error is at most twice the Bayes error.

## Complexity
Brute force costs $O(mnd)$ time for the distances and $O(mn\log n)$ for the sorts. `np.argpartition` brings the selection down to $O(mn)$, but it doesn't respect the index tie rule. KD-trees and ball trees help in low dimensions; approximate methods (HNSW, IVF) are used in high dimensions.

## Common bugs
- Using `np.argsort` without `kind='stable'`. The default quicksort can reorder equal distances (test 3 has a point exactly between two training points).
- `np.bincount(labels).argmax()` is a neat vote, but it fails for negative labels and allocates up to the largest label.
- Including the query point itself when evaluating on the training set (test 8 does that on purpose: with $k=1$ every training point predicts its own label).

## Interview follow-ups
- How should you choose $k$, and why do odd values help for binary problems?
- Why do features need scaling for k-NN, and what happens to nearest-neighbour distances in very high dimensions?
"""
