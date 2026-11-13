import numpy as np

META = dict(title='Gradient-boosted regression trees', tags=['Regression', 'Classic ML'], diff='Hard', runtime='numpy',
            lessons=[], related=['decision-tree-regressor', 'adaboost'])

STATEMENT = r"""
Gradient boosting with squared loss builds $F(x) = F_0 + \nu \sum_{t=1}^{T} h_t(x)$, where each small regression tree $h_t$ is fit to the **residuals** $y - F$ of the model so far and $\nu$ is the learning rate (shrinkage).

Write `gradient_boosting(X_train, y_train, X_test, n_rounds, learning_rate, max_depth)` that returns `(test_predictions, train_mse)`.

## Procedure
1. $F_0$ is the mean of `y_train`. Initialise training and test predictions to $F_0$.
2. Each round: compute residuals $r = y - F(X_\text{train})$, grow a regression tree on $(X_\text{train}, r)$ with the rules below, then add `learning_rate` times its predictions to both the training and test predictions.
3. `train_mse[0]` is the training MSE of $F_0$; `train_mse[t]` is the training MSE after round $t$.

## Tree rules (root depth 0)
- A node is a leaf if its depth equals `max_depth` or it has fewer than 2 samples. A leaf predicts the mean residual of its samples.
- Candidates: for each feature $j$ ascending, the sorted distinct values $v_1<\dots<v_m$ of feature $j$ in the node, thresholds $t = (v_k+v_{k+1})/2$ ascending; left is $x_j \le t$.
- Score: $\mathrm{SSE}(\text{left}) + \mathrm{SSE}(\text{right})$, where $\mathrm{SSE}(S) = \sum_{i\in S}(r_i - \bar r_S)^2$.
- With $\tau = 10^{-9}(1 + \mathrm{SSE}(\text{node}))$: a candidate replaces the best only if lower by more than $\tau$; the node splits only if the best score is below its own SSE by more than $\tau$. Otherwise it's a leaf.

## Input / output
- `X_train`: float `(n, d)`; `y_train`: float `(n,)`; `X_test`: float `(m, d)`; `n_rounds >= 0`; `0 < learning_rate <= 1`; `max_depth >= 1`.
- `test_predictions`: float `(m,)`; `train_mse`: float `(n_rounds + 1,)`.
"""

STARTER = '''import numpy as np

def gradient_boosting(X_train, y_train, X_test, n_rounds, learning_rate, max_depth):
    """Squared-loss gradient boosting with regression trees. Return (test preds (m,), train MSE (n_rounds+1,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sse(r):
    return float(np.sum((r - r.mean()) ** 2))

def grow(X, r, depth, max_depth):
    if depth == max_depth or len(r) < 2:
        return float(r.mean())
    parent, best = sse(r), None
    tol = 1e-9 * (1.0 + parent)
    for j in range(X.shape[1]):
        v = np.unique(X[:, j])
        for t in (v[:-1] + v[1:]) / 2:
            left = X[:, j] <= t
            s = sse(r[left]) + sse(r[~left])
            if best is None or s < best[0] - tol:
                best = (s, j, t)
    if best is None or best[0] >= parent - tol:
        return float(r.mean())
    _, j, t = best
    left = X[:, j] <= t
    return (j, t, grow(X[left], r[left], depth + 1, max_depth), grow(X[~left], r[~left], depth + 1, max_depth))

def predict(tree, X):
    def one(x):
        node = tree
        while isinstance(node, tuple):
            j, t, l, rt = node
            node = l if x[j] <= t else rt
        return node
    return np.array([one(x) for x in X])

def gradient_boosting(X_train, y_train, X_test, n_rounds, learning_rate, max_depth):
    X, y, Xt = np.asarray(X_train, float), np.asarray(y_train, float), np.asarray(X_test, float)
    F, Ft = np.full(len(y), y.mean()), np.full(len(Xt), y.mean())
    mse = [np.mean((y - F) ** 2)]
    for _ in range(n_rounds):
        tree = grow(X, y - F, 0, max_depth)            # residuals = negative gradient of squared loss / 2
        F = F + learning_rate * predict(tree, X)
        Ft = Ft + learning_rate * predict(tree, Xt)
        mse.append(np.mean((y - F) ** 2))
    return Ft, np.array(mse)
'''

HINTS = [
    'Get a regression tree working on its own first (fit on `(X, r)`, predict on any matrix). Boosting is then a short loop around it.',
    'For squared loss $\\tfrac12(y - F)^2$, the negative gradient with respect to $F$ is exactly the residual $y - F$. That\'s why each tree is fit to residuals.',
    'Keep two running prediction vectors, `F` for training rows and `Ft` for test rows, both starting at `y.mean()`. Each round: `tree = grow(X, y - F)`, then add `learning_rate * predict(tree, ·)` to each and append the new training MSE.',
]

_r = np.random.default_rng(71)
_X = _r.uniform(0, 6, size=(40, 2))
_y = np.sin(_X[:, 0]) + 0.5 * _X[:, 1] + 0.1 * _r.normal(size=40)
_Xt = _r.uniform(0, 6, size=(8, 2))
TESTS = [
    (np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([1.0, 2.0, 3.0, 10.0]), np.array([[1.5], [3.9]]), 1, 1.0, 1),
    (np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([1.0, 2.0, 3.0, 10.0]), np.array([[1.5], [3.9]]), 3, 0.5, 1),
    (np.array([[1.0], [2.0]]), np.array([5.0, 7.0]), np.array([[0.0]]), 0, 0.1, 2),
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([4.0, 4.0, 4.0, 4.0]), np.array([[0.5, 0.5]]), 5, 0.3, 2),
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([0.0, 1.0, 1.0, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0]]), 4, 1.0, 2),
    (_X, _y, _Xt, 20, 0.1, 2),
    (_X, _y * 1000.0 - 5e4, _Xt, 10, 0.3, 3),
]

EXPLANATION = r"""
## Why residuals
Gradient boosting is gradient descent in function space. At each round, the loss $\sum_i L(y_i, F(x_i))$ decreases fastest if every $F(x_i)$ moves along $-\partial L/\partial F(x_i)$. A tree fit to those negative gradients is the best approximation of that step that also generalises to new $x$. For $L = \tfrac12(y-F)^2$ the negative gradient is the residual, so "fit the residuals" is the squared-loss special case. Other losses only change what the tree is fit to (and, ideally, the leaf values).

## Why shrinkage
With `learning_rate = 1` (test 1) one depth-1 tree can fit the outlier at $x=4$ completely. Smaller steps make each tree a cautious correction, which reliably improves test error at the price of more rounds. Training MSE never increases for squared loss with mean-valued leaves, which you can see in `train_mse`.

## Complexity
$T$ trees, each $O(d\, n\log n)$ per level with sorted sweeps (brute force here is slower). Unlike random forests, rounds are sequential; parallelism happens within a tree.

## Common bugs
- Fitting each tree to `y` instead of the current residuals.
- Applying the learning rate to the training predictions but not the test predictions (or vice versa).
- Starting from $F_0 = 0$ instead of the mean, which wastes the first rounds learning the offset.

## Interview follow-ups
- What changes for binary classification with log loss? (Trees fit $y - \sigma(F)$, and leaf values use a Newton step: sum of gradients over sum of Hessians.)
- How do subsampling rows per round (stochastic gradient boosting) and early stopping on a validation set help?
"""
