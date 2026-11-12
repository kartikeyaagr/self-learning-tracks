import numpy as np

META = dict(title='Bagging decision stumps', tags=['Classic ML', 'Classification'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:04.07'], related=['random-forest', 'decision-tree-classifier-cart'])

STATEMENT = r"""
Bagging (bootstrap aggregating) trains the same weak learner on many bootstrap resamples of the training set and lets them vote. Here the weak learner is a **decision stump**: a tree with a single Gini split.

Write `bagged_stumps(X_train, y_train, X_test, n_estimators, seed)` that returns the ensemble's predictions for `X_test`.

## Bootstrap (follow exactly)
`rng = np.random.default_rng(seed)`. For each estimator in turn, draw `idx = rng.integers(0, n, size=n)` and train a stump on `X_train[idx], y_train[idx]`.

## Fitting a stump on a sample
- Gini of a set: $1 - \sum_c p_c^2$.
- Candidates: for each feature $j$ ascending, the sorted distinct values $v_1<\dots<v_m$ of feature $j$ in the sample, and thresholds $t = (v_k + v_{k+1})/2$ ascending. Left is $x_j \le t$.
- Score: $(n_L G_L + n_R G_R)/n$. Scanning in order, a candidate replaces the best only if its score is lower by more than `1e-12`.
- If there's no candidate or the best score isn't below the sample's Gini by more than `1e-12`, the stump is constant: it predicts the sample's majority label everywhere.
- Otherwise each side predicts its majority label.
- Majority ties go to the **smallest** label.

## Voting
Each test row gets the label predicted by the most stumps; ties go to the **smallest** label.

## Input / output
`X_train`: float `(n, d)`; `y_train`: int `(n,)`; `X_test`: float `(m, d)`; `n_estimators >= 1`; `seed`: int. Return an int array `(m,)`.
"""

STARTER = '''import numpy as np

def bagged_stumps(X_train, y_train, X_test, n_estimators, seed):
    """Majority vote of Gini stumps fit on bootstrap samples. Return labels (m,)."""
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

def fit_stump(X, y):
    """Return (feature, threshold, left_label, right_label); feature None means constant."""
    best = None
    for j in range(X.shape[1]):
        v = np.unique(X[:, j])
        for t in (v[:-1] + v[1:]) / 2:
            left = X[:, j] <= t
            s = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / len(y)
            if best is None or s < best[0] - 1e-12:
                best = (s, j, t)
    if best is None or best[0] >= gini(y) - 1e-12:
        m = majority(y)
        return None, 0.0, m, m
    _, j, t = best
    left = X[:, j] <= t
    return j, t, majority(y[left]), majority(y[~left])

def bagged_stumps(X_train, y_train, X_test, n_estimators, seed):
    X, y, Xt = np.asarray(X_train, float), np.asarray(y_train), np.asarray(X_test, float)
    rng = np.random.default_rng(seed)
    votes = []
    for _ in range(n_estimators):
        idx = rng.integers(0, len(y), size=len(y))    # sample n rows with replacement
        j, t, lo, hi = fit_stump(X[idx], y[idx])
        votes.append(np.full(len(Xt), lo) if j is None else np.where(Xt[:, j] <= t, lo, hi))
    votes = np.array(votes)                           # (n_estimators, m)
    return np.array([majority(col) for col in votes.T], dtype=y.dtype)
'''

HINTS = [
    'Split the work into `fit_stump(X, y)` returning `(feature, threshold, left_label, right_label)` and an outer loop that draws bootstrap indices and collects each stump\'s predictions on `X_test`.',
    'Draw the bootstrap with exactly `rng.integers(0, n, size=n)`, once per estimator, before fitting it. A bootstrap sample can contain only one class; then the stump is constant.',
    'Stack the predictions into an `(n_estimators, m)` array and take a column-wise majority with `np.unique(col, return_counts=True)`, whose first maximum is the smallest label.',
]

_r = np.random.default_rng(59)
_X = _r.integers(0, 8, size=(40, 2)).astype(float)
_y = np.where(_X[:, 0] + _X[:, 1] > 7, 1, 0)
_Xt = _r.integers(0, 8, size=(12, 2)).astype(float)
TESTS = [
    (np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]), np.array([0, 0, 0, 1, 1, 1]), np.array([[0.0], [3.4], [3.6], [7.0]]), 5, 0),
    (np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 2.0]]), np.array([1, 1, 2, 2]), np.array([[0.5, 0.5], [2.5, 2.5]]), 3, 1),
    (np.array([[1.0], [2.0]]), np.array([4, 4]), np.array([[0.0], [9.0]]), 2, 2),
    (np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 1, 0, 1]), np.array([[1.0], [4.0]]), 1, 3),
    (np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), np.array([2, 2, 0, 1, 1]), np.array([[1.5], [3.0], [4.5]]), 7, 4),
    (_X, _y, _Xt, 25, 5),
    (_X * 1000.0, _y, _Xt * 1000.0, 10, 6),
]

EXPLANATION = r"""
## Why bagging helps
If $B$ predictors each have variance $\sigma^2$ and pairwise correlation $\rho$, their average has variance $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. Bootstrap resamples decorrelate the models somewhat, so averaging removes part of the variance without changing the bias much. That's why bagging shines with high-variance learners (deep trees) and does little for stable ones (linear models, k-NN with large $k$).

## Stumps are a teaching choice
Stumps are high-bias, so bagging them mostly smooths the threshold location rather than fixing their bias. Boosting is the usual partner for stumps. The mechanics here (bootstrap, fit, vote) are the same for any base learner.

## Out-of-bag estimates
Each bootstrap sample misses about $(1 - 1/n)^n \approx 36.8\%$ of the rows. Scoring each row using only the models that didn't see it gives a free validation estimate.

## Complexity
$B$ times the cost of fitting one stump ($O(d n^2)$ brute force here), plus $O(Bm)$ to vote.

## Common bugs
- Creating a new generator inside the loop, so every estimator gets the same bootstrap.
- Drawing indices with `rng.choice(n, n)`: also valid bagging, but a different random stream from `rng.integers`, so the answers differ.
- Averaging labels instead of voting.

## Interview follow-ups
- Implement the out-of-bag accuracy estimate.
- Why does a random forest also subsample features at each split? (To lower $\rho$ further when a few strong features would otherwise dominate every tree.)
"""
