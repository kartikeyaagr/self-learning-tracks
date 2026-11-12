import numpy as np

META = dict(title='Stacking with out-of-fold predictions', tags=['Classic ML', 'Regression'], diff='Hard', runtime='numpy',
            lessons=[], related=['k-fold-cross-validation', 'ridge-regression', 'knn-classifier'])

STATEMENT = r"""
Stacking trains a **meta-model** on the predictions of several base models. The meta-model must be trained on predictions the base models made for rows they did **not** see, or it will learn to trust whichever base model overfits most.

Write `stack(X_train, y_train, X_test, alpha, k, n_folds)` with two base regressors and a linear meta-model.

## Base models
1. **Ridge** with unpenalised intercept: minimise $\sum_i (y_i - x_i^\top w - b)^2 + \alpha\lVert w\rVert^2$ exactly.
2. **k-NN regressor**: the mean target of the $k$ nearest training rows by Euclidean distance. Distance ties go to the row with the **lower original index** in `X_train`.

## Procedure
1. Folds: `np.array_split(np.arange(n), n_folds)` (contiguous, no shuffling).
2. For each fold, fit both base models on the other folds and predict the held-out fold. This fills an out-of-fold matrix $Z$ of shape `(n, 2)`: column 0 ridge, column 1 k-NN.
3. Meta-model: ordinary least squares **with intercept** of `y_train` on $Z$, giving coefficients $\beta \in \mathbb R^2$ and intercept $\beta_0$.
4. Refit both base models on the **full** training set, predict `X_test` to get $Z_\text{test}$ `(m, 2)`, and return $Z_\text{test}\beta + \beta_0$.

## Input / output
- `X_train`: float `(n, d)`; `y_train`: float `(n,)`; `X_test`: float `(m, d)`; `alpha > 0`; `n_folds >= 2`; `k` no larger than the smallest training split.
- The tests guarantee the meta design $[Z \;\; \mathbf 1]$ has full column rank.
- Return `(predictions (m,), beta (2,), beta0 float)`.
"""

STARTER = '''import numpy as np

def stack(X_train, y_train, X_test, alpha, k, n_folds):
    """Ridge + kNN base models, OLS meta-model on out-of-fold predictions. Return (preds, beta, beta0)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ridge_fit(X, y, alpha):
    mx, my = X.mean(axis=0), y.mean()
    Xc = X - mx
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - my))
    return w, my - mx @ w

def knn_predict(X, y, Q, k):
    d2 = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    nn = np.argsort(d2, axis=1, kind='stable')[:, :k]          # rows of X are in original order
    return y[nn].mean(axis=1)

def base_preds(X, y, Q, alpha, k):
    w, b = ridge_fit(X, y, alpha)
    return np.column_stack([Q @ w + b, knn_predict(X, y, Q, k)])

def stack(X_train, y_train, X_test, alpha, k, n_folds):
    X, y, Xt = np.asarray(X_train, float), np.asarray(y_train, float), np.asarray(X_test, float)
    n = len(y)
    Z = np.zeros((n, 2))
    for val in np.array_split(np.arange(n), n_folds):
        tr = np.setdiff1d(np.arange(n), val)                     # sorted, so kNN ties still favour lower index
        Z[val] = base_preds(X[tr], y[tr], X[val], alpha, k)
    A = np.column_stack([Z, np.ones(n)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    Zt = base_preds(X, y, Xt, alpha, k)                          # base models refit on all training rows
    return Zt @ coef[:2] + coef[2], coef[:2], float(coef[2])
'''

HINTS = [
    'Write three helpers: ridge fit (centre, solve, recover the intercept), k-NN regression (stable argsort of squared distances, mean of the first k targets), and one that returns both models\' predictions as an `(m, 2)` matrix.',
    'Out-of-fold: for each `val` in `np.array_split(np.arange(n), n_folds)`, train on `np.setdiff1d(np.arange(n), val)` and write the predictions into `Z[val]`. Every row of `Z` comes from models that never saw that row.',
    'The meta-model is `np.linalg.lstsq(np.column_stack([Z, ones]), y)`. For the test set, refit both base models on all of `X_train` before predicting; don\'t average the fold models.',
]

_r = np.random.default_rng(73)
_X = _r.uniform(-3, 3, size=(36, 2))
_y = 1.5 * _X[:, 0] + np.sin(2 * _X[:, 1]) * 2 + 0.2 * _r.normal(size=36)
_Xt = _r.uniform(-3, 3, size=(6, 2))
TESTS = [
    (np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]), np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0]), np.array([[2.5], [6.0]]), 1.0, 1, 3),
    (np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [4.0, 3.0], [5.0, 5.0], [6.0, 4.0], [7.0, 6.0]]),
     np.array([1.0, 0.5, 3.0, 2.0, 5.5, 8.0, 7.0, 10.0]), np.array([[3.5, 3.0]]), 0.5, 2, 2),
    (np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]), np.array([3.0, -1.0, 4.0, 1.0, 5.0, -9.0, 2.0, 6.0]),
     np.array([[0.5], [3.5], [7.5]]), 2.0, 2, 4),
    (np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]]), np.array([1.0, 3.0, 2.0, 6.0, 5.0, 9.0]), np.array([[3.0], [5.0]]), 0.1, 2, 2),
    (_X, _y, _Xt, 1.0, 3, 4),
    (_X, _y, _Xt, 10.0, 5, 3),
    (_X * 100.0, _y * 100.0, _Xt * 100.0, 1e4, 3, 6),
]

EXPLANATION = r"""
## Why out-of-fold predictions
If the meta-model saw in-sample base predictions, a 1-NN base model would look perfect (it reproduces every training target), and the meta-model would put all its weight on it. Out-of-fold predictions are honest estimates of how each base model behaves on unseen data, which is exactly what the meta-model will face at test time.

## Why refit on the full data
Each fold model saw only $(K-1)/K$ of the data. Refitting on everything gives slightly better base models; the meta-weights learned from out-of-fold predictions still transfer because the refit models behave like the fold models, only a bit stronger. Averaging the $K$ fold models instead is the other common choice.

## Complexity
$K + 1$ fits of each base model plus one small least-squares problem. For k-NN, each fold costs $O(n_\text{val}\, n_\text{tr}\, d)$.

## Common bugs
- Training base models on all rows before building $Z$ (leakage).
- Shuffling folds, or using different folds for the two base models, which breaks the comparison.
- Forgetting the meta intercept, which forces the stack through the origin.

## Interview follow-ups
- When does stacking beat simply averaging the base models? (When base models are diverse and differently accurate in different regions.)
- How would you constrain the meta-model to avoid overfitting with many base models? (Non-negative least squares, ridge on the meta-model, or a simplex constraint.)
"""
