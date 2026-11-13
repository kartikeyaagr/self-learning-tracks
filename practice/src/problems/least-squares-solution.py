import numpy as np

META = dict(title='Least squares with an intercept and ridge', tags=['Regression', 'Linear Algebra'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:08.02', 'mlmaths:08.01'], related=['orthogonal-projection-matrix', 'moore-penrose-pseudoinverse', 'qr-decomposition'])

STATEMENT = r"""
Fit a linear model $\hat y = Xw + b$ by minimising squared error plus an L2 penalty on the **weights only** (the intercept is never penalised):

$$\min_{w, b}\; \lVert Xw + b\mathbf{1} - y\rVert_2^2 + \lambda\lVert w\rVert_2^2$$

Write `fit_linear(X, y, lam=0.0)` that returns the exact minimiser.

## Inputs
- `X`: float array `(n, d)`, $n \ge 2$, $d \ge 1$.
- `y`: float array `(n,)`.
- `lam`: float $\lambda \ge 0$. When `lam == 0`, the tests guarantee the centred columns of `X` are linearly independent, so the answer is unique.

## Output
A tuple `(w, b, r2)`:
- `w`: float array `(d,)`; `b`: Python float.
- `r2`: the coefficient of determination on the training data, $R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$, as a Python float. If `y` is constant, define `r2 = 1.0` when the residuals are all (numerically) zero, i.e. $\sum_i (y_i - \hat y_i)^2 \le 10^{-20}$, else `0.0`.

## Requirements
Don't use `np.linalg.inv`. Don't append a column of ones and penalise it.
"""

STARTER = '''import numpy as np

def fit_linear(X, y, lam=0.0):
    """X: (n, d), y: (n,). Minimise ||Xw + b - y||^2 + lam ||w||^2 (b unpenalised). Return (w, b, r2)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def fit_linear(X, y, lam=0.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    # The optimal intercept makes the residuals sum to zero, so centring removes b from the problem.
    if lam == 0:
        w = np.linalg.lstsq(Xc, yc, rcond=None)[0]        # orthogonal-factorisation route; no normal equations
    else:
        d = X.shape[1]
        w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)
    b = float(y_mean - x_mean @ w)
    sse = float(np.sum((X @ w + b - y) ** 2))
    sst = float(np.sum(yc ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else (1.0 if sse <= 1e-20 else 0.0)
    return w, b, r2
'''

HINTS = [
    'Set the derivative with respect to $b$ to zero. What does it tell you about $b$ once $w$ is known?',
    '$b = \\bar y - \\bar x^\\top w$. Substituting it back leaves a problem in centred variables $X_c = X - \\bar x$, $y_c = y - \\bar y$ with no intercept: $(X_c^\\top X_c + \\lambda I)w = X_c^\\top y_c$.',
    'Solve that system with `np.linalg.solve` (or `np.linalg.lstsq(Xc, yc)` when `lam == 0`), then `b = y_mean - x_mean @ w`. Compute $R^2$ from the fitted values, guarding a constant `y`.',
]

_r = np.random.default_rng(89)
_X = _r.normal(size=(40, 3))
_y = _X @ np.array([2.0, -1.0, 0.5]) + 4.0 + 0.3 * _r.normal(size=40)
_Xs = _r.normal(size=(25, 4)) * np.array([1.0, 100.0, 0.01, 1.0]) + 1000.0
TESTS = [
    (np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([1.0, 3.0, 5.0, 7.0])),
    (np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 2.0, 5.0]), 1.0),
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]), np.array([3.0, 3.0, 3.0, 3.0])),
    (np.array([[1.0], [2.0]]), np.array([5.0, 3.0])),
    (_X, _y),
    (_X, _y, 10.0),
    (_Xs, _r.normal(size=25)),
    (np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]), np.array([1.0, 2.0, 1.0, 3.0]), 0.5),
]

EXPLANATION = r"""
## Why centring handles the intercept
The gradient with respect to $b$ is $2\sum_i(x_i^\top w + b - y_i)$, so at the optimum $b = \bar y - \bar x^\top w$. Plugging that in turns every residual into $(x_i - \bar x)^\top w - (y_i - \bar y)$: the same objective on centred data with no intercept. Its normal equations are $(X_c^\top X_c + \lambda I)w = X_c^\top y_c$, which is symmetric positive definite whenever $\lambda > 0$. That's why ridge also fixes test 8, where the two columns are perfectly collinear and plain least squares has infinitely many solutions.

## Why not a column of ones
Appending ones and solving $(A^\top A + \lambda I)\theta = A^\top y$ penalises the intercept too. Then shifting $y$ by a constant changes the slopes, and the model shrinks predictions towards 0 instead of towards $\bar y$. Test 6 would come out wrong.

## Numerical notes
For `lam == 0`, `lstsq` (SVD-based) works on $X_c$ directly instead of squaring its condition number in $X_c^\top X_c$. Test 7 has features on wildly different scales around an offset of 1000. Solving the normal equations of the *uncentred* data with a ones column gets its weights wrong in the first digit; centring first (or `lstsq` on the uncentred design) keeps it accurate.

## Complexity
$O(nd^2 + d^3)$.

## Common bugs
- Penalising the intercept.
- Recovering $b$ as `y.mean()` alone, forgetting the $\bar x^\top w$ term.
- Dividing by zero in $R^2$ when `y` is constant (test 3).

## Interview follow-ups
- How does the ridge solution behave as $\lambda \to \infty$ and $\lambda \to 0$? ($w \to 0$ and $b \to \bar y$; as $\lambda \to 0$ it tends to the minimum-norm least-squares solution.)
- Why standardise features before ridge? (The penalty treats all weights alike, so a feature measured in millimetres is penalised differently from the same feature in metres.)
"""
