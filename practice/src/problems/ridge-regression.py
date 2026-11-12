import numpy as np

META = dict(title='Ridge regression with an unpenalised intercept', tags=['Regression', 'Optimization'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:08.02', 'air-fanout-v1:02.04'], related=['linear-regression', 'lasso-regression'])

STATEMENT = r"""
Ridge regression adds an L2 penalty on the weights, but **not** on the intercept:

$$\min_{w,\,b}\ \sum_{i=1}^n \left(y_i - x_i^\top w - b\right)^2 + \alpha \lVert w\rVert_2^2 .$$

Write `fit_ridge(X, y, alpha)` that returns the exact minimiser `(w, b)`.

## Input
- `X`: float array `(n, d)`; `y`: float array `(n,)`; `alpha >= 0`.
- When `alpha == 0` the tests guarantee the centred design has full column rank. When `alpha > 0`, `d` may exceed `n` and columns may be collinear.

## Output
- `w`: float array `(d,)`; `b`: Python `float`.

## Notes
- The penalty is on the **sum** of squared errors (not the mean). Get the scaling right or your weights will differ.
- Do not penalise `b`: shifting every target by a constant must shift `b` by that constant and leave `w` unchanged.
"""

STARTER = '''import numpy as np

def fit_ridge(X, y, alpha):
    """Minimise ||y - Xw - b||^2 + alpha ||w||^2 (b unpenalised). Return (w (d,), b float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def fit_ridge(X, y, alpha):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc = X - x_mean                                   # centring removes b from the problem
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xc.T @ (y - y_mean))
    return w, float(y_mean - x_mean @ w)              # b puts the fitted plane through the means
'''

HINTS = [
    'Take the derivative with respect to `b` first. What does it tell you about where the fitted plane passes?',
    'Setting $\\partial/\\partial b = 0$ gives $b = \\bar y - \\bar x^\\top w$. Substituting back leaves ridge on centred data with no intercept.',
    'With $X_c = X - \\bar x$: `w = solve(Xc.T @ Xc + alpha * I, Xc.T @ (y - y.mean()))`, then `b = y.mean() - X.mean(0) @ w`.',
]

_r = np.random.default_rng(8)
_X = _r.normal(size=(20, 4))
_Xw = _r.normal(size=(5, 8))
TESTS = [
    (np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([1.0, 3.0, 5.0, 7.0]), 0.0),
    (np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([1.0, 3.0, 5.0, 7.0]), 5.0),
    (np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), np.array([1.0, 2.0, 2.0]), 1.0),
    (np.array([[1.0], [2.0], [3.0]]), np.array([10.0, 20.0, 30.0]), 1e9),
    (_X, _X @ np.array([1.0, 0.0, -2.0, 3.0]) + 0.5 * _r.normal(size=20) + 7.0, 0.0),
    (_X, _X @ np.array([1.0, 0.0, -2.0, 3.0]) + 0.5 * _r.normal(size=20) + 7.0, 10.0),
    (_Xw, _r.normal(size=5), 0.5),
    (np.array([[-100.0, 3.0], [50.0, -1.0], [20.0, 0.0], [-30.0, 2.0]]), np.array([-5.0, 2.0, 1.0, -1.0]), 100.0),
]

EXPLANATION = r"""
## Why centring handles the intercept
$\partial/\partial b = 0$ gives $b = \bar y - \bar x^\top w$ for any $w$. Plugging it in turns the residual into $(y - \bar y) - (X - \bar x)w$, so the remaining problem is ridge with no intercept on centred data. Its gradient is $-2X_c^\top(y_c - X_c w) + 2\alpha w$; setting it to zero gives $(X_c^\top X_c + \alpha I)\,w = X_c^\top y_c$.

## Why ridge always has a solution
$X_c^\top X_c$ is positive semidefinite, so adding $\alpha I$ with $\alpha > 0$ makes every eigenvalue at least $\alpha$. The matrix is invertible even with collinear columns (test 3) or $d > n$ (test 7). In SVD terms, each component is shrunk by $s_j^2/(s_j^2 + \alpha)$.

## Complexity
$O(nd^2 + d^3)$. When $d \gg n$, use the dual form $w = X_c^\top (X_cX_c^\top + \alpha I)^{-1} y_c$, which costs $O(n^2 d + n^3)$.

## Common bugs
- Appending a ones column and penalising it too. Then shifting `y` changes `w`, and the intercept shrinks toward 0.
- Mixing up mean and sum scaling: $\frac1n\lVert\cdot\rVert^2 + \alpha\lVert w\rVert^2$ is the same as using $n\alpha$ here.
- Using `np.linalg.inv` instead of `solve`.

## Interview follow-ups
- Show that ridge is the MAP estimate under a Gaussian prior $w \sim \mathcal N(0, \tau^2 I)$. What is $\alpha$ in terms of the noise variance and $\tau^2$?
- Why should features be standardised before ridge? (The penalty treats all coefficients equally, so a feature measured in millimetres is penalised differently from the same feature in metres.)
"""
