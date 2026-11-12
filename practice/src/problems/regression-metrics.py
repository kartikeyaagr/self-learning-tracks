import numpy as np

META = dict(title='Core regression metrics', tags=['Metrics & Evaluation', 'Regression'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:10.04'], related=['linear-regression', 'k-fold-cross-validation'])

STATEMENT = r"""
Write `regression_report(y_true, y_pred)` that returns a dict with five metrics, using residuals $e_i = y_i - \hat y_i$:

| key | definition |
| --- | --- |
| `mae` | $\frac1n\sum_i \lvert e_i\rvert$ |
| `mse` | $\frac1n\sum_i e_i^2$ |
| `rmse` | $\sqrt{\text{mse}}$ |
| `r2` | $1 - \dfrac{\sum_i e_i^2}{\sum_i (y_i - \bar y)^2}$ |
| `max_error` | $\max_i \lvert e_i\rvert$ |

## Rules
- `y_true`, `y_pred`: float arrays `(n,)`, $n \ge 1$.
- If `y_true` is constant (the denominator of $R^2$ is 0): `r2` is `1.0` when every prediction is exact, else `0.0`.
- Every value is a Python `float`. The dict has exactly these five keys.
"""

STARTER = '''import numpy as np

def regression_report(y_true, y_pred):
    """Return {'mae', 'mse', 'rmse', 'r2', 'max_error'} as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def regression_report(y_true, y_pred):
    y, yhat = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    e = y - yhat
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0           # R^2 is undefined for a constant target; use the stated convention
    else:
        r2 = 1.0 - ss_res / ss_tot
    mse = ss_res / len(y)
    return {'mae': float(np.mean(np.abs(e))), 'mse': mse, 'rmse': float(np.sqrt(mse)),
            'r2': r2, 'max_error': float(np.max(np.abs(e)))}
'''

HINTS = [
    'Compute the residual vector once and derive everything from it.',
    '$R^2$ compares your squared error with that of always predicting the mean of `y_true`. It can be negative when the model is worse than that baseline.',
    'Handle the constant-target case explicitly before dividing, and wrap every value in `float(...)` so NumPy scalars don\'t leak out.',
]

_r = np.random.default_rng(131)
_y = _r.normal(10, 3, size=40)
TESTS = [
    (np.array([3.0, -0.5, 2.0, 7.0]), np.array([2.5, 0.0, 2.0, 8.0])),
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
    (np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])),
    (np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])),
    (np.array([5.0, 5.0]), np.array([4.0, 6.0])),
    (np.array([42.0]), np.array([40.0])),
    (_y, _y + _r.normal(0, 1, size=40)),
    (_y * 1e6, (_y + 0.5) * 1e6),
]

EXPLANATION = r"""
## What each metric is for
- **MAE** is in the target's units and treats all errors linearly, so it's robust to a few large misses. It's minimised by predicting the conditional **median**.
- **MSE/RMSE** penalise large errors quadratically and are minimised by the conditional **mean**. RMSE is in the target's units, and RMSE $\ge$ MAE always, with a large gap signalling a heavy tail of errors.
- **$R^2$** is the fraction of variance explained relative to predicting $\bar y$. It's unitless, which makes it comparable across targets, but not across datasets with different target variance. It is negative for a model worse than the mean (test 3).
- **Max error** is the worst case, useful when a single large miss is unacceptable.

## The constant-target convention
With constant $y$ the baseline is perfect, so $R^2$'s denominator is 0. Libraries differ: some return `nan`, some 0 or 1 as here. State the convention you use.

## Complexity
$O(n)$ for everything.

## Common bugs
- Computing $R^2$ as the squared Pearson correlation. They agree only for least-squares fits with an intercept evaluated on their own training data.
- Using `np.var(y_true)` with the wrong `ddof` in the denominator. Here it's a sum of squares, so the `n` cancels if you're consistent.
- Returning NumPy scalars; they compare fine but don't serialise like Python floats.

## Interview follow-ups
- When would you prefer MAPE or a log-scale error (RMSLE), and what breaks when $y$ can be 0?
- Why does adding a useless feature never decrease training $R^2$, and how does adjusted $R^2$ correct for it?
"""
