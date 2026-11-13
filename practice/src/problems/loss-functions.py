import numpy as np

META = dict(title='Robust regression losses and their gradients', tags=['Loss Functions', 'Regression'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:02.04'],
            related=['cross-entropy-logits', 'log-loss-binary-cross-entropy', 'regression-metrics'])

STATEMENT = r"""
Squared error is the default regression loss, but one wild target can dominate it. Write `loss_and_grad(pred, target, kind, delta=1.0, q=0.5)` for five losses, returning the **mean** loss and its gradient with respect to `pred`.

Let $r = \text{pred} - \text{target}$, elementwise, and $n$ the number of elements.

| `kind` | per-element loss $\ell(r)$ |
|---|---|
| `'mse'` | $r^2$ |
| `'mae'` | $\lvert r\rvert$ |
| `'huber'` | $\tfrac12 r^2$ if $\lvert r\rvert \le \delta$, else $\delta(\lvert r\rvert - \tfrac12\delta)$ |
| `'log_cosh'` | $\log\cosh r$ |
| `'quantile'` | $q\,(-r)$ if $r < 0$, else $(1 - q)\,r$ (pinball loss: under-prediction costs $q$ per unit) |

The loss is $L = \frac1n \sum \ell(r)$.

## Output
A tuple `(L, grad)`: `L` a Python `float`, `grad` $= \partial L / \partial\,\text{pred}$, a float array shaped like `pred`.

## Conventions
- `pred`, `target`: float arrays of the same shape (any shape, at least one element). $\delta > 0$, $0 < q < 1$.
- Where a loss has a kink at $r = 0$ (`mae`, `quantile`) use gradient 0 there.
- `log_cosh` must stay finite for $\lvert r\rvert$ up to $10^4$ (`np.cosh(1000)` overflows).
"""

STARTER = '''import numpy as np

def loss_and_grad(pred, target, kind, delta=1.0, q=0.5):
    """Return (mean loss as float, dL/dpred shaped like pred)."""
    r = np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)
    if kind == 'mse':
        return float(np.mean(r ** 2)), 2 * r
    raise NotImplementedError(kind)
'''

SOLUTION = '''import numpy as np

def loss_and_grad(pred, target, kind, delta=1.0, q=0.5):
    r = np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)
    a = np.abs(r)
    if kind == 'mse':
        l, g = r ** 2, 2 * r
    elif kind == 'mae':
        l, g = a, np.sign(r)                                  # sign(0) = 0
    elif kind == 'huber':
        small = a <= delta
        l = np.where(small, 0.5 * r ** 2, delta * (a - 0.5 * delta))
        g = np.where(small, r, delta * np.sign(r))           # = clip(r, -delta, delta)
    elif kind == 'log_cosh':
        l = a + np.log1p(np.exp(-2 * a)) - np.log(2.0)        # log((e^a + e^-a)/2), no overflow
        g = np.tanh(r)
    elif kind == 'quantile':
        l = np.where(r < 0, -q * r, (1 - q) * r)
        g = np.where(r < 0, -q, np.where(r > 0, 1 - q, 0.0))
    else:
        raise ValueError(kind)
    return float(l.mean()), g / r.size                         # the 1/n of the mean
'''

HINTS = [
    'Work with the residual $r$ and the per-element derivative $\\ell\'(r)$; since $\\partial r/\\partial\\,\\text{pred} = 1$, the gradient of the mean is $\\ell\'(r)/n$.',
    'Derivatives: mse $2r$, mae $\\operatorname{sign}(r)$, huber $\\operatorname{clip}(r, -\\delta, \\delta)$, log-cosh $\\tanh r$, quantile $-q$ below zero and $1 - q$ above.',
    'For log-cosh use $\\log\\cosh r = |r| + \\log(1 + e^{-2|r|}) - \\log 2$. Don\'t forget to divide the gradient by `r.size`, not by the length of the first axis.',
]

_r = np.random.default_rng(5)
_p = _r.normal(size=(4, 3))
_t = _p + _r.normal(scale=2, size=(4, 3))
_t[0, 0] = _p[0, 0]                                            # one exact zero residual
TESTS = [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 5.0]), 'mae'),
    (np.array([0.0, 0.0]), np.array([0.5, 3.0]), 'huber', 1.0),
    (np.array([0.0, 0.0, 0.0]), np.array([1.0, -1.0, 0.0]), 'quantile', 1.0, 0.9),
    (_p, _t, 'mse'),
    (_p, _t, 'mae'),
    (_p, _t, 'huber', 0.7),
    (_p, _t, 'log_cosh'),
    (np.array([0.0, 0.0, 0.0]), np.array([1e4, -800.0, 1e-4]), 'log_cosh'),
    (_p, _t, 'quantile', 1.0, 0.2),
    (np.array([[2.0, -2.0]]), np.array([[1.0, -1.0]]), 'huber', 1.0),
]

EXPLANATION = r"""
## What each loss estimates
- **MSE** is minimised by the conditional mean; its gradient grows with the error, so outliers dominate.
- **MAE** is minimised by the median; every point pulls with the same force, which makes it robust but non-smooth at 0.
- **Huber** is quadratic near zero and linear beyond $\delta$: the gradient is MSE's, clipped to $[-\delta, \delta]$. Detection models call it smooth-L1.
- **Log-cosh** behaves like $r^2/2$ near zero and like $|r| - \log 2$ far away, smooth everywhere with gradient $\tanh r$.
- **Quantile (pinball)** loss is minimised by the $q$-quantile, the basis of prediction intervals.

## Stable log-cosh
$\cosh r = \tfrac12 e^{|r|}(1 + e^{-2|r|})$, so $\log\cosh r = |r| + \operatorname{log1p}(e^{-2|r|}) - \log 2$. Every exponent is non-positive.

## Complexity
$O(n)$ time and memory.

## Common bugs
- Forgetting the $1/n$ in the gradient, or dividing by `len(pred)` for a 2-D input.
- Huber with the wrong constant: the linear piece must be $\delta(|r| - \delta/2)$ so value and slope match at $|r| = \delta$.
- Swapping the roles of $q$ and $1-q$ in the pinball loss: with $q = 0.9$, predicting too low must be the expensive direction.

## Interview follow-ups
- Show that the minimiser of the expected pinball loss is the $q$-quantile.
- When would you prefer Huber over MSE when training a value function in RL?
"""
