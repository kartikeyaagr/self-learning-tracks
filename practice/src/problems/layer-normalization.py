import numpy as np

META = dict(title='LayerNorm backward pass', tags=['Neural Networks', 'Transformers', 'Calculus'], diff='Medium', runtime='numpy',
            lessons=['ieng:02.03', 'air-fanout-v1:05.04'],
            related=['layer-norm', 'batch-normalization', 'transformer-encoder-block'])

STATEMENT = r"""
LayerNorm normalises every row (token) over its $d$ features:

$$\mu = \tfrac1d \textstyle\sum_j x_j, \quad \sigma^2 = \tfrac1d \sum_j (x_j - \mu)^2, \quad \hat x = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \odot \hat x + \beta.$$

The forward pass is short. The backward pass is the interview question, because $\mu$ and $\sigma$ depend on every element of the row.

Write `layer_norm_backward(x, gamma, beta, dy, eps=1e-5)` that returns `(y, dx, dgamma, dbeta)`.

## Shapes
- `x`, `dy`: float arrays `(..., d)` (any number of leading axes); normalise over the last axis only, with the population variance.
- `gamma`, `beta`: `(d,)`.

## Output
- `y` and `dx` $= \partial L/\partial x$: shaped like `x`, where `dy` $= \partial L/\partial y$ is the upstream gradient.
- `dgamma`, `dbeta`: `(d,)`, summed over all leading axes.
- Use the closed-form gradient (no finite differences, no loops over rows).
"""

STARTER = '''import numpy as np

def layer_norm_backward(x, gamma, beta, dy, eps=1e-5):
    """Return (y, dx, dgamma, dbeta)."""
    x, dy = np.asarray(x, dtype=float), np.asarray(dy, dtype=float)
    mu = x.mean(axis=-1, keepdims=True)
    rstd = 1 / np.sqrt(x.var(axis=-1, keepdims=True) + eps)
    x_hat = (x - mu) * rstd
    y = gamma * x_hat + beta
    dx = dy * gamma * rstd                      # treats mu and sigma as constants: wrong
    lead = tuple(range(x.ndim - 1))
    return y, dx, (dy * x_hat).sum(axis=lead), dy.sum(axis=lead)
'''

SOLUTION = '''import numpy as np

def layer_norm_backward(x, gamma, beta, dy, eps=1e-5):
    x, dy = np.asarray(x, dtype=float), np.asarray(dy, dtype=float)
    d = x.shape[-1]
    mu = x.mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(x.var(axis=-1, keepdims=True) + eps)
    x_hat = (x - mu) * rstd
    y = gamma * x_hat + beta
    lead = tuple(range(x.ndim - 1))
    dgamma, dbeta = (dy * x_hat).sum(axis=lead), dy.sum(axis=lead)
    g = dy * gamma                                         # dL/dx_hat
    # remove the components of g along the two directions the normalisation fixes: the mean and x_hat itself
    dx = rstd * (g - g.mean(axis=-1, keepdims=True) - x_hat * (g * x_hat).mean(axis=-1, keepdims=True))
    return y, dx, dgamma, dbeta
'''

HINTS = [
    '`dgamma` and `dbeta` are easy: $y$ is linear in them, so sum `dy * x_hat` and `dy` over every axis but the last. Let $g = dy \\odot \\gamma$ be the gradient with respect to $\\hat x$.',
    'The Jacobian of $\\hat x$ with respect to $x$ (per row) is $\\frac{r}{d}\\left(d I - \\mathbf 1\\mathbf 1^\\top - \\hat x \\hat x^\\top\\right)$ with $r = 1/\\sqrt{\\sigma^2 + \\epsilon}$. It is symmetric, so apply it to $g$ directly.',
    'Result: $dx = r\\,\\big(g - \\operatorname{mean}(g) - \\hat x \\cdot \\operatorname{mean}(g \\odot \\hat x)\\big)$, with both means over the last axis and `keepdims=True`.',
]

_r = np.random.default_rng(79)
TESTS = [
    (np.array([[1.0, 2.0, 3.0]]), np.ones(3), np.zeros(3), np.array([[1.0, 0.0, 0.0]])),
    (np.array([[0.0, 2.0]]), np.array([2.0, 1.0]), np.array([0.5, 0.0]), np.array([[1.0, 1.0]])),
    (_r.normal(size=(4, 6)), _r.normal(size=6), _r.normal(size=6), _r.normal(size=(4, 6))),
    (_r.normal(size=(2, 3, 5)) * 10 + 3, _r.normal(size=5), _r.normal(size=5), _r.normal(size=(2, 3, 5))),
    (np.full((2, 4), 3.0), np.ones(4), np.zeros(4), _r.normal(size=(2, 4))),
    (_r.normal(size=(3, 8)), np.ones(8), np.zeros(8), np.ones((3, 8))),
    (_r.normal(size=(1, 2, 2, 7)), _r.normal(size=7), np.zeros(7), _r.normal(size=(1, 2, 2, 7)), 1e-2),
]

EXPLANATION = r"""
## Deriving it
Write $r = (\sigma^2 + \epsilon)^{-1/2}$. Then $\partial \hat x_i/\partial x_j = r\left(\delta_{ij} - \tfrac1d - \tfrac1d \hat x_i \hat x_j\right)$: the $\tfrac1d$ comes from $\mu$, the $\hat x_i \hat x_j$ term from $\sigma$ (using $\partial \sigma^2/\partial x_j = \tfrac2d (x_j - \mu)$). Multiplying by $g$ gives
$$dx = r\left(g - \bar g - \hat x\,\overline{g \hat x}\right).$$
Geometrically, $\hat x$ always has mean 0 and mean square $\approx 1$, so the gradient has no component along $\mathbf 1$ or along $\hat x$: the two directions the normalisation erases.

## Sanity checks
- Each row of `dx` sums to 0 (shifting a row by a constant doesn't change $y$).
- `dy = 1` everywhere with $\gamma = 1$ gives `dx = 0` (test 6): $\sum_j y_j$ is constant.
- A constant row (test 5) has $\hat x = 0$ and $dx = g - \bar g$ scaled by $1/\sqrt\epsilon$; large, which is why $\epsilon$ matters.

## Complexity
$O(n d)$ time with three row reductions; no $d \times d$ Jacobian is ever formed.

## Common bugs
- Treating $\mu$ and $\sigma$ as constants (the starter does this): it passes only when the row-mean terms happen to vanish.
- Reducing `dgamma` over the last axis instead of the leading ones.
- Using $\sqrt{\sigma^2} + \epsilon$ instead of $\sqrt{\sigma^2 + \epsilon}$.

## Interview follow-ups
- Do the same for RMSNorm (no mean subtraction): which term disappears?
- Why do fused LayerNorm kernels recompute $\hat x$ in the backward pass instead of storing it?
"""
