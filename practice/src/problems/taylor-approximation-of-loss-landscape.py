import numpy as np

META = dict(title='First- and second-order Taylor models of a logistic loss', tags=['Calculus', 'Optimization'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:09.28', 'mlmaths:07.01'],
            related=['higher-order-derivatives-and-loss-curvature', 'hessian-tensor-of-vector-valued-function',
                     'hessian-matrix-and-convexity-of-linear-regressio'])

STATEMENT = r"""
Optimisers reason about the loss through local models. Around a point $w_0$,

$$L(w_0 + \delta) \approx \underbrace{L_0 + g^\top \delta}_{\text{first order}} \qquad\text{and}\qquad L(w_0 + \delta) \approx \underbrace{L_0 + g^\top\delta + \tfrac12\,\delta^\top H \delta}_{\text{second order}}$$

where $L_0 = L(w_0)$, $g = \nabla L(w_0)$ and $H = \nabla^2 L(w_0)$.

For logistic regression with labels $y_n \in \{-1, +1\}$, the mean loss is

$$L(w) = \frac1n \sum_{n} \log\bigl(1 + e^{-y_n x_n^\top w}\bigr)$$

Write `taylor_models(X, y, w0, deltas)` that evaluates the true loss and both Taylor models at $w_0 + \delta_k$ for every row $\delta_k$ of `deltas`, and returns the tuple `(true, first, second)` of float arrays, each of shape `(K,)`.

## Requirements
- `X`: `(n, d)`, `y`: `(n,)` of $\pm 1$ floats, `w0`: `(d,)`, `deltas`: `(K, d)`.
- Derive $g$ and $H$ by hand. Compute $\log(1 + e^{z})$ stably with `np.logaddexp(0, z)`.
- Vectorise over the $K$ perturbations (no loop over rows of `deltas`).
"""

STARTER = '''import numpy as np

def taylor_models(X, y, w0, deltas):
    """Return (true loss, first-order model, second-order model) at w0 + each row of deltas."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def taylor_models(X, y, w0, deltas):
    X, y, w0, D = (np.asarray(a, dtype=float) for a in (X, y, w0, deltas))
    n = len(y)
    loss = lambda W: np.mean(np.logaddexp(0, -y[:, None] * (X @ W)), axis=0)   # W: (d, K) -> (K,)
    m = y * (X @ w0)                                   # margins at w0
    p = 1 / (1 + np.exp(m))                            # sigma(-m): how wrong each sample is
    g = -X.T @ (y * p) / n
    H = (X * (p * (1 - p))[:, None]).T @ X / n         # X^T diag(sigma(m) sigma(-m)) X / n
    L0 = loss(w0[:, None])[0]
    first = L0 + D @ g
    second = first + 0.5 * np.einsum('kd,de,ke->k', D, H, D)
    return loss((w0 + D).T), first, second
'''

HINTS = [
    'Write the loss per sample as $\\ell(m) = \\log(1+e^{-m})$ with margin $m = y\\,x^\\top w$. Then $\\ell\'(m) = -\\sigma(-m)$ and $\\ell\'\'(m) = \\sigma(m)\\sigma(-m)$.',
    'Chain rule through $m = y\\,x^\\top w$: $g = -\\frac1n\\sum_n y_n\\sigma(-m_n)\\,x_n$ and $H = \\frac1n\\sum_n \\sigma(m_n)\\sigma(-m_n)\\,x_nx_n^\\top$ (the $y_n^2 = 1$ drops out).',
    'Vectorise over perturbations: `first = L0 + deltas @ g`, the quadratic term is `np.einsum("kd,de,ke->k", deltas, H, deltas)`, and the true loss can be evaluated for all $K$ points at once from `X @ (w0 + deltas).T`.',
]

_r = np.random.default_rng(53)
_X = _r.normal(size=(40, 3))
_y = np.where(_X @ np.array([1.0, -2.0, 0.5]) + 0.5 * _r.normal(size=40) > 0, 1.0, -1.0)
_w0 = np.array([0.5, -1.0, 0.2])
_dirs = _r.normal(size=(1, 3))
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, -1.0]), np.zeros(2), np.array([[0.1, 0.0], [0.0, 0.1]])),
    (np.array([[1.0, 2.0]]), np.array([1.0]), np.array([0.0, 0.0]), np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])),
    (_X, _y, _w0, np.array([[1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0]])),
    (_X, _y, _w0, _dirs * np.array([[0.01], [0.1], [1.0], [10.0]])),
    (np.array([[50.0, 0.0], [0.0, 50.0]]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([[-0.02, 0.0], [-0.04, -0.04]])),
    (_r.normal(size=(15, 5)), np.where(_r.random(15) < 0.5, 1.0, -1.0), _r.normal(size=5) * 0.3, _r.normal(size=(6, 5)) * 0.5),
]

EXPLANATION = r"""
## Deriving g and H
With margins $m_n = y_n x_n^\top w$, the loss is a mean of $\ell(m_n)$ where $\ell(m) = \log(1 + e^{-m})$. Since $\ell'(m) = -\sigma(-m)$ and $\ell''(m) = \sigma(m)\sigma(-m)$, and $\partial m_n/\partial w = y_n x_n$:

$$g = -\frac1n X^\top\bigl(y \odot \sigma(-m)\bigr), \qquad H = \frac1n X^\top \mathrm{diag}\bigl(\sigma(m)\sigma(-m)\bigr) X$$

$H$ is positive semidefinite, so the second-order model is a convex bowl.

## How good are the models?
The first-order error is $O(\lVert\delta\rVert^2)$ and the second-order error is $O(\lVert\delta\rVert^3)$. Test 4 scales one direction from 0.01 to 10: at 0.01 both models are nearly exact, at 1 the quadratic is still close while the linear model is off, and at 10 both fail, the quadratic badly because the real loss grows only linearly in the tails while the model grows quadratically. Trust-region methods use exactly this comparison (predicted vs actual decrease) to decide how far to trust the model.

## Complexity
$O(nd^2)$ for $H$, $O(K d^2)$ for the quadratic terms and $O(nKd)$ for the true losses.

## Common bugs
- A sign error in $g$ (forgetting $\ell' < 0$), which makes the linear model predict ascent where there's descent.
- Evaluating the true loss as `np.log(1 + np.exp(-m))`, which overflows to `inf` once a margin drops below about $-710$.
- Building $\mathrm{diag}(\cdot)$ as an $n\times n$ matrix. Scale the rows of $X$ instead.

## Interview follow-ups
- Minimise the second-order model in closed form. What update do you get, and what goes wrong on separable data where $H \to 0$?
- Why does a positive definite $H$ make the second-order model's minimiser a descent direction for the true loss?
"""
