import numpy as np

META = dict(title='Every derivative of the Poisson loss', tags=['Calculus', 'Regression'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:01.06', 'mlmaths:07.01'],
            related=['taylor-approximation-of-loss-landscape', 'gradient-descent-on-non-convex-loss'])

STATEMENT = r"""
Poisson regression models counts $y_n$ with rate $e^{w x_n}$. With a single weight $w$, the negative log-likelihood (dropping the constant $\log y_n!$) is

$$L(w) = \frac{1}{N}\sum_{n=1}^{N} \left(e^{w x_n} - y_n\, w\, x_n\right)$$

Its derivatives follow a neat pattern. Write `poisson_derivatives(w, x, y, order)` that returns a tuple `(derivs, newton_step)`:

- `derivs`: float array of shape `(order + 1,)` with $\left[L(w), L'(w), L''(w), \dots, L^{(\text{order})}(w)\right]$.
- `newton_step`: $-L'(w) / L''(w)$ as a Python `float`, the step a Newton update would take from $w$. If $L''(w) = 0$, return `0.0`.

## Inputs
- `w`: float. `x`, `y`: float arrays of shape `(N,)`, $N \ge 1$, with $y_n \ge 0$ and $|w x_n| \le 50$.
- `order`: integer $\ge 2$.

Vectorise: no loop over samples. A loop over the derivative order is fine.
"""

STARTER = '''import numpy as np

def poisson_derivatives(w, x, y, order):
    """Return ([L, L', ..., L^(order)] at w, Newton step -L'/L'')."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def poisson_derivatives(w, x, y, order):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mu = np.exp(w * x)                         # predicted rate
    derivs = np.empty(order + 1)
    derivs[0] = np.mean(mu - y * w * x)
    derivs[1] = np.mean(x * (mu - y))          # the linear term differentiates once, then vanishes
    for k in range(2, order + 1):
        derivs[k] = np.mean(x ** k * mu)       # d^k/dw^k e^{wx} = x^k e^{wx}
    step = -derivs[1] / derivs[2] if derivs[2] != 0 else 0.0
    return derivs, float(step)
'''

HINTS = [
    'Differentiate one term $e^{wx} - y w x$ with respect to $w$ a few times. Each derivative of $e^{wx}$ pulls down one more factor of $x$.',
    '$L\'(w) = \\frac1N\\sum_n x_n(e^{wx_n} - y_n)$, and for $k \\ge 2$ the $y$ term is gone: $L^{(k)}(w) = \\frac1N\\sum_n x_n^k e^{w x_n}$.',
    'Compute `mu = np.exp(w * x)` once, fill entries 0 and 1 by hand, then loop `k` from 2 to `order` with `np.mean(x**k * mu)`.',
]

_r = np.random.default_rng(17)
_x = _r.normal(size=30)
_y = _r.poisson(np.exp(0.7 * _x)).astype(float)
TESTS = [
    (0.0, np.array([1.0, 2.0]), np.array([1.0, 3.0]), 2),
    (0.5, np.array([1.0, -1.0, 2.0]), np.array([2.0, 0.0, 3.0]), 4),
    (0.0, np.array([0.0, 0.0]), np.array([1.0, 2.0]), 3),
    (-2.0, np.array([10.0, 20.0]), np.array([0.0, 0.0]), 3),
    (1.0, np.array([3.0]), np.array([20.0]), 6),
    (0.0, _x, _y, 5),
    (0.7, _x, _y, 3),
]

EXPLANATION = r"""
## The pattern
$\frac{d}{dw} e^{wx} = x e^{wx}$, so each derivative multiplies by $x$. The term $-y w x$ is linear in $w$: it contributes $-yx$ to $L'$ and nothing after. So $L^{(k)}(w) = \frac1N\sum_n x_n^k \mu_n$ for $k \ge 2$, with $\mu_n = e^{w x_n}$.

## Curvature and convexity
$L''(w) = \frac1N\sum_n x_n^2 \mu_n \ge 0$, so the loss is convex in $w$, and strictly convex unless every $x_n = 0$. The Newton step $-L'/L''$ uses that curvature as a local step size: flat regions get big steps, sharp ones small steps. Odd derivatives like $L'''$ measure how lopsided the bowl is; when $|L'''|$ is large relative to $L''$, a Newton step overshoots or undershoots.

## Complexity
$O(N \cdot \text{order})$, with one `exp` per sample.

## Common bugs
- Keeping the $-y x$ term in every derivative.
- Recomputing `np.exp` inside the loop (correct, just wasteful).
- Dividing by $L''$ without checking for 0 when all inputs are 0.

## Interview follow-ups
- Run Newton's method from $w = 0$ on test 6. How many steps until $|L'| < 10^{-10}$? Compare with gradient descent.
- With a vector $w$, the Hessian is $\frac1N X^\top \mathrm{diag}(\mu) X$. What's the third-derivative object, and why do second-order optimisers stop at the Hessian?
"""
