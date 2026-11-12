import numpy as np

META = dict(title='Minimum-norm subgradient of the lasso', tags=['Calculus', 'Optimization', 'Regression'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:09.08', 'air-fanout-v1:02.04'],
            related=['continuity-of-activation-functions', 'gradient-of-multivariate-linear-regression-loss',
                     'lagrange-multipliers-regularization-as-constrain'])

STATEMENT = r"""
The lasso objective has a kink wherever a weight is zero:

$$f(w) = \frac{1}{2n}\lVert Xw - y\rVert_2^2 + \lambda \lVert w \rVert_1$$

At a kink there is no gradient, only a **subdifferential**: the set of slopes of all lines that touch $f$ from below. For $|w_j|$ that set is $\{\operatorname{sign}(w_j)\}$ when $w_j \ne 0$ and the interval $[-1, 1]$ when $w_j = 0$. So with $g = \frac1n X^\top(Xw - y)$, the gradient of the smooth part, every subgradient has coordinates

$$v_j = g_j + \lambda s_j, \qquad s_j = \operatorname{sign}(w_j) \text{ if } w_j \ne 0, \quad s_j \in [-1, 1] \text{ if } w_j = 0$$

The coordinates are independent, so the **minimum-norm** subgradient picks the best $s_j$ separately in each one. A point minimises $f$ exactly when that minimum-norm subgradient is the zero vector.

Write `lasso_min_subgrad(X, y, w, lam)` that returns a tuple `(v, optimal)`:
- `v`: the minimum-norm subgradient, float array of shape `(d,)`.
- `optimal`: Python `bool`, `True` when $\max_j |v_j| \le 10^{-10}$.

## Inputs
`X`: `(n, d)` floats, `y`: `(n,)`, `w`: `(d,)`, `lam` $\ge 0$. A coordinate counts as a kink only when `w[j] == 0.0` exactly.
"""

STARTER = '''import numpy as np

def lasso_min_subgrad(X, y, w, lam):
    """Return (minimum-norm subgradient of the lasso objective at w, is_optimal)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def lasso_min_subgrad(X, y, w, lam):
    X, y, w = (np.asarray(a, dtype=float) for a in (X, y, w))
    g = X.T @ (X @ w - y) / len(y)                       # gradient of the smooth part
    at_kink = w == 0
    # off the kink the subgradient is unique; on it, choose s in [-1, 1] to pull g towards 0
    shrunk = np.sign(g) * np.maximum(np.abs(g) - lam, 0.0)
    v = np.where(at_kink, shrunk, g + lam * np.sign(w))
    return v, bool(np.max(np.abs(v), initial=0.0) <= 1e-10)
'''

HINTS = [
    'Split the coordinates into two groups: $w_j \\ne 0$, where the subgradient is just $g_j + \\lambda\\,\\mathrm{sign}(w_j)$, and $w_j = 0$, where you get to choose.',
    'At a kink, $v_j$ ranges over the interval $[g_j - \\lambda,\\ g_j + \\lambda]$. Pick the point of that interval closest to 0.',
    'The closest point is the soft-threshold: `np.sign(g) * np.maximum(np.abs(g) - lam, 0)`. Combine both groups with `np.where(w == 0, ...)`.',
]

_r = np.random.default_rng(4)
_X = _r.normal(size=(12, 5))
_y = _r.normal(size=12)
TESTS = [
    (np.eye(2) * 2, np.array([4.0, 1.0]), np.array([0.0, 0.0]), 0.5),
    (np.eye(2) * 2, np.array([4.0, 1.0]), np.array([1.0, 0.0]), 0.5),
    (np.eye(4) * 2, np.array([4.0, -1.0, 0.2, -6.0]), np.array([1.5, 0.0, 0.0, -2.5]), 0.5),
    (_X, _y, np.zeros(5), 10.0),
    (_X, _y, np.zeros(5), 0.0),
    (_X, _y, np.array([0.3, 0.0, -0.2, 0.0, 0.1]), 0.05),
    (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([1.0, -1.0, 2.0]), np.array([0.0, -1.0]), 1.0),
    (_r.normal(size=(20, 6)), _r.normal(size=20), _r.normal(size=6) * (_r.random(6) < 0.5), 0.2),
]

EXPLANATION = r"""
## Why the soft-threshold
At a kink, $v_j = g_j + \lambda s_j$ with $s_j \in [-1, 1]$ sweeps the interval $[g_j - \lambda, g_j + \lambda]$. If that interval contains 0 (i.e. $|g_j| \le \lambda$), the best choice is $v_j = 0$; otherwise it's the endpoint nearer 0, which is $g_j - \lambda\,\mathrm{sign}(g_j)$. Together: $\mathrm{sign}(g_j)\max(|g_j| - \lambda, 0)$.

## Optimality
$f$ is convex, so $w$ is a minimiser iff $0 \in \partial f(w)$, iff the minimum-norm subgradient is 0. Reading it coordinate-wise gives the lasso's KKT conditions: $g_j = -\lambda\,\mathrm{sign}(w_j)$ for active weights, $|g_j| \le \lambda$ for zero ones. This is also why $w = 0$ is optimal whenever $\lambda \ge \lVert X^\top y\rVert_\infty / n$ (test 4).

## Steepest descent on a kinked function
$-v$ is the steepest-descent direction of $f$, which is why proximal and coordinate-descent lasso solvers effectively use it. Picking an arbitrary subgradient (say $s_j = 0$ at kinks) gives a direction that may not decrease $f$ at all.

## Complexity
$O(nd)$ for $X^\top (Xw - y)$.

## Common bugs
- Using `np.sign(w)` at kinks, which gives $s_j = 0$ and returns $g_j$: a valid subgradient, but not the minimum-norm one.
- Clipping $v_j$ to $[-\lambda, \lambda]$ instead of shrinking it towards 0.
- Forgetting the $1/n$, which changes which $\lambda$ makes $w = 0$ optimal.

## Interview follow-ups
- Write one step of ISTA (proximal gradient) for the lasso and show its fixed points are exactly the points where this function returns `optimal=True`.
- Why does an L1 penalty produce exact zeros while L2 only shrinks weights?
"""
