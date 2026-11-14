import numpy as np

META = dict(title='Norm-constrained least squares and its multiplier', tags=['Optimization', 'Regression'], diff='Hard',
            runtime='numpy', lessons=['mlmaths:09.52', 'mlmaths:08.03'],
            related=['subgradients-for-non-smooth-functions', 'hessian-matrix-and-convexity-of-linear-regressio',
                     'gradient-of-multivariate-linear-regression-loss'])

STATEMENT = r"""
Ridge regression adds a penalty $\lambda\lVert w\rVert^2$. An equivalent view is a hard budget on the weights:

$$\min_w\ \lVert Xw - y\rVert_2^2 \quad \text{subject to} \quad \lVert w\rVert_2 \le t$$

The Lagrangian $\mathcal L(w, \lambda) = \lVert Xw - y\rVert^2 + \lambda\,(\lVert w\rVert^2 - t^2)$ with $\lambda \ge 0$ connects the two. The KKT conditions say the solution is either

- **inactive**: the unconstrained least-squares solution already satisfies $\lVert w\rVert \le t$, and $\lambda = 0$; or
- **active**: $\lambda > 0$ and $w(\lambda) = (X^\top X + \lambda I)^{-1} X^\top y$, a ridge solution, with $\lambda$ chosen so that $\lVert w(\lambda)\rVert = t$ exactly.

Write `budget_lstsq(X, y, t)` that returns `(w, lam)`: the constrained minimiser (float array of shape `(d,)`) and its multiplier (Python `float`).

## Inputs
- `X`: `(n, d)` with full column rank ($n \ge d$), `y`: `(n,)`, `t > 0`.
- Find $\lambda$ to a relative accuracy of $10^{-10}$ or better.
- Hint at the design: $\lVert w(\lambda)\rVert$ decreases strictly as $\lambda$ grows, so a bracketing method is safe. Factor $X$ once (SVD), not once per trial $\lambda$.
"""

STARTER = '''import numpy as np

def budget_lstsq(X, y, t):
    """Minimise ||Xw - y||^2 subject to ||w|| <= t. Return (w, Lagrange multiplier)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def budget_lstsq(X, y, t):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    c = s * (U.T @ y)                                  # X^T y in the right-singular basis
    w_of = lambda lam: Vt.T @ (c / (s ** 2 + lam))     # ridge solution for any lambda, O(d^2)
    norm_of = lambda lam: np.linalg.norm(c / (s ** 2 + lam))
    if norm_of(0.0) <= t:
        return w_of(0.0), 0.0                          # constraint inactive
    # ||w(lam)|| <= ||c|| / lam, so lam = ||c|| / t is guaranteed to satisfy the budget
    lo, hi = 0.0, np.linalg.norm(c) / t
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid in (lo, hi):                            # interval can't shrink further in floating point
            break
        lo, hi = (mid, hi) if norm_of(mid) > t else (lo, mid)
    lam = (lo + hi) / 2
    return w_of(lam), float(lam)
'''

HINTS = [
    'First solve without the constraint. If that solution is inside the ball, you are done with $\\lambda = 0$. Otherwise the answer sits on the sphere $\\lVert w\\rVert = t$.',
    'With the SVD $X = USV^\\top$ and $c = S U^\\top y$, the ridge solution is $w(\\lambda) = V\\,\\frac{c}{s^2 + \\lambda}$ and its norm is $\\sqrt{\\sum_i c_i^2/(s_i^2+\\lambda)^2}$: one cheap formula per trial $\\lambda$.',
    'Bisect on $\\lambda \\in [0, \\lVert c\\rVert/t]$; the upper end works because $\\lVert w(\\lambda)\\rVert \\le \\lVert c\\rVert/\\lambda$. Halve the bracket until the midpoint stops changing, then return $w(\\lambda)$ and $\\lambda$.',
]

_r = np.random.default_rng(47)
_X = _r.normal(size=(30, 4))
_y = _X @ np.array([3.0, -2.0, 1.0, 0.5]) + 0.1 * _r.normal(size=30)
TESTS = [
    (np.eye(2), np.array([3.0, 4.0]), 10.0),
    (np.eye(2), np.array([3.0, 4.0]), 1.0),
    (np.diag([2.0, 1.0]), np.array([2.0, 2.0]), 1.0),
    (_X, _y, 10.0),
    (_X, _y, 2.0),
    (_X, _y, 0.01),
    (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]), np.zeros(3), 0.5),
    (_r.normal(size=(12, 6)) * np.array([10.0, 1.0, 1.0, 0.1, 0.1, 0.01]), _r.normal(size=12), 0.3),
]

EXPLANATION = r"""
## Why the multiplier is a ridge penalty
Setting $\nabla_w \mathcal L = 2X^\top(Xw - y) + 2\lambda w = 0$ gives $(X^\top X + \lambda I)w = X^\top y$: exactly ridge regression with penalty $\lambda$. Complementary slackness, $\lambda(\lVert w\rVert^2 - t^2) = 0$, says either the budget isn't binding ($\lambda = 0$) or the solution is on the sphere. So every budget $t$ corresponds to one ridge $\lambda$, and tighter budgets mean larger $\lambda$ (compare tests 4, 5 and 6).

## Why bisection is safe
In the SVD basis, $\lVert w(\lambda)\rVert^2 = \sum_i c_i^2/(s_i^2+\lambda)^2$, a sum of strictly decreasing functions of $\lambda$. So exactly one $\lambda > 0$ hits the budget, and any sign change bracket converges. Newton's method on $1/\lVert w(\lambda)\rVert - 1/t$ (the "secular equation" used in trust-region solvers) converges much faster, but bisection is simpler to get right and still only needs about 60 cheap $O(d)$ norm evaluations.

## The multiplier as a price
$\lambda = -\frac{1}{2t}\frac{d}{dt}\bigl(\min \lVert Xw-y\rVert^2\bigr)$: it measures how much the optimal loss would drop per unit of extra budget, scaled. A big $\lambda$ means the constraint is costing a lot.

## Complexity
One SVD, $O(nd^2)$, then $O(d)$ per bisection step and $O(d^2)$ to rebuild $w$.

## Common bugs
- Solving $(X^\top X + \lambda I)w = X^\top y$ from scratch at every trial $\lambda$: correct, but $O(d^3)$ each time.
- Returning a $\lambda > 0$ when the unconstrained solution already fits (test 1).
- Using the Lagrangian with $\frac12$ factors and getting a $\lambda$ that is off by a factor of 2 from the ridge penalty.

## Interview follow-ups
- Replace the L2 budget with $\lVert w\rVert_1 \le t$. Why is there no closed form anymore, and what solution path algorithm traces all $t$ at once?
- Trust-region methods solve exactly this subproblem with a quadratic model of the loss. Where does the budget $t$ come from in that setting?
"""
