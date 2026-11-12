import numpy as np

META = dict(title='Step-size limits and convergence rate of gradient descent', tags=['Optimization', 'Training'], diff='Hard',
            runtime='numpy', lessons=['mlmaths:07.01', 'air-fanout-v1:05.05'],
            related=['hessian-matrix-and-convexity-of-linear-regressio', 'gradient-descent-on-non-convex-loss',
                     'limit-of-learning-rate-schedule'])

STATEMENT = r"""
On a quadratic $f(w) = \tfrac12 w^\top A w - b^\top w$ with $A$ symmetric positive definite, gradient descent $w_{t+1} = w_t - \eta(Aw_t - b)$ can be analysed exactly. With $w^* = A^{-1}b$ and eigenvalues $\lambda_{\min} \le \dots \le \lambda_{\max}$ of $A$, the error obeys $w_t - w^* = (I - \eta A)^t (w_0 - w^*)$, so

$$\lVert w_t - w^*\rVert_2 \le \rho^t\,\lVert w_0 - w^*\rVert_2, \qquad \rho(\eta) = \max\bigl(\lvert 1 - \eta\lambda_{\min}\rvert,\ \lvert 1 - \eta\lambda_{\max}\rvert\bigr)$$

Write `gd_convergence(A, b, w0, eta, eps)` that returns a tuple `(eta_max, eta_opt, rho, t_bound, t_actual)`:

| Output | Definition |
|---|---|
| `eta_max` | $2/\lambda_{\max}$: gradient descent converges from every start iff $0 < \eta <$ `eta_max` |
| `eta_opt` | $2/(\lambda_{\min} + \lambda_{\max})$, the step that minimises $\rho$ |
| `rho` | $\rho(\eta)$ for the given `eta` |
| `t_bound` | smallest integer $t \ge 0$ with $\rho^t \le \varepsilon$; `-1` if $\rho \ge 1$ |
| `t_actual` | smallest $t \ge 0$ with $\lVert w_t - w^*\rVert_2 \le \varepsilon\,\lVert w_0 - w^*\rVert_2$, found by running gradient descent from `w0`; `-1` if $\rho \ge 1$ |

## Rules
- `A`: `(d, d)` symmetric positive definite; `b`, `w0`: `(d,)`; `eta > 0`; `0 < eps < 1`.
- If $\rho = 0$ exactly, `t_bound` is 1.
- Simulate at most `t_bound` steps: the bound guarantees `t_actual <= t_bound`, so if the simulation hasn't met the condition by then (rounding), return `t_bound`.
- The float outputs are Python floats; the two counts are Python ints.
"""

STARTER = '''import numpy as np

def gd_convergence(A, b, w0, eta, eps):
    """Return (eta_max, eta_opt, rho, t_bound, t_actual) for gradient descent on 0.5 w'Aw - b'w."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

def gd_convergence(A, b, w0, eta, eps):
    A, b, w = np.asarray(A, dtype=float), np.asarray(b, dtype=float), np.array(w0, dtype=float)
    lam = np.linalg.eigvalsh(A)
    lo, hi = float(lam[0]), float(lam[-1])
    rho = max(abs(1 - eta * lo), abs(1 - eta * hi))      # |1 - eta*lam| is largest at an extreme eigenvalue
    out = (2 / hi, 2 / (lo + hi), rho)
    if rho >= 1:
        return (*out, -1, -1)
    if rho == 0:
        t_bound = 1
    else:
        t_bound = max(0, math.ceil(math.log(eps) / math.log(rho)))
        while rho ** t_bound > eps:                      # repair log rounding in either direction
            t_bound += 1
        while t_bound > 0 and rho ** (t_bound - 1) <= eps:
            t_bound -= 1
    w_star = np.linalg.solve(A, b)
    target = eps * np.linalg.norm(w - w_star)
    for t in range(t_bound + 1):
        if np.linalg.norm(w - w_star) <= target:
            return (*out, t_bound, t)
        w -= eta * (A @ w - b)
    return (*out, t_bound, t_bound)
'''

HINTS = [
    'In the eigenbasis of $A$ the error decouples: each component is multiplied by $(1 - \\eta\\lambda_i)$ every step. Converging needs every $|1 - \\eta\\lambda_i| < 1$.',
    '$|1 - \\eta\\lambda|$ is convex in $\\lambda$, so its maximum over the spectrum is at $\\lambda_{\\min}$ or $\\lambda_{\\max}$. Minimise the max by making $1 - \\eta\\lambda_{\\min} = -(1 - \\eta\\lambda_{\\max})$. For the bound, solve $\\rho^t \\le \\varepsilon$ with logs and check the integer you get.',
    'Use `np.linalg.eigvalsh` for the eigenvalues and `np.linalg.solve` for $w^*$. Then loop `t` from 0 to `t_bound`: test the error first, then take a step, so a start already at $w^*$ returns 0.',
]

_r = np.random.default_rng(43)
_Q, _ = np.linalg.qr(_r.normal(size=(4, 4)))
_A4 = _Q @ np.diag([0.5, 1.0, 3.0, 8.0]) @ _Q.T
_A4 = (_A4 + _A4.T) / 2
_M = _r.normal(size=(6, 6))
_A6 = _M @ _M.T / 6 + 0.3 * np.eye(6)
TESTS = [
    (np.diag([1.0, 4.0]), np.array([1.0, 4.0]), np.array([0.0, 0.0]), 0.25, 1e-3),
    (np.diag([1.0, 4.0]), np.array([1.0, 4.0]), np.array([0.0, 0.0]), 0.6, 1e-3),
    (np.diag([1.0, 10.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]), 0.1, 1e-6),
    (np.diag([1.0, 10.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0.18, 1e-6),
    (np.diag([2.0, 4.0]), np.array([2.0, 4.0]), np.array([1.0, 1.0]), 0.3, 1e-4),
    (np.eye(3) * 2, np.ones(3), np.zeros(3), 0.5, 1e-8),
    (_A4, _r.normal(size=4), _r.normal(size=4), 0.2, 1e-5),
    (_A6, _r.normal(size=6), _r.normal(size=6), 0.1, 1e-6),
]

EXPLANATION = r"""
## Why the analysis is exact on quadratics
The gradient is $Aw - b = A(w - w^*)$, so the error evolves linearly: $e_{t+1} = (I - \eta A)e_t$. In $A$'s eigenbasis that's $d$ independent scalar recurrences $e_{t+1,i} = (1-\eta\lambda_i)e_{t,i}$. The slowest one sets $\rho$, and the norm can't shrink slower than that.

## The two step sizes
- **Stability**: $|1 - \eta\lambda_{\max}| < 1 \iff \eta < 2/\lambda_{\max}$. Above it, the stiffest direction oscillates with growing amplitude (test 2). This is the classic "loss explodes when the LR is too high".
- **Optimal**: equalising the two extremes gives $\eta^* = 2/(\lambda_{\min}+\lambda_{\max})$ and $\rho^* = \frac{\kappa - 1}{\kappa + 1}$ with $\kappa = \lambda_{\max}/\lambda_{\min}$. Iterations scale like $\kappa\log(1/\varepsilon)$, which is why conditioning, not dimension, governs speed.

## Bound vs reality
The bound is tight only when $w_0 - w^*$ has weight on the slowest eigen-direction. Test 3 starts along the eigenvector with $\lambda = 10$, where $1 - 0.1\cdot10 = 0$: the actual run converges in 1 step while the bound says 132. Test 5 starts at $w^*$ and needs 0 steps.

## Complexity
$O(d^3)$ for the eigenvalues and the solve, plus $O(d^2)$ per simulated step.

## Common bugs
- Using only $\lambda_{\max}$ for $\rho$: at large $\eta$ the binding constraint is $\lambda_{\max}$, at small $\eta$ it's $\lambda_{\min}$.
- Off-by-one in `t_bound` from `int(...)` or `floor` instead of a checked `ceil`.
- Stepping before testing, so $t = 0$ is never reported.

## Interview follow-ups
- Heavy-ball momentum improves the rate to about $\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$. Why does it need the spectrum bounds to tune?
- Neural network training often sits right at the edge $\eta \approx 2/\lambda_{\max}$ of the loss Hessian ("edge of stability"). What would you measure to check that?
"""
