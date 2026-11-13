import numpy as np

META = dict(title='Nesterov accelerated gradient', tags=['Optimization', 'Training'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:01.14'],
            related=['classical-momentum', 'vanilla-gradient-descent', 'optimizer-showdown'])

STATEMENT = r"""
Nesterov momentum differs from heavy-ball momentum in one place: it evaluates the gradient at the **look-ahead** point where the velocity is about to carry it, then corrects.

$$v_{t+1} = \mu\, v_t - \eta\, \nabla f(x_t + \mu v_t), \qquad x_{t+1} = x_t + v_{t+1}, \qquad v_0 = 0.$$

Run it on $f(x) = \tfrac12 x^\top Q x - b^\top x$ ($Q$ symmetric, $\nabla f(x) = Qx - b$).

Write `nesterov_descent(Q, b, x0, lr, mu, steps)` that returns the tuple `(path, grad_norms)`:

- `path`: float array `(steps + 1, d)`, row 0 is `x0`, row $t$ is $x_t$.
- `grad_norms`: float array `(steps + 1,)`, $\lVert \nabla f(x_t) \rVert_2$ at each row of `path` (the gradient at the iterate itself, not at the look-ahead point).

## Inputs
- `Q`: float `(d, d)` symmetric; `b`, `x0`: float `(d,)`; `lr > 0`; `0 <= mu < 1`; `steps >= 0`.
"""

STARTER = '''import numpy as np

def nesterov_descent(Q, b, x0, lr, mu, steps):
    """Return (path (steps+1, d), grad_norms (steps+1,)) for Nesterov momentum with v0 = 0."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def nesterov_descent(Q, b, x0, lr, mu, steps):
    Q, b = np.asarray(Q, dtype=float), np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(steps):
        look = x + mu * v                      # where the velocity alone would take us
        v = mu * v - lr * (Q @ look - b)       # correct using the gradient there
        x = x + v
        path.append(x.copy())
    path = np.array(path)
    grads = path @ Q - b                       # Q symmetric, so row t is (Q x_t - b)
    return path, np.linalg.norm(grads, axis=1)
'''

HINTS = [
    'Start from the heavy-ball loop. Only the point where you evaluate the gradient changes.',
    'Compute `look = x + mu * v` before touching `v`, take the gradient there, then update `v` and `x` exactly as in classical momentum.',
    'The reported gradient norms are at the iterates in `path`: `np.linalg.norm(path @ Q - b, axis=1)` (using symmetry of `Q`).',
]

_r = np.random.default_rng(19)
_M = _r.normal(size=(4, 4))
TESTS = [
    (np.array([[1.0]]), np.array([1.0]), np.array([0.0]), 0.5, 0.5, 3),
    (np.array([[1.0, 0.0], [0.0, 25.0]]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0.03, 0.9, 4),
    (np.array([[1.0, 0.0], [0.0, 25.0]]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0.03, 0.0, 4),
    (np.array([[4.0, 1.0], [1.0, 3.0]]), np.array([1.0, 2.0]), np.array([0.0, 0.0]), 0.1, 0.9, 0),
    (np.array([[1.0, 0.0], [0.0, 100.0]]), np.array([1.0, 1.0]), np.array([-2.0, 2.0]), 0.01, 0.95, 40),
    (_M @ _M.T + 0.2 * np.eye(4), _r.normal(size=4), _r.normal(size=4) * 3, 0.03, 0.85, 30),
]

EXPLANATION = r"""
## Why the look-ahead helps
Heavy-ball momentum computes the gradient at $x_t$ but then moves by $\mu v_t$ anyway. If that momentum is about to overshoot the valley, the gradient at $x_t$ doesn't know yet. Nesterov peeks at $x_t + \mu v_t$: if the look-ahead has already crossed the minimum, the gradient there points back and brakes the velocity one step earlier. The result is less oscillation for the same $\mu$.

For smooth convex functions, a Nesterov schedule with growing $\mu_t$ attains the optimal $O(1/t^2)$ rate for first-order methods, versus $O(1/t)$ for GD.

## The form frameworks use
Substituting $y_t = x_t + \mu v_t$ (track the look-ahead point instead) gives an update that needs only the gradient at the stored parameters, which is how PyTorch's `nesterov=True` is implemented: $v \leftarrow \mu v + g$, $\theta \leftarrow \theta - \eta (g + \mu v)$.

## Complexity
$O(d^2)$ per step, the same as classical momentum. One gradient evaluation per step.

## Common bugs
- Updating `v` first and then computing the look-ahead from the new velocity.
- Reporting gradient norms at the look-ahead points rather than at the iterates.
- Expecting monotone decrease: Nesterov is not a descent method, so $f$ and $\lVert\nabla f\rVert$ can bump up.

## Interview follow-ups
- On a 1-D quadratic with curvature $\lambda$, write the 2x2 linear recurrence for $(x_t, v_t)$ under each method. Which has smaller spectral radius for $\mu = 0.9$, $\eta\lambda = 0.5$?
- Why does Nesterov momentum matter less in deep learning with small batches than the theory suggests? (Gradient noise dominates, and the acceleration guarantee is for exact gradients.)
"""
