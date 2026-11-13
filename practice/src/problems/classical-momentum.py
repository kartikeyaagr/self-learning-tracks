import numpy as np

META = dict(title='Gradient descent with momentum', tags=['Optimization', 'Training'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:01.14'],
            related=['vanilla-gradient-descent', 'nesterov-momentum'])

STATEMENT = r"""
Momentum (the "heavy ball") keeps a velocity that accumulates past gradients, so it builds speed along directions where gradients agree and cancels oscillation where they flip sign. Use this form:

$$v_{t+1} = \mu\, v_t - \eta\, \nabla f(x_t), \qquad x_{t+1} = x_t + v_{t+1}, \qquad v_0 = 0.$$

Run it on $f(x) = \tfrac12 x^\top Q x - b^\top x$ ($Q$ symmetric, $\nabla f(x) = Qx - b$).

Write `momentum_descent(Q, b, x0, lr, mu, steps)` that returns the tuple `(path, f_values)`:

- `path`: float array `(steps + 1, d)`, row 0 is `x0`.
- `f_values`: float array `(steps + 1,)`, $f$ evaluated at each row of `path`.

## Inputs
- `Q`: float `(d, d)` symmetric; `b`, `x0`: float `(d,)`.
- `lr > 0`, `0 <= mu < 1`, `steps >= 0`. With `mu = 0` this must reduce exactly to plain gradient descent.
"""

STARTER = '''import numpy as np

def momentum_descent(Q, b, x0, lr, mu, steps):
    """Return (path (steps+1, d), f_values (steps+1,)) for heavy-ball momentum with v0 = 0."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def momentum_descent(Q, b, x0, lr, mu, steps):
    Q, b = np.asarray(Q, dtype=float), np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(steps):
        v = mu * v - lr * (Q @ x - b)        # velocity: decayed history plus the new downhill push
        x = x + v
        path.append(x.copy())
    path = np.array(path)
    # f for every row at once: 0.5 * x^T Q x - b^T x
    f = 0.5 * np.einsum('ti,ij,tj->t', path, Q, path) - path @ b
    return path, f
'''

HINTS = [
    'You need one extra state vector, the velocity, initialised to zeros with the same shape as `x`.',
    'Order matters: update `v` from the gradient at the **current** `x`, then move `x` by the new `v`.',
    'Evaluate $f$ on the stacked path in one shot: `0.5 * np.sum((path @ Q) * path, axis=1) - path @ b`.',
]

_r = np.random.default_rng(17)
_M = _r.normal(size=(3, 3))
TESTS = [
    (np.array([[1.0]]), np.array([1.0]), np.array([0.0]), 0.5, 0.5, 3),
    (np.array([[1.0, 0.0], [0.0, 25.0]]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0.03, 0.9, 4),
    (np.array([[1.0, 0.0], [0.0, 25.0]]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0.03, 0.0, 4),
    (np.array([[2.0, 0.5], [0.5, 1.0]]), np.array([1.0, -1.0]), np.array([3.0, 3.0]), 0.1, 0.8, 0),
    (np.array([[1.0, 0.0], [0.0, 100.0]]), np.array([1.0, 1.0]), np.array([-2.0, 2.0]), 0.01, 0.95, 40),
    (_M @ _M.T + 0.5 * np.eye(3), _r.normal(size=3), _r.normal(size=3) * 4, 0.05, 0.7, 30),
]

EXPLANATION = r"""
## Why momentum helps on ill-conditioned bowls
Plain GD on $\kappa = \lambda_{\max}/\lambda_{\min}$ needs about $\kappa$ steps per factor of $e$. Momentum turns each eigen-direction into a damped oscillator. With $\eta = 4/(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}})^2$ and $\mu = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^2$ the rate becomes $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$: about $\sqrt\kappa$ steps instead of $\kappa$. Velocity also averages away components that alternate sign (the zig-zag across a ravine) and reinforces components that are consistent (along the valley floor).

## The other convention
PyTorch writes $v \leftarrow \mu v + g$, $x \leftarrow x - \eta v$. That's the same algorithm with velocity rescaled by $-1/\eta$, identical for a fixed learning rate. They differ once $\eta$ changes over time: in PyTorch's form a learning-rate drop takes effect immediately; in this form the old velocity keeps its old scale.

## Complexity
$O(d^2)$ per step for the matrix-vector product, $O(d)$ extra memory for the velocity.

## Common bugs
- Moving `x` with the old velocity and updating `v` afterwards (that's a different, lagged method).
- Forgetting `v0 = 0` and starting with the first gradient, which skips the warm-up behaviour.
- With `mu` close to 1 and a large step, $f$ is **not** monotone; overshooting is expected (see test 5), not a bug.

## Interview follow-ups
- Momentum is an exponential moving average of gradients. What effective number of past gradients does $\mu = 0.9$ average over? (About $1/(1-\mu) = 10$.)
- What does momentum do on a saddle point compared with plain GD? (It accumulates speed along the escape direction, leaving faster.)
"""
