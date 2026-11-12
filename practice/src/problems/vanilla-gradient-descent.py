import numpy as np

META = dict(title='Gradient descent on a quadratic bowl', tags=['Optimization', 'Calculus'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:01.04', 'mlmaths:09.03'],
            related=['minimum-of-axis-aligned-paraboloid', 'classical-momentum', 'learning-rate-sweep'])

STATEMENT = r"""
Gradient descent repeatedly steps against the gradient:

$$x_{t+1} = x_t - \eta\, \nabla f(x_t).$$

Run it on the quadratic

$$f(x) = \tfrac12 x^\top Q x - b^\top x, \qquad \nabla f(x) = Q x - b,$$

where $Q$ is symmetric.

Write `gradient_descent(Q, b, x0, lr, steps)` that returns the whole trajectory as a float array of shape `(steps + 1, d)`: row 0 is `x0`, row $t$ is $x_t$.

## Inputs
- `Q`: float `(d, d)`, symmetric. `b`, `x0`: float `(d,)`.
- `lr`: positive float. `steps`: int `>= 0` (with `steps == 0` the result is just `x0` as a single row).
- Don't modify `x0` in place.
"""

STARTER = '''import numpy as np

def gradient_descent(Q, b, x0, lr, steps):
    """Return the trajectory, shape (steps + 1, d), starting with x0."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gradient_descent(Q, b, x0, lr, steps):
    Q, b = np.asarray(Q, dtype=float), np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float)            # a copy, so the caller's x0 is untouched
    path = [x.copy()]
    for _ in range(steps):
        x = x - lr * (Q @ x - b)
        path.append(x.copy())
    return np.array(path)
'''

HINTS = [
    'The gradient of $\\tfrac12 x^\\top Q x$ is $Qx$ when $Q$ is symmetric, and the gradient of $b^\\top x$ is $b$.',
    'Keep a Python list of positions, append a copy after every update, and stack it with `np.array` at the end.',
    'Use `x = x - lr * (Q @ x - b)` rather than `x -= ...` so you never write into the caller\'s array.',
]

_r = np.random.default_rng(13)
_M = _r.normal(size=(4, 4))
_spd = _M @ _M.T + np.eye(4)
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 10.0]]), np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.1, 3),
    (np.array([[2.0]]), np.array([4.0]), np.array([0.0]), 0.25, 4),
    (np.array([[2.0]]), np.array([4.0]), np.array([10.0]), 1.5, 5),
    (np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([0.0, 0.0]), np.array([5.0, -5.0]), 0.2, 0),
    (np.array([[1.0, 0.0], [0.0, -1.0]]), np.array([0.0, 0.0]), np.array([1.0, 1e-3]), 0.5, 6),
    (_spd, _r.normal(size=4), _r.normal(size=4), 1.0 / np.linalg.eigvalsh(_spd).max(), 25),
]

EXPLANATION = r"""
## Why it converges (and when it doesn't)
Write $e_t = x_t - x^\star$ with $Q x^\star = b$. Then $e_{t+1} = (I - \eta Q)\, e_t$. In the eigenbasis of $Q$ each component is multiplied by $1 - \eta\lambda_k$ every step, so GD converges iff $|1 - \eta\lambda_k| < 1$ for all $k$, i.e. $0 < \eta < 2/\lambda_{\max}$ (with $Q$ positive definite). Test 3 uses $\eta\lambda = 3$ and oscillates outward. Test 5 has a negative eigenvalue: the component along it grows by $1.5\times$ per step, which is GD sliding off a saddle.

## Rate
The slowest component shrinks by $\max_k |1 - \eta\lambda_k|$. The best fixed step $\eta = 2/(\lambda_{\min} + \lambda_{\max})$ gives the rate $(\kappa - 1)/(\kappa + 1)$, where $\kappa = \lambda_{\max}/\lambda_{\min}$. Ill-conditioned problems crawl.

## Complexity
Each step is one matrix-vector product, $O(d^2)$, so $O(T d^2)$ total, plus $O(T d)$ to store the path.

## Common bugs
- `x -= lr * grad` on the input array mutates the caller's `x0`.
- Appending `x` without copying when updating in place: every row ends up equal to the last position.
- Using `Q.T @ x` vs `Q @ x` doesn't matter here (symmetric), but writing the gradient as $2Qx$ does: that is the gradient of $x^\top Q x$, not $\tfrac12 x^\top Q x$.

## Interview follow-ups
- Why does the step size have to shrink as $\lambda_{\max}$ grows, and how does batch normalisation or feature scaling help? (They reduce curvature spread, so one step size suits every direction.)
- Exact line search picks $\eta_t$ minimising $f(x_t - \eta g_t)$. Derive it for a quadratic. ($\eta_t = g_t^\top g_t / g_t^\top Q g_t$.)
"""
