import numpy as np

META = dict(title='RMSProp on the Rosenbrock valley', tags=['Optimization', 'Training'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:01.14', 'air-fanout-v1:05.05'],
            related=['adagrad', 'adam-optimizer', 'optimizer-showdown'])

STATEMENT = r"""
RMSProp fixes AdaGrad's ever-shrinking step by replacing the running sum of squared gradients with an exponential moving average:

$$s_t = \rho\, s_{t-1} + (1-\rho)\, g_t^2, \qquad x_t = x_{t-1} - \eta\, \frac{g_t}{\sqrt{s_t} + \epsilon}, \qquad s_0 = 0.$$

Test it on the Rosenbrock function, a curved, narrow valley with its minimum at $(a, a^2)$:

$$f(x, y) = (a - x)^2 + b\,(y - x^2)^2.$$

Write `rmsprop_rosenbrock(start, lr, steps, rho=0.9, eps=1e-8, a=1.0, b=100.0)` that returns the trajectory as a float array of shape `(steps + 1, 2)`; row 0 is `start`.

## Requirements
- Derive the gradient of $f$ yourself and compute it exactly (no finite differences).
- `start`: float array `(2,)`. `steps >= 0`, `lr > 0`, `0 <= rho < 1`.
"""

STARTER = '''import numpy as np

def rmsprop_rosenbrock(start, lr, steps, rho=0.9, eps=1e-8, a=1.0, b=100.0):
    """Return the RMSProp trajectory on the Rosenbrock function, shape (steps + 1, 2)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def rosenbrock_grad(p, a, b):
    x, y = p
    r = y - x * x                                   # residual of the curved valley floor
    return np.array([-2.0 * (a - x) - 4.0 * b * x * r, 2.0 * b * r])

def rmsprop_rosenbrock(start, lr, steps, rho=0.9, eps=1e-8, a=1.0, b=100.0):
    p = np.array(start, dtype=float)
    s = np.zeros(2)
    path = [p.copy()]
    for _ in range(steps):
        g = rosenbrock_grad(p, a, b)
        s = rho * s + (1 - rho) * g * g             # EMA of squared gradients, forgets old scale
        p = p - lr * g / (np.sqrt(s) + eps)
        path.append(p.copy())
    return np.array(path)
'''

HINTS = [
    'Differentiate: $\\partial f/\\partial x = -2(a - x) - 4bx(y - x^2)$ and $\\partial f/\\partial y = 2b(y - x^2)$.',
    'Keep `s` as a length-2 array of zeros. Each step: gradient, update `s` with the moving average, then divide the step by `np.sqrt(s) + eps`.',
    'On step 1, `s = (1 - rho) g^2`, so the first move is about `lr / sqrt(1 - rho)` per coordinate, which is large. That\'s expected (Adam\'s bias correction exists to tame it).',
]

TESTS = [
    (np.array([-1.0, 1.5]), 0.01, 3),
    (np.array([0.0, 0.0]), 0.01, 3, 0.5),
    (np.array([1.0, 1.0]), 0.01, 5),
    (np.array([-1.5, 2.0]), 0.001, 0),
    (np.array([-1.2, 1.0]), 0.002, 150),
    (np.array([2.0, -1.0]), 0.005, 100, 0.99),
    (np.array([-0.5, 2.0]), 0.002, 300),
    (np.array([0.5, 0.5]), 0.01, 50, 0.9, 1e-8, 2.0, 10.0),
]
META['tol'] = [1e-6, 1e-7]

EXPLANATION = r"""
## Why an EMA instead of a sum
$s_t$ estimates the recent mean of $g^2$ with an effective window of about $1/(1-\rho)$ steps. The step $g/\sqrt{s}$ is then roughly the gradient divided by its recent RMS: a unitless quantity of size $\sim 1$, so each coordinate moves about $\eta$ per step whether its raw gradient is $10^{-3}$ or $10^3$. Unlike AdaGrad, old large gradients are forgotten, so the step size can recover when the landscape flattens.

## Why Rosenbrock
Across the valley the curvature is huge ($\sim 200b x^2$), along it it's tiny. Plain GD must use a step small enough for the cross-valley direction and then crawls along the floor. RMSProp normalises both, so it moves along the floor at a speed set by $\eta$. It does **not** converge exactly with a constant $\eta$: near the minimum it hovers at a distance on the order of $\eta$, because the normalised step doesn't shrink with the gradient. In that regime tiny gradients get blown up to full-size steps, so trajectories become extremely sensitive to round-off; the tests stop before reaching it.

## Complexity
$O(1)$ per step here; $O(d)$ in general, with one extra state vector.

## Common bugs
- Sign errors in $\partial f/\partial x$: there are two terms, and the second is $-4bx(y - x^2)$.
- Initialising $s$ with $g_1^2$ instead of 0: a different algorithm (no first-step jump), and different numbers.
- Using `rho` as the weight on the **new** gradient.

## Interview follow-ups
- RMSProp with momentum on top of the normalised step is very close to which optimizer? (Adam without bias correction.)
- Why do adaptive methods sometimes generalise worse than SGD with momentum on image classification? (One proposed reason: they equalise step sizes across directions, including ones SGD would barely move along, landing in different minima.)
"""
