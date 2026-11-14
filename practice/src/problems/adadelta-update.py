import numpy as np

META = dict(title='AdaDelta: an update with no global learning rate', tags=['Optimization', 'Training'], diff='Medium', runtime='numpy',
            related=['rmsprop', 'adagrad', 'adam-step', 'nadam-update'])

STATEMENT = r"""
AdaDelta keeps two running averages per parameter and rescales each step so that its units match the parameter's. Write `adadelta(A, b, x0, steps, rho=0.9, eps=1e-6)`.

Start with $E[g^2] = 0$ and $E[\Delta x^2] = 0$. Each step:
$$E[g^2] \leftarrow \rho E[g^2] + (1-\rho) g^2$$
$$\Delta x = -\frac{\sqrt{E[\Delta x^2] + \epsilon}}{\sqrt{E[g^2] + \epsilon}}\, g$$
$$E[\Delta x^2] \leftarrow \rho E[\Delta x^2] + (1-\rho) \Delta x^2, \qquad x \leftarrow x + \Delta x$$

The objective is the quadratic $f(x) = \tfrac12 x^\top A x - b^\top x$, so the gradient is $g = Ax - b$ (`A` is a symmetric positive-definite matrix, `b` a vector). Return the list of iterates `[x_0, x_1, ..., x_steps]` as float arrays.
"""

STARTER = '''import numpy as np

def adadelta(A, b, x0, steps, rho=0.9, eps=1e-6):
    """Return the list of iterates, starting with x0."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def adadelta(A, b, x0, steps, rho=0.9, eps=1e-6):
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    Eg, Ed = np.zeros_like(x), np.zeros_like(x)
    out = [x.copy()]
    for _ in range(steps):
        g = A @ x - b
        Eg = rho * Eg + (1 - rho) * g * g
        dx = -np.sqrt(Ed + eps) / np.sqrt(Eg + eps) * g
        Ed = rho * Ed + (1 - rho) * dx * dx
        x = x + dx
        out.append(x.copy())
    return out
'''

HINTS = [
    'Update the average of squared gradients before computing the step, and the average of squared steps after it.',
    'At the start E[dx^2] = 0, so the first steps have size about sqrt(eps) / RMS(g): AdaDelta starts slowly and speeds up.',
    'Everything is element-wise; each parameter gets its own effective step size.',
]

_I2 = np.eye(2)
_ill = np.diag([1.0, 100.0])
_sp = np.array([[3.0, 1.0], [1.0, 2.0]])
TESTS = [
    (np.diag([2.0, 20.0]), np.zeros(2), np.array([1.0, 1.0]), 5),
    (2 * np.eye(1), np.array([6.0]), np.array([0.0]), 40),
    (_ill, np.array([1.0, 1.0]), np.array([-1.0, 1.0]), 25),
    (_sp, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 10, 0.95, 1e-4),
    (2 * _I2, np.array([6.0, 6.0]), np.array([5.0, -5.0]), 3, 0.5, 1e-2),
]

EXPLANATION = r"""
## Units
In plain gradient descent the step $\eta g$ has units of the gradient times the learning rate, not of the parameter. AdaDelta's ratio $\sqrt{E[\Delta x^2]}/\sqrt{E[g^2]}$ is an approximation to the inverse curvature, like a diagonal Newton step, so the update has the parameter's units and no global learning rate is needed.

## Relation to RMSProp and AdaGrad
AdaGrad divides by the root of the **sum** of all squared gradients, so steps shrink forever. RMSProp replaces the sum with an exponential average. AdaDelta goes further and replaces the learning rate with the running RMS of past updates.

## In practice
AdaDelta's slow start (steps of size about $\sqrt{\epsilon}$) and its sensitivity to $\epsilon$ mean it is rarely used for large models; Adam and AdamW dominate. It is still a useful example of how adaptive methods are built.

## Interview follow-ups
- What happens to the step size if gradients suddenly become very large?
"""
