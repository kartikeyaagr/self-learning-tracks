import numpy as np

META = dict(title='Gradient descent from many starts on a non-convex loss', tags=['Optimization'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:09.08', 'air-fanout-v1:05.05'],
            related=['learning-rate-stability-and-convergence-bounds', 'directional-derivative-and-steepest-descent-dire',
                     'higher-order-derivatives-and-loss-curvature'])

STATEMENT = r"""
On a non-convex loss, where gradient descent ends up depends on where it starts. Take a one-dimensional polynomial loss

$$f(x) = \sum_{i=0}^{p} c_i\, x^i$$

and run plain gradient descent from many starting points at once:

1. At iteration $t$, if $\lvert f'(x_t)\rvert < \text{tol}$, that start has **converged**: freeze it.
2. Otherwise update $x_{t+1} = x_t - \eta\, f'(x_t)$.
3. Stop after `max_iter` updates, or earlier once every start has converged.

Write `gd_multistart(coeffs, x0, lr, max_iter, tol)` that returns `(x_final, iters)`:
- `x_final`: float array, the final position of each start.
- `iters`: int array, the number of updates each start received (0 if it began at a point with $|f'| <$ `tol`; `max_iter` if it never converged).

## Inputs
- `coeffs`: list of floats in **increasing** powers, $c_0, c_1, \dots, c_p$ with $p \ge 1$.
- `x0`: float array `(S,)` of starting points. `lr > 0`, `max_iter >= 0` (int), `tol > 0`.
- Vectorise over starts: one loop over iterations is fine, but no loop over starts.
- The tests use step sizes where the iterates stay bounded.
"""

STARTER = '''import numpy as np

def gd_multistart(coeffs, x0, lr, max_iter, tol):
    """Gradient descent on a polynomial from every start in x0. Return (x_final, updates per start)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from numpy.polynomial import polynomial as P

def gd_multistart(coeffs, x0, lr, max_iter, tol):
    dcoef = P.polyder(np.asarray(coeffs, dtype=float))
    x = np.array(x0, dtype=float)
    iters = np.zeros(len(x), dtype=int)
    for _ in range(max_iter):
        g = P.polyval(x, dcoef)
        active = np.abs(g) >= tol          # converged starts stay frozen
        if not active.any():
            break
        x[active] -= lr * g[active]
        iters += active
    return x, iters
'''

HINTS = [
    'Differentiate the polynomial once up front (`numpy.polynomial.polynomial.polyder`) and evaluate $f\'$ at all starts with `polyval`.',
    'Keep a boolean mask of starts that haven\'t converged. Each iteration, compute the gradient, update the mask with `abs(g) >= tol`, and move only the masked entries.',
    'Count updates with `iters += active` (booleans add as 0/1). Check the convergence test **before** updating, so a start that begins at a stationary point gets 0 updates.',
]

_tilted = [0.0, 1.0, -3.0, 0.0, 1.0]          # x^4 - 3x^2 + x: two minima of different depth, one maximum
TESTS = [
    (_tilted, np.array([-2.0, 2.0]), 0.01, 1000, 1e-8),
    ([0.0, 0.0, 1.0], np.array([1.0, -4.0, 0.0]), 0.25, 100, 1e-6),
    (_tilted, np.linspace(-2.0, 2.0, 9), 0.01, 2000, 1e-8),
    ([0.0, 0.0, 0.0, 1.0], np.array([0.0, 1.0]), 0.1, 50, 1e-6),
    (_tilted, np.array([0.5]), 0.01, 0, 1e-8),
    ([0.0, 0.0, -1.0, 0.0, 0.25], np.array([0.0, 1e-3, -1e-3, 3.0]), 0.05, 500, 1e-8),
    ([1.0, -2.0, 0.5, 0.3, -0.1, 0.01], np.array([-1.0, 0.0, 2.0, 4.0]), 0.02, 3000, 1e-8),
]

EXPLANATION = r"""
## What the tests show
On $f(x) = x^4 - 3x^2 + x$ there are two minima, near $x \approx -1.30$ (the deeper one) and $x \approx 1.13$, separated by a maximum near $0.17$. Starts on either side of the maximum fall into different basins, and gradient descent has no way to know the other basin is better: it only sees the local slope. On $f = x^3$ (test 4) the start at 0 is already stationary, and the start at 1 creeps towards the flat inflection point and never meets the tolerance, because $f'(x) = 3x^2$ shrinks quadratically as it approaches.

## Freezing converged starts
Without the mask, a converged start keeps taking tiny steps, and its iteration count stops meaning anything. Testing the stopping rule before the update also handles starts that sit exactly on a stationary point, including the unstable ones such as the maximum of $-x^2 + x^4/4$ at 0 (test 6): the gradient is exactly 0 there, so plain gradient descent never leaves. Starts at $\pm 10^{-3}$ do escape, slowly at first.

## Complexity
$O(\text{max\_iter}\cdot S\cdot p)$.

## Common bugs
- Checking convergence after updating, which gives every start at least one update.
- Using `np.polyder`/`np.polyval` (the old API, which expects **decreasing** powers) on increasing-power coefficients.
- Updating all starts and only masking the counter, which moves converged starts.

## Interview follow-ups
- Add momentum. Can it carry a start over the hump into the deeper basin? Find a start where it does.
- Why are saddle points, rather than bad local minima, the main obstacle in high-dimensional neural network losses?
"""
