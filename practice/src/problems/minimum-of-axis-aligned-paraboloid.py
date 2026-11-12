import numpy as np

META = dict(title='Minimum of a separable paraboloid', tags=['Optimization', 'Calculus'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:07.01', 'mlmaths:09.08'],
            related=['minimum-of-univariate-quadratic', 'convexity-certificate-via-hessian', 'vanilla-gradient-descent'])

STATEMENT = r"""
An axis-aligned paraboloid in $d$ dimensions has no cross terms, so it splits into $d$ independent 1-D problems:

$$f(x) = \sum_{i=1}^{d} \left(a_i x_i^2 + b_i x_i\right) + c, \qquad a_i > 0.$$

Write `paraboloid_minimum(a, b, c)` that returns the tuple `(x_star, f_star, kappa)`:

- `x_star`: float array `(d,)`, the unique minimiser.
- `f_star`: Python `float`, the minimum value.
- `kappa`: Python `float`, the condition number of the Hessian, $\kappa = \max_i a_i / \min_i a_i$. It measures how stretched the bowl is, and it will matter once you run gradient descent on it.

## Inputs
- `a`: float array `(d,)` with every entry strictly positive, `d >= 1`.
- `b`: float array `(d,)`; `c`: float.

No loops: vectorise over the coordinates.
"""

STARTER = '''import numpy as np

def paraboloid_minimum(a, b, c):
    """Return (x_star (d,), f_star float, kappa float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def paraboloid_minimum(a, b, c):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    x = -b / (2 * a)                         # each coordinate solves 2 a_i x_i + b_i = 0
    f = c - np.sum(b**2 / (4 * a))           # the d one-dimensional minima add up
    return x, float(f), float(a.max() / a.min())   # Hessian is diag(2a); the 2 cancels in the ratio
'''

HINTS = [
    'With no cross terms, the gradient has components $2a_i x_i + b_i$. Each one depends on a single coordinate.',
    'Solve each component for zero: `x = -b / (2 * a)`. Plug back in coordinate by coordinate and sum.',
    'The Hessian is the diagonal matrix $\\mathrm{diag}(2a)$. Its eigenvalues are the diagonal entries, so $\\kappa = \\max a / \\min a$.',
]

_r = np.random.default_rng(21)
TESTS = [
    (np.array([1.0, 4.0]), np.array([-2.0, 8.0]), 0.0),
    (np.array([0.5]), np.array([3.0]), 2.0),
    (np.array([2.0, 2.0, 2.0]), np.array([0.0, 0.0, 0.0]), -1.5),
    (np.array([1e-4, 1e4]), np.array([1.0, 1.0]), 10.0),
    (np.array([3.0, 1.0, 9.0]), np.array([-6.0, 0.5, 18.0]), 4.0),
    (_r.uniform(0.1, 5.0, size=10), _r.normal(size=10) * 3, 0.25),
    (_r.uniform(1.0, 2.0, size=50), _r.normal(size=50), -3.0),
]

EXPLANATION = r"""
## Why it separates
$\partial f/\partial x_i = 2a_i x_i + b_i$ involves only $x_i$, and the Hessian $\nabla^2 f = \mathrm{diag}(2a_1, \dots, 2a_d)$ is diagonal and positive definite when every $a_i > 0$. A positive-definite Hessian means $f$ is strictly convex, so the single stationary point is the global minimum. Minimising a sum of independent terms is the same as minimising each term: $f^\star = c + \sum_i (-b_i^2/4a_i)$.

## Why the condition number matters
Gradient descent with one step size $\eta$ must keep $\eta < 1/a_{\max}$ to stay stable along the steepest axis. Along the flattest axis the error then shrinks by a factor $1 - 2\eta a_{\min} \approx 1 - 2/\kappa$ per step, so the number of iterations grows roughly linearly with $\kappa$. A bowl with $\kappa = 10^8$ is "easy" analytically and miserable for plain gradient descent.

## Complexity
$O(d)$ time and memory.

## Common bugs
- Dropping the 2 in $x_i = -b_i/(2a_i)$.
- Forgetting $c$ in the minimum value, or adding it once per coordinate.
- Returning NumPy scalars where floats are asked for is fine for grading, but `kappa` computed as `a.max() / a.min()` must use `a`, not `b`.

## Interview follow-ups
- Now allow cross terms: $f(x) = x^\top A x + b^\top x + c$ with $A$ symmetric positive definite. What is the minimiser? ($x^\star = -\tfrac12 A^{-1} b$, solved with `np.linalg.solve`, never an explicit inverse.)
- How does a diagonal preconditioner (rescaling coordinate $i$ by $1/\sqrt{a_i}$) change $\kappa$? (It becomes 1. That is the intuition behind adaptive methods like AdaGrad and Adam.)
"""
