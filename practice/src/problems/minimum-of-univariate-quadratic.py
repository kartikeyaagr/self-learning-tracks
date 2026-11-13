import numpy as np

META = dict(title='Minimise a batch of 1-D quadratics', tags=['Optimization', 'Calculus'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:09.08', 'air-fanout-v1:01.02'],
            related=['minimum-of-axis-aligned-paraboloid', 'vanilla-gradient-descent'])

STATEMENT = r"""
A one-variable quadratic $f(x) = a x^2 + b x + c$ has a global minimum only when it opens upward. Setting $f'(x) = 2ax + b = 0$ gives the minimiser

$$x^\star = -\frac{b}{2a}, \qquad f(x^\star) = c - \frac{b^2}{4a}.$$

Write `quadratic_minimum(a, b, c)` that handles a whole batch of quadratics at once and returns the tuple `(x_star, f_star)`.

## Inputs
- `a`, `b`, `c`: float arrays of the same shape `(n,)`, one quadratic per index. `n` may be 0.

## Output rules, per index
| Case | `x_star` | `f_star` |
|---|---|---|
| `a > 0` | $-b / 2a$ | $c - b^2 / 4a$ |
| `a == 0` and `b == 0` (constant) | `0.0` | `c` |
| otherwise (`a < 0`, or `a == 0` with `b != 0`) | `nan` | `-inf` |

Both outputs are float arrays of shape `(n,)`. Use NumPy, no Python loop, and don't let a division by zero emit a warning.
"""

STARTER = '''import numpy as np

def quadratic_minimum(a, b, c):
    """a, b, c: float arrays (n,). Return (x_star, f_star), each a float array (n,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def quadratic_minimum(a, b, c):
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    up = a > 0
    safe_a = np.where(up, a, 1.0)                    # divide only where it is meaningful
    x = np.where(up, -b / (2 * safe_a), np.nan)
    f = np.where(up, c - b**2 / (4 * safe_a), -np.inf)
    const = (a == 0) & (b == 0)                      # flat line: every x is optimal, report x = 0
    x = np.where(const, 0.0, x)
    f = np.where(const, c, f)
    return x, f
'''

HINTS = [
    'Split the batch with boolean masks: opens upward, constant, and everything else (unbounded below).',
    'Dividing by `a` where `a <= 0` produces warnings even if you throw the result away. Replace those denominators with 1 first, e.g. `np.where(a > 0, a, 1.0)`.',
    'Build `x` and `f` with `np.where(up, formula, nan / -inf)`, then overwrite the constant entries with `0.0` and `c`.',
]

_r = np.random.default_rng(3)
TESTS = [
    (np.array([1.0]), np.array([-4.0]), np.array([1.0])),
    (np.array([2.0, -1.0, 0.0]), np.array([4.0, 0.0, 3.0]), np.array([0.0, 5.0, 1.0])),
    (np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([7.5, -2.0])),
    (np.array([1e-12, 1e8]), np.array([1.0, -2e8]), np.array([0.0, 3.0])),
    (np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)),
    (np.array([3.0, 3.0]), np.array([-6.0, 6.0]), np.array([-1.0, -1.0])),
    (_r.uniform(-2, 2, size=12), _r.normal(size=12) * 5, _r.normal(size=12)),
]

EXPLANATION = r"""
## Why the formula
$f'(x) = 2ax + b$ vanishes at $x^\star = -b/2a$, and $f''(x) = 2a$. When $a > 0$ the second derivative is positive everywhere, so this stationary point is the unique global minimum. Substituting back gives $f(x^\star) = c - b^2/4a$ (complete the square: $f(x) = a(x + b/2a)^2 + c - b^2/4a$).

When $a < 0$ the parabola opens downward and $f \to -\infty$ as $|x| \to \infty$. When $a = 0$ and $b \ne 0$ it is a sloped line, also unbounded below. Only the flat line has a (non-unique) minimum.

## Complexity
$O(n)$ time and memory, all vectorised.

## Common bugs
- Computing `-b / (2 * a)` on the whole batch and masking afterwards. The answer is right but NumPy emits divide-by-zero warnings, and in some codebases warnings are errors.
- Treating `a < 0` as a maximum problem and returning $-b/2a$ anyway. That point is the worst choice, not the best.
- Forgetting that `a == 0, b == 0` is bounded (constant), not unbounded.

## Interview follow-ups
- Gradient descent with step $\eta$ on this function multiplies the error $x - x^\star$ by $1 - 2a\eta$ each step. For which $\eta$ does it converge? ($0 < \eta < 1/a$.)
- How would you minimise the quadratic on an interval $[l, u]$? (Clip $x^\star$ to the interval; for $a \le 0$ compare the two endpoints.)
"""
