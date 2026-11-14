import numpy as np

META = dict(title='Directional derivatives and steepest descent in three norms', tags=['Calculus', 'Optimization'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:09.08', 'air-fanout-v1:01.04'],
            related=['gradient-descent-on-non-convex-loss', 'hessian-matrix-and-convexity-of-linear-regressio'])

STATEMENT = r"""
Take the quadratic $f(x) = \tfrac12 x^\top A x + b^\top x$ with $A$ symmetric, so $\nabla f(x) = Ax + b$.

The **directional derivative** along a unit vector $u$ is $D_u f(x) = \nabla f(x)^\top u$. "Steepest descent" means the unit direction that makes $D_u f$ most negative, and what counts as a unit vector depends on the norm. Write `steepest(A, b, x, u, norm)` returning a tuple `(dd, d)`:

- `dd`: the directional derivative of $f$ at `x` along `u / ||u||_2` (always the Euclidean normalisation), as a Python `float`.
- `d`: the steepest-descent direction $\arg\min_{\lVert v\rVert \le 1} \nabla f(x)^\top v$ in the requested norm, as a float array of shape `(n,)`:

| `norm` | Steepest-descent direction |
|---|---|
| `'l2'` | $-g / \lVert g\rVert_2$ |
| `'l1'` | $-\operatorname{sign}(g_i)\, e_i$ for the coordinate $i$ with the largest $\lvert g_i\rvert$ (smallest index on ties) |
| `'linf'` | $-\operatorname{sign}(g)$, elementwise, with $\operatorname{sign}(0) = 0$ |

where $g = \nabla f(x)$. If $g$ is the zero vector, return `d` as all zeros for every norm.

## Inputs
`A`: `(n, n)` symmetric floats; `b`, `x`, `u`: `(n,)` floats, with `u` nonzero.
"""

STARTER = '''import numpy as np

def steepest(A, b, x, u, norm):
    """Return (directional derivative along u/||u||, steepest-descent direction in the given norm)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def steepest(A, b, x, u, norm):
    A, b, x, u = (np.asarray(a, dtype=float) for a in (A, b, x, u))
    g = A @ x + b
    dd = float(g @ u / np.linalg.norm(u))
    d = np.zeros_like(g)
    if not g.any():
        return dd, d
    if norm == 'l2':
        d = -g / np.linalg.norm(g)
    elif norm == 'l1':
        i = int(np.argmax(np.abs(g)))      # argmax returns the first maximum: smallest index on ties
        d[i] = -np.sign(g[i])              # spend the whole unit L1 budget on the steepest coordinate
    else:
        d = -np.sign(g)                    # every coordinate at its +-1 box corner
    return dd, d
'''

HINTS = [
    'Everything starts from $g = Ax + b$. The directional derivative is one dot product with the normalised `u`.',
    'Minimising $g^\\top v$ over a norm ball is a linear objective over a convex set, so the answer sits on the boundary. For L2 it points straight against $g$. For L1 the ball is a diamond: which corner gives the most negative $g^\\top v$?',
    'L1: put all the weight on `i = np.argmax(np.abs(g))` with sign `-np.sign(g[i])`. Linf: the box corner `-np.sign(g)`. Check `g.any()` first for the zero-gradient case.',
]

_r = np.random.default_rng(22)
_M = _r.normal(size=(4, 4))
_A = _M + _M.T
TESTS = [
    (np.eye(2), np.zeros(2), np.array([3.0, 4.0]), np.array([1.0, 0.0]), 'l2'),
    (np.eye(2), np.zeros(2), np.array([3.0, 4.0]), np.array([1.0, 1.0]), 'l1'),
    (np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([1.0, -1.0]), np.array([1.0, 2.0]), np.array([0.0, -2.0]), 'linf'),
    (np.eye(3), np.array([1.0, -2.0, 0.0]), np.zeros(3), np.array([0.0, 0.0, 5.0]), 'linf'),
    (np.eye(3), np.array([-2.0, 2.0, 1.0]), np.zeros(3), np.array([1.0, 2.0, 2.0]), 'l1'),
    (np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([-3.0, -6.0]), np.array([1.0, 1.0]), np.array([1.0, -1.0]), 'l2'),
    (_A, _r.normal(size=4), _r.normal(size=4), _r.normal(size=4), 'l2'),
    (_A, _r.normal(size=4), _r.normal(size=4), _r.normal(size=4), 'l1'),
    (_A, _r.normal(size=4), _r.normal(size=4), _r.normal(size=4), 'linf'),
]

EXPLANATION = r"""
## Why these three directions
$\min_{\lVert v\rVert\le 1} g^\top v = -\lVert g\rVert_*$, the **dual norm** of $g$, and the minimiser is where that's attained:
- L2 is its own dual: $-\lVert g\rVert_2$, at $v = -g/\lVert g\rVert_2$ (Cauchy-Schwarz).
- L1's dual is L-infinity: $-\max_i|g_i|$, at a single signed coordinate vector. That's **coordinate descent**.
- L-infinity's dual is L1: $-\sum_i |g_i|$, at $v = -\mathrm{sign}(g)$. That's **sign descent**, the idea behind signSGD; Adam with very short moving averages behaves much the same way.

So "steepest descent" isn't one direction: it's a choice of geometry. Gradient descent is steepest descent in the Euclidean norm.

## Ties and zeros
With L1, several coordinates can share the largest $|g_i|$, and any convex combination of their corners is optimal; the statement picks the first so the answer is unique. For L-infinity, a zero coordinate contributes nothing, so its sign is free; the statement fixes it at 0.

## Complexity
$O(n^2)$ for $Ax$, then $O(n)$.

## Common bugs
- Normalising `u` by its L1 norm when `norm == 'l1'`. The statement always uses the Euclidean length for `dd`.
- Returning `+g` directions (ascent).
- Dividing by $\lVert g\rVert_2 = 0$ at a stationary point.

## Interview follow-ups
- Steepest descent in the norm $\lVert v\rVert_P = \sqrt{v^\top P v}$ gives $-P^{-1}g$ (up to scale). What choice of $P$ gives Newton's method?
- Why does sign descent behave well when gradient coordinates have wildly different scales?
"""
