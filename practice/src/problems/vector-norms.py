import numpy as np

META = dict(title='Overflow-safe p-norms', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.04', 'mlmaths:06.01'], related=['cosine-similarity', 'euclidean-distance'])

STATEMENT = r"""
The $p$-norm of a vector $x \in \mathbb{R}^d$ is

$$\lVert x\rVert_p = \Big(\sum_i |x_i|^p\Big)^{1/p}, \qquad \lVert x\rVert_\infty = \max_i |x_i|$$

$p = 1$ gives the L1 (Manhattan) norm used by lasso, $p = 2$ the Euclidean length used by weight decay and gradient clipping. Write `p_norm(x, p)`.

## Inputs
- `x`: 1-D float array of length $d \ge 0$.
- `p`: a float with $p \ge 1$, or `float('inf')`.

## Output
- A Python float. The empty vector has norm `0.0` for every `p`.

## Requirement: no overflow or underflow
Computing $|x_i|^p$ directly fails for perfectly representable answers: $\lVert[10^{200}, 10^{200}]\rVert_2 \approx 1.41 \times 10^{200}$ but $(10^{200})^2$ overflows to `inf`, and tiny entries like $10^{-200}$ underflow to 0. Your function must return the correct value in both cases (within relative tolerance $10^{-6}$).
"""

STARTER = '''import numpy as np

def p_norm(x, p):
    """x: 1-D array. p >= 1 or inf. Return the p-norm as a float, without overflow/underflow."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def p_norm(x, p):
    a = np.abs(np.asarray(x, dtype=float))
    if a.size == 0:
        return 0.0
    m = a.max()
    if m == 0 or np.isinf(p):
        return float(m)
    # Homogeneity: ||x||_p = m * ||x / m||_p. After scaling every entry is in [0, 1],
    # so the powers cannot overflow, and the largest term is exactly 1, so the sum cannot underflow to 0.
    return float(m * np.sum((a / m) ** p) ** (1.0 / p))
'''

HINTS = [
    'Norms are homogeneous: $\\lVert c\\,x\\rVert_p = |c|\\,\\lVert x\\rVert_p$. Can you choose a $c$ that makes the computation safe?',
    'Divide by $m = \\max_i |x_i|$, compute the norm of $x/m$ (whose entries lie in $[0, 1]$), then multiply the result by $m$.',
    'Special cases first: empty vector or $m = 0$ gives 0; $p = \\infty$ gives $m$. Otherwise `m * np.sum((a/m)**p) ** (1/p)`.',
]

_r = np.random.default_rng(8)
TESTS = [
    (np.array([3.0, -4.0]), 2.0),
    (np.array([3.0, -4.0]), 1.0),
    (np.array([3.0, -4.0, 1.0]), float('inf')),
    (np.array([1.0, 2.0, 3.0]), 3.0),
    (np.zeros(0), 2.0),
    (np.zeros(4), 1.0),
    (np.array([1e200, 1e200]), 2.0),
    (np.array([3e-200, 4e-200]), 2.0),
    (np.array([1e300, -1e300, 5.0]), 1.5),
    (_r.normal(size=20), 2.5),
]

EXPLANATION = r"""
## Why scaling works
Every norm is homogeneous, $\lVert c\,x\rVert = |c|\,\lVert x\rVert$. With $m = \max_i |x_i|$, the scaled vector $x/m$ has entries in $[-1, 1]$ and at least one entry equal to $\pm 1$. So $\sum_i |x_i/m|^p \in [1, d]$: no term can overflow, and the sum can't underflow to 0. Multiplying back by $m$ recovers the exact norm. This is the trick behind LAPACK's `dnrm2` and `math.hypot`.

$p = \infty$ is the limit of $\lVert x\rVert_p$ as $p \to \infty$: the largest entry dominates the sum.

## Complexity
$O(d)$ time: one pass for the max, one for the sum. $O(d)$ extra memory for the scaled copy.

## Common bugs
- `np.sqrt(np.sum(x**2))` or even `np.linalg.norm(x)`: both square the raw entries and return `inf` for test 7 and `0.0` for test 8.
- Forgetting `abs`, which breaks odd $p$ such as $p = 3$ for negative entries.
- Dividing by $m$ when $m = 0$.

## Interview follow-ups
- Why is the L1 penalty sparsity-inducing and L2 is not? (The L1 ball has corners on the axes; its subgradient at 0 is an interval, so the optimum often sits exactly at 0.)
- Is $\lVert x\rVert_p$ for $0 < p < 1$ a norm? (No: it fails the triangle inequality, since its unit ball isn't convex.)
"""
