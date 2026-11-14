import numpy as np

META = dict(title='Project vectors onto a direction', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:06.13', 'mlmaths:06.01'], related=['dot-product', 'gram-schmidt-orthogonalization', 'orthogonal-projection-matrix'])

STATEMENT = r"""
The projection of $x$ onto the line spanned by a non-zero vector $u$ is the closest point to $x$ on that line:

$$\operatorname{proj}_u(x) = \frac{x \cdot u}{u \cdot u}\, u$$

The **residual** $x - \operatorname{proj}_u(x)$ is orthogonal to $u$. Removing a direction this way is used to debias embeddings and is the core step of Gram–Schmidt.

Write `project(X, u)` for a batch of vectors.

## Inputs
- `X`: float array `(n, d)`, one vector per row, $n, d \ge 1$.
- `u`: float array `(d,)`. It does not need to be unit length.

## Output
- A tuple `(P, R)` of two `(n, d)` arrays: `P[i]` is the projection of `X[i]` onto `u`, and `R = X - P`.
- If `u` is the zero vector, the line is just the origin: `P` is all zeros and `R` equals `X`.

## Requirements
No Python loops over rows.
"""

STARTER = '''import numpy as np

def project(X, u):
    """X: (n, d), u: (d,). Return (P, R): projections of each row onto u, and residuals X - P."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def project(X, u):
    X = np.asarray(X, dtype=float)
    u = np.asarray(u, dtype=float)
    uu = u @ u
    if uu == 0:
        return np.zeros_like(X), X.copy()
    coef = X @ u / uu                  # (n,): scalar coordinate of each row along u
    P = coef[:, None] * u              # (n, 1) * (d,) -> (n, d)
    return P, X - P
'''

HINTS = [
    'Each row needs just one scalar, $x_i \\cdot u / (u \\cdot u)$, which then scales $u$.',
    '`X @ u` gives all $n$ dot products at once. Divide by `u @ u`, then turn the `(n,)` coefficients into a column to broadcast against `u`.',
    '`coef = X @ u / (u @ u)`; `P = coef[:, None] * u`; return `(P, X - P)`. Handle `u @ u == 0` first.',
]

_r = np.random.default_rng(31)
TESTS = [
    (np.array([[3.0, 4.0]]), np.array([1.0, 0.0])),
    (np.array([[2.0, 2.0], [1.0, -1.0]]), np.array([2.0, 2.0])),
    (np.array([[1.0, 2.0, 3.0]]), np.array([0.0, 0.0, 0.0])),
    (np.array([[5.0], [-2.0]]), np.array([3.0])),
    (np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]]), np.array([1e-150, 2e-150, -1e-150])),
    (_r.normal(size=(6, 4)), _r.normal(size=4)),
    (_r.normal(size=(3, 8)) * 100, _r.normal(size=8) * 0.01),
]

EXPLANATION = r"""
## Why the formula works
The closest point $c\,u$ on the line minimises $\lVert x - c\,u\rVert^2$. Setting the derivative in $c$ to zero gives $u\cdot(x - c\,u) = 0$, i.e. the residual is orthogonal to $u$, and $c = (x\cdot u)/(u\cdot u)$. Scaling $u$ by any non-zero constant leaves $c\,u$ unchanged, so `u` need not be normalised.

## Checking your answer
`R @ u` should be ~0 for every row, and `P + R == X`. Test 5 uses a tiny $u$ ($\sim 10^{-150}$). It works here because `u @ u` ($\sim 10^{-300}$) is still a normal float, but a further factor of $10^{-5}$ would underflow it to 0. Normalising $u$ first (divide by its largest entry) makes the computation scale-safe.

## Complexity
$O(nd)$ time and memory.

## Common bugs
- Dividing by $\lVert u\rVert$ instead of $\lVert u\rVert^2$ when $u$ isn't unit length.
- `coef * u` without `[:, None]`: shape `(n,)` times `(d,)` fails, or broadcasts wrongly when $n = d$.

## Interview follow-ups
- How do you project onto the span of several vectors? (If they're orthonormal, sum the individual projections; otherwise use $A(A^\top A)^{-1}A^\top$ or orthonormalise first.)
- Why does Gram–Schmidt subtract projections one at a time from the running residual rather than from the original vector? (Modified Gram–Schmidt: it loses far less orthogonality in floating point.)
"""
