import numpy as np

META = dict(title='Hessian and convexity of the ridge loss', tags=['Optimization', 'Regression'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:07.01', 'mlmaths:08.03'],
            related=['gradient-of-multivariate-linear-regression-loss', 'learning-rate-stability-and-convergence-bounds',
                     'pca-projection'])

STATEMENT = r"""
For linear regression with an L2 penalty,

$$L(w) = \frac{1}{2n}\lVert Xw - y\rVert_2^2 + \frac{\lambda}{2}\lVert w\rVert_2^2$$

the Hessian $\nabla^2 L$ decides everything about the landscape: whether there's a unique minimum, and how fast gradient descent can get there.

Write `ridge_hessian(X, lam)` that returns a tuple `(H, eig_min, eig_max, strictly_convex)`:

- `H`: the Hessian matrix, shape `(d, d)`.
- `eig_min`, `eig_max`: its smallest and largest eigenvalues, as Python floats.
- `strictly_convex`: Python `bool`, `True` when `eig_min > 1e-10 * max(1.0, eig_max)`.

## Inputs
- `X`: float array `(n, d)` with $n \ge 1$; `lam` $\ge 0$.
- The function doesn't take `y` or `w`. Work out why it doesn't need them.
"""

STARTER = '''import numpy as np

def ridge_hessian(X, lam):
    """Return (Hessian of the ridge loss, smallest eigenvalue, largest eigenvalue, strictly convex?)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ridge_hessian(X, lam):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    H = X.T @ X / n + lam * np.eye(d)            # constant: the loss is exactly quadratic in w
    eig = np.linalg.eigvalsh(H)                  # symmetric: real eigenvalues, ascending
    lo, hi = float(eig[0]), float(eig[-1])
    return H, lo, hi, bool(lo > 1e-10 * max(1.0, hi))
'''

HINTS = [
    'Differentiate twice. The gradient is $\\frac1n X^\\top(Xw - y) + \\lambda w$. What is left after differentiating once more with respect to $w$?',
    '$H = \\frac1n X^\\top X + \\lambda I$. It doesn\'t depend on $w$ or $y$, because the loss is a quadratic in $w$.',
    'Use `np.linalg.eigvalsh` (for symmetric matrices, ascending order). $X^\\top X$ is positive semidefinite, so `eig_min` can only be 0 (up to rounding) when $X$ has dependent columns and $\\lambda = 0$.',
]

_r = np.random.default_rng(16)
_dup = _r.normal(size=(10, 2))
_dup = np.c_[_dup, _dup[:, 0] - 2 * _dup[:, 1]]          # third column depends on the first two
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 2.0]]), 0.0),
    (np.array([[1.0, 1.0], [2.0, 2.0]]), 0.0),
    (np.array([[1.0, 1.0], [2.0, 2.0]]), 0.5),
    (_dup, 0.0),
    (_dup, 1e-3),
    (np.array([[1.0, 2.0, 3.0]]), 0.0),
    (_r.normal(size=(50, 4)) * np.array([1.0, 10.0, 0.1, 3.0]), 0.0),
    (_r.normal(size=(5, 8)), 0.2),
]

EXPLANATION = r"""
## Why the Hessian is constant
$L$ is a quadratic in $w$: $\frac{1}{2}w^\top H w - \frac1n y^\top X w + \text{const}$. The second derivative of a quadratic is its matrix, so $H = \frac1n X^\top X + \lambda I$ everywhere. Neither $y$ nor $w$ appears.

## Convexity from eigenvalues
$v^\top X^\top X v = \lVert Xv\rVert^2 \ge 0$, so $X^\top X$ is always positive semidefinite and the loss is always convex. It's **strictly** convex (a unique minimiser) iff $H \succ 0$, i.e. $\lambda > 0$ or $X$ has full column rank. Duplicate or collinear features (tests 2 and 4), or fewer samples than features (test 6), give a flat valley of equally good solutions; any $\lambda > 0$ lifts every eigenvalue by $\lambda$ and removes it. That's ridge's original motivation.

## Why both extreme eigenvalues matter
$\lambda_{\max}$ caps the gradient-descent step size ($\eta < 2/\lambda_{\max}$), and $\kappa = \lambda_{\max}/\lambda_{\min}$ sets the convergence speed. Test 7 scales features by up to 100x, which makes $\kappa$ about $10^4$: that's why you standardise features before gradient descent.

## Complexity
$O(nd^2)$ to form $H$, $O(d^3)$ for the eigenvalues.

## Common bugs
- Using `np.linalg.eig`: it may return complex numbers with tiny imaginary parts and in no particular order.
- Testing `eig_min > 0` exactly. Rounding turns a true 0 into $\pm10^{-16}$, so a relative threshold is needed.
- Forgetting the $1/n$, which scales every eigenvalue by $n$ but leaves the convexity verdict unchanged.

## Interview follow-ups
- Logistic regression's Hessian is $\frac1n X^\top \mathrm{diag}(p(1-p)) X$. Is the loss strictly convex on separable data? What happens to the weights?
- How does batch normalisation relate to conditioning the Hessian of the next layer?
"""
