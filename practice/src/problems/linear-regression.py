import numpy as np

META = dict(title='Least-squares linear regression with an intercept', tags=['Regression', 'Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:08.03', 'mlmaths:08.01'], related=['ridge-regression', 'regression-metrics'])

STATEMENT = r"""
Ordinary least squares fits $\hat y = Xw + b$ by minimising the sum of squared residuals

$$\min_{w,\,b}\ \sum_{i=1}^{n} \left(y_i - x_i^\top w - b\right)^2 .$$

Write `fit_linear_regression(X, y)` that returns the exact minimiser `(w, b)`. No gradient descent: solve it directly.

## Input
- `X`: float array `(n, d)`; `y`: float array `(n,)`.
- The augmented matrix $[X \;\; \mathbf{1}]$ has full column rank, so the answer is unique.

## Output
- `w`: float array `(d,)`; `b`: Python `float`.

## Requirements
- The intercept is a real parameter, not something you can drop: data are not centred.
- Use NumPy only. Prefer a numerically stable solver over inverting $X^\top X$ by hand.
"""

STARTER = '''import numpy as np

def fit_linear_regression(X, y):
    """X: (n, d), y: (n,). Return (w (d,), b float) minimising squared error."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def fit_linear_regression(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    A = np.column_stack([X, np.ones(len(X))])          # a column of ones carries the intercept
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)       # QR/SVD-based, never forms A^T A
    return coef[:-1], float(coef[-1])
'''

HINTS = [
    'An intercept is just one more weight whose feature is always 1.',
    'Append a column of ones to `X`, then solve the least-squares problem for the augmented weight vector in one call.',
    '`coef, *_ = np.linalg.lstsq(np.column_stack([X, np.ones(n)]), y, rcond=None)`; the last entry is `b`, the rest is `w`.',
]

_r = np.random.default_rng(3)
_X = _r.normal(size=(30, 3))
_X2 = _r.normal(size=(50, 5)) * np.array([1.0, 10.0, 0.1, 3.0, 1.0])
TESTS = [
    (np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([1.0, 3.0, 5.0, 7.0])),
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]), np.array([2.0, 3.0, 4.0, 6.0])),
    (np.array([[1.0], [2.0], [3.0]]), np.array([4.0, 4.0, 4.0])),
    (np.array([[-5.0], [0.0], [5.0], [10.0]]), np.array([-3.0, 1.0, 2.0, 9.0])),
    (np.array([[1000.0, 2.0], [1001.0, 3.0], [1003.0, 2.5], [1002.0, 4.0], [1005.0, 1.0]]), np.array([1.0, 2.0, 0.0, 3.0, -1.0])),
    (_X, _X @ np.array([2.0, -1.0, 0.5]) + 4.0 + 0.1 * _r.normal(size=30)),
    (_X2, _X2 @ _r.normal(size=5) - 2.0 + _r.normal(size=50)),
]

EXPLANATION = r"""
## Why it works
Stack a column of ones onto $X$ to get $A$ and let $\theta = (w, b)$. The objective is $\lVert y - A\theta\rVert^2$, and setting its gradient to zero gives the normal equations $A^\top A\,\theta = A^\top y$. Geometrically, $A\theta$ is the orthogonal projection of $y$ onto the column space of $A$: the residual is perpendicular to every column, including the column of ones, which is why OLS residuals always sum to zero.

## Why lstsq and not inv
`np.linalg.inv(A.T @ A) @ A.T @ y` is algebraically the same but squares the condition number of $A$. With features on very different scales (test 5 has one column near 1000) you lose about twice as many digits. `lstsq` uses an SVD of $A$ directly.

## Complexity
$O(n d^2)$ time for the factorisation, $O(nd)$ memory.

## Common bugs
- Forgetting the intercept, or centring `X` but not `y` (or vice versa).
- Returning `b` as a 1-element array instead of a float.
- Using `np.linalg.solve(X.T @ X, ...)` on rank-deficient data and getting a `LinAlgError` or garbage.

## Interview follow-ups
- What changes if $d > n$? (Infinitely many minimisers. `lstsq` returns the minimum-norm one; ridge picks a unique one.)
- Derive the closed form for the slope in 1-D. ($\hat w = \mathrm{cov}(x, y)/\mathrm{var}(x)$, $\hat b = \bar y - \hat w \bar x$.)
"""
