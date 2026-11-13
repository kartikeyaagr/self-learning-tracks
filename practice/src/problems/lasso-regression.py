import numpy as np

META = dict(title='Lasso by cyclic coordinate descent', tags=['Regression', 'Optimization'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.04'], related=['ridge-regression', 'linear-regression'])

STATEMENT = r"""
The lasso uses an L1 penalty, which drives some weights to **exactly** zero:

$$\min_w\ \frac{1}{2n}\lVert y - Xw\rVert_2^2 + \alpha \lVert w\rVert_1 .$$

There is no closed form, but each one-dimensional subproblem has one. Holding every other weight fixed, the best $w_j$ is

$$\rho_j = \frac{1}{n} x_j^\top \Big(y - \sum_{k\ne j} x_k w_k\Big), \qquad z_j = \frac1n x_j^\top x_j, \qquad w_j \leftarrow \frac{S(\rho_j, \alpha)}{z_j},$$

where $x_j$ is column $j$ and $S(\rho, \alpha) = \mathrm{sign}(\rho)\,\max(|\rho| - \alpha,\, 0)$ is soft-thresholding.

Write `lasso_cd(X, y, alpha, n_sweeps)` that starts from `w = 0` and performs `n_sweeps` full sweeps. Each sweep updates $w_0, w_1, \dots, w_{d-1}$ **in that order**, and each update uses the latest values of the other weights.

## Input / output
- `X`: float array `(n, d)` with no all-zero column; `y`: float array `(n,)`; `alpha >= 0`; `n_sweeps >= 0`.
- No intercept: fit exactly the objective above.
- Return `w` as a float array `(d,)`.
"""

STARTER = '''import numpy as np

def lasso_cd(X, y, alpha, n_sweeps):
    """Cyclic coordinate descent for (1/2n)||y - Xw||^2 + alpha*||w||_1 from w = 0."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def lasso_cd(X, y, alpha, n_sweeps):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    z = (X ** 2).sum(axis=0) / n
    r = y.copy()                                       # running residual y - Xw
    for _ in range(n_sweeps):
        for j in range(d):
            rho = X[:, j] @ r / n + z[j] * w[j]        # residual with feature j's contribution added back
            new = np.sign(rho) * max(abs(rho) - alpha, 0.0) / z[j]
            r -= X[:, j] * (new - w[j])                # keep r in sync in O(n)
            w[j] = new
    return w
'''

HINTS = [
    'Within a sweep, the update for $w_j$ must see the values of $w_0, \\dots, w_{j-1}$ you just changed. Vectorising across features computes a different (Jacobi-style) algorithm.',
    'Keep a residual vector `r = y - X @ w`. Then $\\rho_j = x_j^\\top r / n + z_j w_j$, because adding back feature $j$\'s own contribution gives the partial residual.',
    'After computing the new $w_j$, update the residual with `r -= X[:, j] * (new - old)` so the next coordinate sees it. Soft-threshold with `np.sign(rho) * max(abs(rho) - alpha, 0)`.',
]

_r = np.random.default_rng(21)
_X = _r.normal(size=(40, 6))
_y = _X @ np.array([3.0, 0.0, 0.0, -2.0, 0.0, 0.5]) + 0.2 * _r.normal(size=40)
TESTS = [
    (np.array([[1.0], [2.0], [3.0]]), np.array([2.0, 4.0, 6.0]), 0.5, 1),
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([3.0, 0.2, 3.1]), 0.3, 5),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 1.0]), 0.0, 0),
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([1.0, -1.0, 0.0]), 10.0, 3),
    (np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), np.array([1.0, 2.0, 3.0]), 0.1, 50),
    (_X, _y, 0.1, 100),
    (_X, _y, 1.0, 100),
    (_X * 100.0, _y, 5.0, 30),
]

EXPLANATION = r"""
## Why soft-thresholding
Restricted to one coordinate, the objective is $\frac{z_j}{2}w_j^2 - \rho_j w_j + \alpha|w_j| + \text{const}$. Away from 0 it's a parabola with minimum at $(\rho_j \mp \alpha)/z_j$. At 0 the subgradient of $|w_j|$ is $[-1, 1]$, so 0 is optimal whenever $|\rho_j| \le \alpha$. Combining the three cases gives $S(\rho_j, \alpha)/z_j$. The flat region of $S$ is what produces exact zeros: test 4 has a large $\alpha$ and every weight stays at 0.

## Why it converges
The smooth part is convex and the penalty is separable across coordinates. For that structure, cyclic coordinate descent converges to the global minimum (Tseng, 2001). Collinear features (test 5) are fine too: the solution may not be unique, but the iterates still converge.

## Complexity
With the running residual, one coordinate update is $O(n)$, so a sweep is $O(nd)$. Recomputing $y - Xw$ from scratch for each coordinate would cost $O(nd^2)$ per sweep.

## Common bugs
- Updating all coordinates simultaneously from the same residual. It can oscillate and doesn't match the cyclic algorithm.
- Forgetting the $+ z_j w_j$ term, so $\rho_j$ uses the full residual instead of the partial one.
- Dividing $\alpha$ by $z_j$ as well, which is the right threshold only when columns are normalised to $z_j = 1$.

## Interview follow-ups
- Why does L1 give sparse solutions and L2 doesn't? (Draw the diamond vs the circle touching the loss contours, or compare the subgradient at 0.)
- How do you compute a whole regularisation path efficiently? (Start at $\alpha_{\max} = \max_j |x_j^\top y|/n$, where $w = 0$, and warm-start each smaller $\alpha$ from the previous solution.)
"""
