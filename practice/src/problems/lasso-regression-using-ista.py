import numpy as np

META = dict(title='Lasso regression with ISTA', tags=['Optimization', 'Regression'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:02.04', 'mlmaths:08.02'],
            related=['vanilla-gradient-descent', 'adamw-decoupled-weight-decay', 'convex-polytope-membership'])

STATEMENT = r"""
The lasso adds an $\ell_1$ penalty to least squares, which drives some weights to **exactly** zero:

$$\min_w\; F(w) = \frac{1}{2n}\lVert Xw - y\rVert^2 + \lambda \lVert w\rVert_1.$$

The $\ell_1$ term isn't differentiable at 0, so plain gradient descent doesn't apply. ISTA (iterative shrinkage-thresholding) takes a gradient step on the smooth part and then applies the **soft-threshold** operator, the exact minimiser of the $\ell_1$ part plus a proximity term:

$$w_{k+1} = S_{\lambda/L}\!\left(w_k - \tfrac1L \nabla g(w_k)\right), \qquad \nabla g(w) = \tfrac1n X^\top(Xw - y), \qquad S_\tau(z) = \mathrm{sign}(z)\max(|z| - \tau, 0).$$

Use the step $1/L$ with $L = \lambda_{\max}\!\left(\tfrac1n X^\top X\right)$, the Lipschitz constant of $\nabla g$.

Write `lasso_ista(X, y, lam, iters)` that starts from $w_0 = 0$, runs exactly `iters` ISTA updates, and returns the tuple `(w, objective)`:

- `w`: float array `(d,)`.
- `objective`: float array `(iters + 1,)` with $F(w_0), F(w_1), \dots, F(w_{\text{iters}})$.

## Inputs
`X`: float `(n, d)`, not all zeros; `y`: float `(n,)`; `lam >= 0`; `iters >= 0`. No intercept: centre your data first if you need one.
"""

STARTER = '''import numpy as np

def lasso_ista(X, y, lam, iters):
    """Return (w (d,), objective (iters+1,)) after `iters` ISTA steps from w = 0."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def lasso_ista(X, y, lam, iters):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    L = np.linalg.norm(X, 2) ** 2 / n                 # top eigenvalue of X^T X / n = sigma_max^2 / n
    F = lambda w: 0.5 * np.mean((X @ w - y) ** 2) + lam * np.abs(w).sum()
    w = np.zeros(d)
    obj = [F(w)]
    for _ in range(iters):
        z = w - (X.T @ (X @ w - y) / n) / L           # gradient step on the smooth part
        w = np.sign(z) * np.maximum(np.abs(z) - lam / L, 0.0)   # prox of (lam/L)*||.||_1
        obj.append(F(w))
    return w, np.array(obj)
'''

HINTS = [
    'Split $F$ into the smooth least-squares part and the $\\ell_1$ part. Take a normal gradient step on the first, then fix it up with the soft-threshold.',
    '$L$ is the largest eigenvalue of $X^\\top X / n$, which equals `np.linalg.norm(X, 2) ** 2 / n` (the spectral norm squared).',
    'Soft-threshold in one vectorised line: `np.sign(z) * np.maximum(np.abs(z) - lam / L, 0)`. Note the threshold is `lam / L`, not `lam`.',
]

_r = np.random.default_rng(43)
_X = _r.normal(size=(50, 8))
_w = np.array([3.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0])
_y = _X @ _w + 0.1 * _r.normal(size=50)
_lam_max = np.abs(_X.T @ _y).max() / 50
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([2.0, 0.2]), 0.25, 3),
    (np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]), 0.0, 2),
    (_X, _y, 0.1, 200),
    (_X, _y, 1.1 * _lam_max, 20),
    (_X, _y, 0.5, 0),
    (np.array([[1.0, 1.0], [1.0, 1.0], [1.0, -1.0]]), np.array([2.0, 2.0, 0.0]), 0.05, 100),
    (_r.normal(size=(10, 20)), _r.normal(size=10), 0.2, 150),
]
META['tol'] = [1e-6, 1e-7]

EXPLANATION = r"""
## Why soft-thresholding
ISTA is proximal gradient descent. Each step minimises a local model: the linearisation of the smooth part plus a quadratic proximity term $\frac L2\lVert w - w_k\rVert^2$, plus the **exact** $\ell_1$ term. That subproblem separates by coordinate, and each 1-D problem $\min_w \frac L2 (w - z)^2 + \lambda|w|$ has the closed-form answer $S_{\lambda/L}(z)$: shrink toward 0 by $\lambda/L$, and stop at 0 if you would cross it. That "stop at 0" is why the lasso returns exact zeros, which subgradient descent never does.

## Why step $1/L$
With $L$ the Lipschitz constant of $\nabla g$, the local model upper-bounds $F$, so every ISTA step decreases $F$ (check `objective`: it is non-increasing). The rate is $F(w_k) - F^\star = O(1/k)$. FISTA adds Nesterov-style momentum and gets $O(1/k^2)$.

## When the answer is all zeros
$w = 0$ is optimal iff $0 \in \partial F(0)$, i.e. $\lVert X^\top y\rVert_\infty / n \le \lambda$. Test 4 uses $1.1\times$ that threshold, so ISTA never leaves the origin.

## Complexity
$O(nd)$ per iteration, plus a one-off spectral-norm computation ($O(nd\min(n, d))$, or a few power-iteration steps in practice).

## Common bugs
- Thresholding at $\lambda$ instead of $\lambda/L$ (only correct when $L = 1$).
- Using $\lVert X\rVert_F^2/n$ for $L$: a valid upper bound, so ISTA still converges, but it takes smaller steps and produces different iterates.
- Penalising an intercept column. Standard lasso leaves the intercept unpenalised.

## Interview follow-ups
- Coordinate descent is what scikit-learn uses for the lasso. Why is it so effective here? (Each coordinate update is the same closed-form soft-threshold, and sparse solutions mean most coordinates stay at 0 cheaply.)
- How does elastic net change the prox step? (Add $\frac{\lambda_2}{2}\lVert w\rVert^2$: the prox becomes soft-threshold followed by division by $1 + \lambda_2/L$.)
"""
