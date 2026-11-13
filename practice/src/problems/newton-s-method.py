import numpy as np

META = dict(title="Newton's method for logistic regression", tags=['Optimization', 'Classification'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:12.02', 'mlmaths:07.01'],
            related=['adam-optimizer', 'l-bfgs', 'classify-critical-point-from-hessian'])

STATEMENT = r"""
Gradient descent uses only the slope. Newton's method also uses the curvature, jumping to the minimum of the local quadratic model:

$$w \leftarrow w - H(w)^{-1}\, \nabla F(w).$$

Apply it to L2-regularised logistic regression (no separate bias; include a column of ones in `X` if you want one):

$$F(w) = \frac1n \sum_{i=1}^n\left[\log(1 + e^{z_i}) - y_i z_i\right] + \frac{\lambda}{2}\lVert w\rVert^2, \qquad z = Xw.$$

With $p = \sigma(z)$:

$$\nabla F = \frac1n X^\top (p - y) + \lambda w, \qquad H = \frac1n X^\top \mathrm{diag}\big(p(1-p)\big)\, X + \lambda I.$$

Write `newton_logistic(X, y, lam, iters)` that starts at $w_0 = 0$, takes exactly `iters` full Newton steps, and returns the tuple `(w, losses)`: `w` float `(d,)` and `losses` float `(iters + 1,)` with $F$ at every iterate including $w_0$.

## Requirements
- `X`: float `(n, d)`; `y`: float `(n,)` of 0s and 1s; `lam >= 0`; `iters >= 0`. Tests guarantee $H$ stays invertible.
- Solve the linear system with `np.linalg.solve`; don't form $H^{-1}$.
- Compute $F$ stably (`np.logaddexp(0, z)`).
"""

STARTER = '''import numpy as np

def newton_logistic(X, y, lam, iters):
    """Return (w (d,), losses (iters+1,)) after `iters` Newton steps from w = 0."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def newton_logistic(X, y, lam, iters):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    F = lambda w: np.mean(np.logaddexp(0.0, X @ w) - y * (X @ w)) + 0.5 * lam * w @ w
    w = np.zeros(d)
    losses = [F(w)]
    for _ in range(iters):
        p = 0.5 * (1.0 + np.tanh(0.5 * (X @ w)))      # overflow-free sigmoid
        g = X.T @ (p - y) / n + lam * w
        H = (X.T * (p * (1 - p))) @ X / n + lam * np.eye(d)   # X^T diag(.) X without building diag
        w = w - np.linalg.solve(H, g)
        losses.append(F(w))
    return w, np.array(losses)
'''

HINTS = [
    'Each iteration needs $p = \\sigma(Xw)$, then the gradient and the Hessian at the current $w$, then one linear solve.',
    'Build $X^\\top \\mathrm{diag}(s) X$ without a diagonal matrix: `(X.T * s) @ X` scales column $i$ of $X^\\top$ by $s_i$.',
    'Step with `w = w - np.linalg.solve(H, g)`. Record $F$ after each step, and once before the loop.',
]

_r = np.random.default_rng(59)
_X = np.hstack([_r.normal(size=(80, 3)), np.ones((80, 1))])
_y = (_X @ np.array([1.0, -2.0, 0.5, 0.3]) + _r.normal(size=80) > 0).astype(float)
TESTS = [
    (np.array([[1.0], [-1.0], [2.0], [-0.5]]), np.array([1.0, 0.0, 0.0, 1.0]), 0.0, 3),
    (np.array([[1.0, 1.0], [2.0, 1.0], [-1.0, 1.0]]), np.array([1.0, 1.0, 0.0]), 0.1, 2),
    (_X, _y, 0.0, 8),
    (_X, _y, 0.01, 8),
    (_X, _y, 1.0, 0),
    (np.array([[3.0], [-3.0]]), np.array([1.0, 0.0]), 0.5, 10),
    (_r.normal(size=(30, 6)) * 5, (_r.random(30) < 0.5).astype(float), 0.1, 6),
]

EXPLANATION = r"""
## Why Newton converges so fast
Near a minimum with positive-definite Hessian, Newton's error satisfies $\lVert w_{k+1} - w^\star\rVert \le C\lVert w_k - w^\star\rVert^2$: the number of correct digits roughly **doubles** each step. Watch `losses` in test 3: after 4-5 steps it stops changing in the printed digits, where gradient descent would need hundreds. Newton is also affine-invariant: rescaling features doesn't change its iterates at all, so it needs no learning rate and no feature scaling.

## IRLS
For logistic regression, the Newton step is a weighted least-squares solve with weights $p(1-p)$, hence the classical name "iteratively reweighted least squares". It's how `glm` in R fits logistic models.

## Cost and when not to use it
Each step costs $O(nd^2)$ to form $H$ and $O(d^3)$ to solve, versus $O(nd)$ for a gradient step. Fine for $d$ in the hundreds or low thousands; hopeless for a neural net with $10^9$ parameters. Quasi-Newton methods (L-BFGS) approximate $H^{-1}$ from gradient differences instead.

## Common bugs
- Inverting $H$ explicitly: slower and less accurate than `solve`.
- Forgetting $\lambda I$ in the Hessian while including $\lambda w$ in the gradient: the step is then wrong and convergence becomes linear at best.
- Running unregularised Newton on separable data: $p(1-p) \to 0$, $H$ becomes singular and $w$ diverges.

## Interview follow-ups
- Newton's method can increase the loss far from the optimum. What are the standard fixes? (Damped Newton with a backtracking line search, or trust regions / Levenberg-Marquardt.)
- What does Newton's method do at a saddle point of a non-convex function? (It is attracted to it, since it seeks any stationary point of the local model. That is one reason it isn't used directly for deep nets.)
"""
