import numpy as np

META = dict(title='Linear SVM with Pegasos-style SGD', tags=['Classification', 'Optimization'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:09.08'], related=['averaged-perceptron', 'logistic-regression'])

STATEMENT = r"""
A soft-margin linear SVM minimises the regularised hinge loss

$$F(w) = \frac{\lambda}{2}\lVert w\rVert^2 + \frac1n\sum_{i=1}^n \max\!\big(0,\ 1 - y_i\, w^\top x_i\big).$$

(There's no separate bias: if you want one, the data include a constant column.) $F$ is strongly convex, so SGD with step size $\eta_t = 1/(\lambda t)$ converges to its unique minimum.

Write `train_linear_svm(X, y, lam, n_epochs, seed)` that returns `(w, F(w))` for the final `w`.

## Procedure (follow exactly)
1. `rng = np.random.default_rng(seed)`, `w = 0` (shape `(d,)`), step counter `t = 0`.
2. For each epoch, draw the visiting order with `rng.permutation(n)` (one call per epoch, from the same `rng`). For each index `i` in that order:
   1. `t += 1`, $\eta = 1/(\lambda t)$.
   2. Compute the margin $m = y_i\,w^\top x_i$ with the **current** `w`.
   3. Shrink: $w \leftarrow (1 - \eta\lambda)\,w$. Then, if $m < 1$, add $\eta\,y_i x_i$.
   4. Project onto the ball of radius $1/\sqrt\lambda$: if $\lVert w\rVert > 1/\sqrt\lambda$, rescale $w$ to have exactly that norm.
3. Return `w` (float array `(d,)`) and the objective $F(w)$ on the full data (Python `float`).

## Input
- `X`: float `(n, d)`; `y`: int `(n,)` in $\{-1, +1\}$; `lam > 0`; `n_epochs >= 1`; `seed`: int.
"""

STARTER = '''import numpy as np

def train_linear_svm(X, y, lam, n_epochs, seed):
    """Pegasos SGD on the regularised hinge loss. Return (w (d,), objective float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def train_linear_svm(X, y, lam, n_epochs, seed):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w, t = np.zeros(d), 0
    radius = 1.0 / np.sqrt(lam)
    for _ in range(n_epochs):
        for i in rng.permutation(n):
            t += 1
            eta = 1.0 / (lam * t)
            margin = y[i] * (X[i] @ w)                  # uses w before this step's shrink
            w = (1.0 - eta * lam) * w                   # gradient of the L2 term
            if margin < 1.0:
                w = w + eta * y[i] * X[i]               # subgradient of an active hinge
            norm = np.linalg.norm(w)
            if norm > radius:                           # the optimum is known to lie in this ball
                w = w * (radius / norm)
    hinge = np.maximum(0.0, 1.0 - y * (X @ w))
    return w, float(0.5 * lam * (w @ w) + hinge.mean())
'''

HINTS = [
    'The stochastic subgradient for sample $i$ is $\\lambda w - [y_i w^\\top x_i < 1]\\, y_i x_i$. A step of size $\\eta$ along it is "shrink, then maybe add".',
    'Compute the margin **before** shrinking. Note $1 - \\eta\\lambda = 1 - 1/t$, so the first step wipes `w` to zero before adding the first example.',
    'Create the generator once, outside the epoch loop, and call `rng.permutation(n)` at the start of each epoch. Project with `w *= radius / norm` only when `norm > radius`.',
]

_r = np.random.default_rng(41)
_X = np.column_stack([_r.normal(size=(50, 2)), np.ones(50)])
_y = np.where(_X[:, 0] - 0.5 * _X[:, 1] + 0.3 + 0.4 * _r.normal(size=50) > 0, 1, -1)
TESTS = [
    (np.array([[2.0, 1.0], [1.5, 2.0], [-1.0, -1.5], [-2.0, -0.5]]), np.array([1, 1, -1, -1]), 0.1, 1, 0),
    (np.array([[2.0, 1.0], [1.5, 2.0], [-1.0, -1.5], [-2.0, -0.5]]), np.array([1, 1, -1, -1]), 0.1, 20, 0),
    (np.array([[1.3]]), np.array([1]), 0.5, 3, 7),
    (np.array([[1.0, 0.2], [0.9, -0.1], [-1.1, 0.3], [0.8, 0.5]]), np.array([1, -1, -1, 1]), 2.0, 5, 3),
    (np.array([[30.0, -10.0], [25.0, 5.0], [-40.0, 12.0], [-35.0, -3.0]]), np.array([1, 1, -1, -1]), 0.01, 4, 11),
    (_X, _y, 0.05, 10, 1),
    (_X, _y, 0.5, 3, 2),
]

EXPLANATION = r"""
## Why this step size
$F$ is $\lambda$-strongly convex. For strongly convex objectives, SGD with $\eta_t = 1/(\lambda t)$ reaches $F(\bar w) - F^\star = O(\log T/(\lambda T))$, with no learning rate to tune. A nice side effect: ignoring the projection, after $t$ steps $w$ is exactly $\frac{1}{\lambda t}$ times the sum of $y_i x_i$ over the steps whose hinge was active, because each shrink factor $(1 - 1/t)$ telescopes.

## Why the projection is safe
At the optimum, strong duality gives $\frac{\lambda}{2}\lVert w^\star\rVert^2 \le F(0) = 1$, so $\lVert w^\star\rVert \le \sqrt{2/\lambda}$. A tighter dual argument gives $1/\sqrt\lambda$. Projecting onto a set that contains the optimum never hurts convergence and stops early iterates from wandering far, which is visible in test 5 with a tiny $\lambda$.

## Complexity
$O(d)$ per step and $O(nd)$ per epoch, independent of how many support vectors there are. That's why Pegasos-style training scales to millions of examples while kernel SVM solvers don't.

## Common bugs
- Computing the margin after shrinking, or after adding.
- Re-creating the generator each epoch (every epoch then repeats the same order).
- Using the mean hinge gradient over the whole batch: that's subgradient descent, a different algorithm.
- Regularising a bias column without realising it. Here that's intended, but it means the "bias" is shrunk too.

## Interview follow-ups
- Why does the hinge loss give sparse dual solutions (support vectors) while logistic loss doesn't?
- How would you add a kernel? (Keep the coefficients $\alpha_i$ instead of $w$: the same update increments $\alpha_i$, and each margin costs a sum over past support vectors.)
"""
