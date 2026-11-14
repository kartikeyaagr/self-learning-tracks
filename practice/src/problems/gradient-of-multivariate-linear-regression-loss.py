import numpy as np

META = dict(title='Gradient of multi-output ridge regression', tags=['Regression', 'Calculus'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:08.03', 'mlmaths:09.34'],
            related=['hessian-matrix-and-convexity-of-linear-regressio', 'gradient-of-matrix-valued-function',
                     'subgradients-for-non-smooth-functions'])

STATEMENT = r"""
A linear model with $k$ outputs predicts $\hat Y = XW + \mathbf 1 b^\top$ for inputs $X \in \mathbb R^{n\times d}$, weights $W \in \mathbb R^{d\times k}$ and bias $b \in \mathbb R^{k}$. Train it with a halved mean squared error plus an L2 penalty on the weights only:

$$L(W, b) = \frac{1}{2n}\bigl\lVert XW + \mathbf 1 b^\top - Y\bigr\rVert_F^2 + \frac{\lambda}{2}\lVert W\rVert_F^2$$

Write `ridge_loss_grad(X, Y, W, b, lam)` that returns `(loss, dW, db)`.

## Shapes and types
- `X`: `(n, d)`, `Y`: `(n, k)`, `W`: `(d, k)`, `b`: `(k,)`, `lam` $\ge 0$; all floats, $n \ge 1$.
- Return `loss` as a Python `float`, `dW` with shape `(d, k)` and `db` with shape `(k,)`.
- The bias is **not** penalised.
- Fully vectorised: matrix products only, no loops.
"""

STARTER = '''import numpy as np

def ridge_loss_grad(X, Y, W, b, lam):
    """Return (loss, dL/dW, dL/db) for multi-output ridge regression."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ridge_loss_grad(X, Y, W, b, lam):
    X, Y, W, b = (np.asarray(a, dtype=float) for a in (X, Y, W, b))
    n = len(X)
    R = X @ W + b - Y                                   # residuals, (n, k); b broadcasts over rows
    loss = 0.5 * np.sum(R ** 2) / n + 0.5 * lam * np.sum(W ** 2)
    dW = X.T @ R / n + lam * W
    db = R.sum(axis=0) / n
    return float(loss), dW, db
'''

HINTS = [
    'Work with the residual matrix $R = XW + \\mathbf 1 b^\\top - Y$. The loss is $\\frac1{2n}\\sum R_{ij}^2$ plus the penalty.',
    'Perturb $W$ by $\\Delta$: $R$ changes by $X\\Delta$, and $\\frac{1}{2n}\\lVert R\\rVert^2$ changes by $\\frac1n\\langle R, X\\Delta\\rangle = \\frac1n\\langle X^\\top R, \\Delta\\rangle$. Read the gradient off the inner product.',
    '`dW = X.T @ R / n + lam * W` and `db = R.sum(axis=0) / n`. The bias is added to every row, so its gradient sums the residuals over rows.',
]

_r = np.random.default_rng(14)
TESTS = [
    (np.array([[1.0], [2.0]]), np.array([[1.0], [3.0]]), np.array([[1.0]]), np.array([0.0]), 0.0),
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]),
     np.zeros((2, 2)), np.array([0.5, -0.5]), 0.1),
    (np.array([[2.0, -1.0, 0.5]]), np.array([[4.0, 1.0]]), np.ones((3, 2)), np.zeros(2), 1.0),
    (_r.normal(size=(20, 4)), _r.normal(size=(20, 3)), np.zeros((4, 3)), np.zeros(3), 0.0),
    (np.array([[1e3, -2e3], [3e3, 1e3]]), np.array([[1.0], [2.0]]), np.array([[1e-3], [2e-3]]), np.array([0.5]), 1e-2),
    (_r.normal(size=(30, 5)), _r.normal(size=(30, 2)), _r.normal(size=(5, 2)), _r.normal(size=2), 0.5),
    (_r.normal(size=(8, 6)), _r.normal(size=(8, 4)), _r.normal(size=(6, 4)), _r.normal(size=4), 3.0),
]

EXPLANATION = r"""
## Deriving it with differentials
Perturb $W \to W + \Delta$. The residual becomes $R + X\Delta$, so
$$\tfrac1{2n}\lVert R + X\Delta\rVert_F^2 = \tfrac1{2n}\lVert R\rVert_F^2 + \tfrac1n \operatorname{tr}(R^\top X\Delta) + O(\lVert\Delta\rVert^2)$$
The linear term is $\langle \tfrac1n X^\top R, \Delta\rangle$, so $\nabla_W = \tfrac1n X^\top R$, plus $\lambda W$ from the penalty. The bias enters every row identically, so $\nabla_b = \tfrac1n R^\top\mathbf 1$.

## Shapes as a check
$\nabla_W$ must have $W$'s shape $(d, k)$: the only way to get that from $X$ $(n, d)$ and $R$ $(n, k)$ is $X^\top R$. Shape-matching catches most transposition bugs before you run anything.

## Setting it to zero
Without the bias, $\nabla_W = 0$ gives $(X^\top X + n\lambda I)W = X^\top Y$: one ridge solve per output column, sharing the same matrix. The $n\lambda$ (not $\lambda$) comes from using the **mean** squared error.

## Complexity
$O(ndk)$ for the two matrix products.

## Common bugs
- `R @ X` or `X @ R.T`: shape errors, or silent wrong answers when $n = d$ or $n = k$.
- Penalising the bias, which shrinks predictions towards 0 instead of towards the mean target.
- Forgetting the factor $\tfrac12$ in one place but not the other, giving gradients off by 2.

## Interview follow-ups
- Why do people centre $X$ and $Y$ and drop the bias entirely? Show the optimal $b$ is $\bar y - W^\top \bar x$.
- How does the gradient change if each output column has its own $\lambda_j$?
"""
