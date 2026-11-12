import numpy as np

META = dict(title='Gradient of a sandwiched least-squares loss', tags=['Calculus', 'Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:09.05'],
            related=['gradient-of-multivariate-linear-regression-loss', 'jacobian-of-neural-network-layer'])

STATEMENT = r"""
Many matrix problems (low-rank adapters, bilinear models, matrix completion) optimise a matrix $W$ that sits between two fixed matrices. Take

$$f(W) = \tfrac12\,\lVert A\,W\,B - C\rVert_F^2$$

with $A \in \mathbb R^{p\times d}$, $W \in \mathbb R^{d \times k}$, $B \in \mathbb R^{k \times q}$ and $C \in \mathbb R^{p \times q}$. The gradient $\nabla_W f$ is the matrix of the same shape as $W$ with entries $\partial f / \partial W_{ij}$.

Write `sandwich_grad(A, W, B, C)` that returns the tuple `(f, G)`: the loss as a Python `float` and the gradient `G` with shape `(d, k)`.

## Requirements
- All inputs are float arrays with compatible shapes.
- Derive $G$ in closed form and compute it with matrix products (no loops, no finite differences).
"""

STARTER = '''import numpy as np

def sandwich_grad(A, W, B, C):
    """Return (0.5 * ||A W B - C||_F^2, gradient with respect to W)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sandwich_grad(A, W, B, C):
    A, W, B, C = (np.asarray(a, dtype=float) for a in (A, W, B, C))
    R = A @ W @ B - C                    # residual, (p, q)
    return 0.5 * float(np.sum(R ** 2)), A.T @ R @ B.T
'''

HINTS = [
    'Perturb $W$ by a small $\\Delta$. The residual $R = AWB - C$ changes by $A\\Delta B$, and the loss changes by $\\langle R, A\\Delta B\\rangle$ to first order.',
    'Move $A$ and $B$ to the other side of the Frobenius inner product: $\\langle R, A\\Delta B\\rangle = \\operatorname{tr}(R^\\top A \\Delta B) = \\langle A^\\top R B^\\top, \\Delta\\rangle$.',
    '$G = A^\\top (AWB - C) B^\\top$. Check the shapes: $(d\\times p)(p\\times q)(q\\times k) = d \\times k$.',
]

_r = np.random.default_rng(37)
TESTS = [
    (np.eye(2), np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2), np.zeros((2, 2))),
    (np.array([[1.0, 2.0]]), np.array([[1.0], [1.0]]), np.array([[2.0, -1.0]]), np.array([[1.0, 0.0]])),
    (np.array([[2.0, 0.0, 1.0]]), np.zeros((3, 2)), np.array([[1.0], [1.0]]), np.array([[5.0]])),
    (_r.normal(size=(4, 3)), _r.normal(size=(3, 2)), _r.normal(size=(2, 5)), _r.normal(size=(4, 5))),
    (_r.normal(size=(6, 2)), _r.normal(size=(2, 1)), _r.normal(size=(1, 3)), _r.normal(size=(6, 3))),
    (np.array([[1e3, 0.0], [0.0, 1e-3]]), np.array([[1.0, 1.0], [1.0, 1.0]]), np.eye(2), np.ones((2, 2))),
]

EXPLANATION = r"""
## The differential trick
$f(W+\Delta) - f(W) = \langle R, A\Delta B\rangle + O(\lVert\Delta\rVert^2)$. Using the cyclic property of the trace, $\operatorname{tr}(R^\top A\Delta B) = \operatorname{tr}(B R^\top A \Delta) = \langle A^\top R B^\top, \Delta\rangle$. Whatever multiplies $\Delta$ in that inner product is the gradient. This one recipe handles almost every matrix-calculus gradient in ML.

## Sanity checks
- Shapes: only $A^\top R B^\top$ produces a $d\times k$ matrix from these factors.
- Special cases: with $A = I$, $B = I$ it's $W - C$, and with $B = I$ it's the least-squares gradient $A^\top(AW - C)$.

## Complexity
Two matrix products for the forward pass and two for the gradient. Pick the multiplication order to keep intermediates small: $(A^\top R)B^\top$ costs $O(dpq + dqk)$.

## Common bugs
- $A R B$ or $A^\top R^\top B^\top$: they don't type-check, which is the hint to transpose differently.
- Forgetting the $\tfrac12$, giving $2\times$ the gradient.

## Interview follow-ups
- LoRA parametrises an update as $W = UV^\top$. Derive $\nabla_U$ and $\nabla_V$ of $\tfrac12\lVert X UV^\top - Y\rVert_F^2$.
- What's the gradient of $\log\det W$, and why does it show up in normalising flows?
"""
