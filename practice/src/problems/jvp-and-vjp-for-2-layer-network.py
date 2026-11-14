import numpy as np

META = dict(title='JVPs and VJPs through a two-layer MLP', tags=['Calculus', 'Neural Networks'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:01.09', 'mlmaths:09.28'],
            related=['jacobian-of-neural-network-layer', 'backpropagation-through-computation-graph',
                     'forward-mode-autodiff-with-dual-numbers'])

STATEMENT = r"""
Autodiff systems never build Jacobians. They push vectors through them: forward mode computes a **Jacobian-vector product** (JVP) $J v$, reverse mode computes a **vector-Jacobian product** (VJP) $u^\top J$.

Consider the two-layer MLP applied row by row to a batch:

$$h = \mathrm{relu}(W_1 x + b_1), \qquad y = W_2 h + b_2$$

with $W_1 \in \mathbb R^{H\times n}$, $W_2 \in \mathbb R^{m \times H}$. Write `jvp_vjp(W1, b1, W2, b2, X, V, U)` that returns a tuple `(Y, jvp, vjp_x, vjp_W1, vjp_W2)`:

| Output | Shape | Meaning |
|---|---|---|
| `Y` | `(B, m)` | the forward outputs |
| `jvp` | `(B, m)` | row $k$: $J_k v_k$, where $J_k = \partial y_k / \partial x_k$ and $v_k$ = `V[k]` |
| `vjp_x` | `(B, n)` | row $k$: $u_k^\top J_k$, with $u_k$ = `U[k]` |
| `vjp_W1` | `(H, n)` | $\sum_k u_k^\top\, \partial y_k / \partial W_1$, summed over the batch |
| `vjp_W2` | `(m, H)` | $\sum_k u_k^\top\, \partial y_k / \partial W_2$, summed over the batch |

## Requirements
- `X`, `V`: `(B, n)`; `U`: `(B, m)`; biases are 1-D. All floats.
- Use $\mathrm{relu}'(0) = 0$.
- Never materialise a Jacobian: no array with a dimension of size $m \cdot n$ or more per sample. Matrix products only, no loops.
"""

STARTER = '''import numpy as np

def jvp_vjp(W1, b1, W2, b2, X, V, U):
    """Return (Y, J V, U^T J wrt x, U^T J wrt W1 (batch-summed), U^T J wrt W2 (batch-summed))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def jvp_vjp(W1, b1, W2, b2, X, V, U):
    W1, b1, W2, b2, X, V, U = (np.asarray(a, dtype=float) for a in (W1, b1, W2, b2, X, V, U))
    Z = X @ W1.T + b1
    mask = (Z > 0).astype(float)              # relu'(z), with relu'(0) = 0
    Hid = Z * mask
    Y = Hid @ W2.T + b2
    # forward mode: push the tangent V through each linear map and the relu mask
    jvp = (V @ W1.T * mask) @ W2.T
    # reverse mode: pull the cotangent U back through the same maps, transposed
    G = U @ W2 * mask                         # dL/dZ for "loss" = sum_k u_k . y_k
    return Y, jvp, G @ W1, G.T @ X, U.T @ Hid
'''

HINTS = [
    'The per-sample Jacobian is $J = W_2\\,\\mathrm{diag}(m)\\,W_1$, where $m$ is the 0/1 relu mask. A JVP evaluates $J v$ right to left, a VJP evaluates $u^\\top J$ left to right, and neither needs $J$ itself.',
    'For a batch, rows are samples, so "$W v$" becomes `V @ W.T`. JVP: `((V @ W1.T) * mask) @ W2.T`. VJP: `G = (U @ W2) * mask`, then `G @ W1`.',
    'The parameter VJPs are the usual backprop formulas, because a VJP with $u$ is exactly the gradient of $\\sum_k u_k\\cdot y_k$: `vjp_W2 = U.T @ Hid` and `vjp_W1 = G.T @ X`.',
]

_r = np.random.default_rng(29)
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros(2), np.array([[1.0, 1.0]]), np.zeros(1),
     np.array([[1.0, -1.0]]), np.array([[1.0, 1.0]]), np.array([[2.0]])),
    (np.array([[1.0, 2.0], [-1.0, 1.0], [0.5, 0.0]]), np.array([0.0, 0.5, -1.0]), np.array([[1.0, -1.0, 2.0], [0.0, 1.0, 1.0]]),
     np.array([0.1, -0.1]), np.array([[1.0, 1.0], [2.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])),
    (np.array([[1.0, -1.0]]), np.array([0.0]), np.array([[3.0]]), np.array([1.0]),
     np.array([[2.0, 2.0]]), np.array([[1.0, 0.0]]), np.array([[1.0]])),
    (-np.ones((3, 2)), np.full(3, -1.0), np.ones((2, 3)), np.zeros(2),
     np.array([[1.0, 2.0]]), np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]])),
    (_r.normal(size=(5, 4)), _r.normal(size=5), _r.normal(size=(3, 5)), _r.normal(size=3),
     _r.normal(size=(6, 4)), _r.normal(size=(6, 4)), _r.normal(size=(6, 3))),
    (_r.normal(size=(8, 3)), _r.normal(size=8), _r.normal(size=(2, 8)), _r.normal(size=2),
     _r.normal(size=(10, 3)), _r.normal(size=(10, 3)), _r.normal(size=(10, 2))),
]

EXPLANATION = r"""
## The Jacobian you never build
Per sample, $J = W_2\,\mathrm{diag}(\mathbb 1[z > 0])\,W_1$, an $m\times n$ matrix. Forming it costs $O(mHn)$; a JVP evaluated right to left costs $O(Hn + mH)$, the same as a forward pass. The VJP is the same product read left to right.

## Forward vs reverse mode
- **JVP** (forward mode) gives one column-combination of $J$ per pass: the derivative of all outputs along one input direction. Good when inputs are few.
- **VJP** (reverse mode) gives one row-combination per pass: the gradient of one scalar with respect to all inputs. Training has one scalar loss and millions of parameters, which is why backprop is reverse mode.

The parameter VJPs are exactly backprop: with $u_k = \partial L/\partial y_k$, `vjp_W1` and `vjp_W2` are $\partial L/\partial W_1$ and $\partial L/\partial W_2$.

## A useful identity to test with
For any $u$ and $v$, $u^\top (Jv) = (u^\top J)\, v$. Checking `np.sum(U * jvp)` against `np.sum(vjp_x * V)` catches most bugs without knowing the answer.

## Common bugs
- Applying the mask to $W_1$'s input instead of its output.
- Using `>=` for the mask, which gives $\mathrm{relu}'(0) = 1$.
- Forgetting that parameter VJPs **sum** over the batch while input VJPs stay per-sample.

## Interview follow-ups
- How many JVPs, and how many VJPs, would you need to build the full Jacobian? When is each cheaper?
- How do you compute a Hessian-vector product with one JVP and one VJP (forward-over-reverse)?
"""
