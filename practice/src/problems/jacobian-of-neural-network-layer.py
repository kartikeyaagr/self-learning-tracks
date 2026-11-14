import numpy as np

META = dict(title='Every Jacobian of a tanh layer', tags=['Calculus', 'Neural Networks'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:09.28', 'air-fanout-v1:01.09'],
            related=['softmax-jacobian', 'jvp-and-vjp-for-2-layer-network', 'hessian-tensor-of-vector-valued-function'])

STATEMENT = r"""
A dense layer with tanh activation maps each input row $x \in \mathbb R^{n}$ to $y = \tanh(Wx + b) \in \mathbb R^{m}$. Its output depends on three things, and backprop needs the derivative with respect to each.

Write `tanh_layer_jacobians(W, b, X)` that, for a batch $X \in \mathbb R^{B \times n}$, returns the tuple `(Jx, JW, Jb)` of per-sample Jacobians:

| Output | Shape | Entry |
|---|---|---|
| `Jx` | `(B, m, n)` | `Jx[k, i, j]` $= \partial y_{k,i} / \partial x_{k,j}$ |
| `JW` | `(B, m, m, n)` | `JW[k, i, p, q]` $= \partial y_{k,i} / \partial W_{p,q}$ |
| `Jb` | `(B, m, m)` | `Jb[k, i, p]` $= \partial y_{k,i} / \partial b_{p}$ |

## Requirements
- `W`: `(m, n)`, `b`: `(m,)`, `X`: `(B, n)`; all floats.
- Fully vectorised with broadcasting or `np.einsum`: no Python loops at all.
- Derive the entries yourself. Most of `JW` is zero; part of the exercise is seeing which entries are not.
"""

STARTER = '''import numpy as np

def tanh_layer_jacobians(W, b, X):
    """Per-sample Jacobians of y = tanh(W x + b): (Jx (B,m,n), JW (B,m,m,n), Jb (B,m,m))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def tanh_layer_jacobians(W, b, X):
    W, b, X = (np.asarray(a, dtype=float) for a in (W, b, X))
    m = len(b)
    Y = np.tanh(X @ W.T + b)          # (B, m)
    D = 1 - Y ** 2                    # tanh'(z), (B, m)
    I = np.eye(m)
    Jx = D[:, :, None] * W[None]                               # diag(D_k) @ W
    Jb = D[:, :, None] * I[None]                               # diag(D_k)
    JW = np.einsum('ki,ip,kq->kipq', D, I, X)                  # D_ki * [i == p] * x_kq
    return Jx, JW, Jb
'''

HINTS = [
    'Write $z = Wx + b$ and $y_i = \\tanh(z_i)$. Each $y_i$ depends on $z_i$ only, so every Jacobian is $\\mathrm{diag}(1 - y^2)$ times the Jacobian of $z$.',
    '$\\partial z_i/\\partial x_j = W_{ij}$, $\\partial z_i/\\partial b_p = [i = p]$, and $\\partial z_i / \\partial W_{pq} = [i = p]\\, x_q$: output $i$ only depends on row $i$ of $W$.',
    'Let `D = 1 - Y**2` with shape `(B, m)`. Then `Jx = D[:, :, None] * W`, `Jb = D[:, :, None] * np.eye(m)`, and `JW = np.einsum("ki,ip,kq->kipq", D, np.eye(m), X)`.',
]

_r = np.random.default_rng(23)
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 2.0]]), np.zeros(2), np.array([[0.0, 0.0]])),
    (np.array([[0.5, -1.0, 2.0]]), np.array([0.1]), np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])),
    (np.zeros((3, 2)), np.zeros(3), np.array([[5.0, -5.0]])),
    (np.array([[10.0, 10.0]]), np.array([0.0]), np.array([[3.0, 3.0]])),
    (_r.normal(size=(3, 4)), _r.normal(size=3), _r.normal(size=(5, 4))),
    (_r.normal(size=(2, 2)) * 3, _r.normal(size=2), _r.normal(size=(3, 2))),
]

EXPLANATION = r"""
## One structure, three Jacobians
With $D = \mathrm{diag}(1 - y^2)$, the chain rule gives

$$\frac{\partial y}{\partial x} = D\,W, \qquad \frac{\partial y}{\partial b} = D, \qquad \frac{\partial y_i}{\partial W_{pq}} = D_{ii}\,[i = p]\,x_q$$

The last one is a 3-index object per sample: it's block-sparse, because output $i$ only sees row $i$ of $W$. Of its $m^2 n$ entries only $mn$ are nonzero.

## Why nobody builds JW in practice
For a 4096-by-4096 layer, `JW` per sample has $4096^3 \approx 7\times10^{10}$ entries. Backprop never forms it. It only needs the **vector-Jacobian product** with the upstream gradient $\bar y$: $\sum_i \bar y_i\, \partial y_i/\partial W_{pq} = (\bar y \odot (1-y^2))_p\, x_q$, which is an outer product. That's the familiar `dW = delta.T @ X`. Building the full tensor once on a tiny layer is how you convince yourself the outer-product formula is right.

## Complexity
`Jx` and `Jb` cost $O(Bmn)$ and $O(Bm^2)$; `JW` costs $O(Bm^2n)$ memory, mostly zeros.

## Common bugs
- `W * D` broadcasting along the wrong axis: $D$ must scale the **rows** of $W$ (outputs), not the columns.
- Using $1 - z^2$ instead of $1 - \tanh(z)^2$.
- Getting `JW[k, i, p, q]` nonzero for $i \ne p$, which treats every output as depending on every weight row.

## Interview follow-ups
- Contract `JW` with an upstream gradient using `np.einsum` and show it equals `np.outer(gbar * (1 - y**2), x)`.
- How would the Jacobian with respect to $x$ change for softmax instead of tanh? (It's no longer diagonal-times-$W$.)
"""
