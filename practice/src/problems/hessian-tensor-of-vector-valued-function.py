import numpy as np

META = dict(title='Hessian tensor of a gated linear unit', tags=['Calculus', 'Neural Networks'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:09.04', 'mlmaths:09.03'],
            related=['jacobian-of-neural-network-layer', 'second-order-partial-derivatives', 'taylor-approximation-of-loss-landscape'])

STATEMENT = r"""
A gated linear unit multiplies a linear "value" path by a sigmoid "gate" path:

$$F_i(x) = a_i\, s_i, \qquad a = Wx, \qquad s = \sigma(Vx)$$

with $W, V \in \mathbb R^{m\times n}$, so $F: \mathbb R^n \to \mathbb R^m$. Its second derivative is a third-order tensor: one Hessian matrix per output,

$$H_{ijk} = \frac{\partial^2 F_i}{\partial x_j\, \partial x_k}$$

Write `glu_derivatives(W, V, x)` that returns the tuple `(F, J, H)` with shapes `(m,)`, `(m, n)` and `(m, n, n)`, where $J_{ij} = \partial F_i / \partial x_j$.

## Requirements
- `W`, `V`: float arrays `(m, n)`; `x`: `(n,)`.
- Use $\sigma'(t) = \sigma(1-\sigma)$ and $\sigma''(t) = \sigma(1-\sigma)(1-2\sigma)$.
- Vectorise with broadcasting or `np.einsum`: no Python loops.
- Each $H_{i}$ must come out exactly symmetric (build it from symmetric pieces).
"""

STARTER = '''import numpy as np

def glu_derivatives(W, V, x):
    """F = (W x) * sigmoid(V x). Return (F (m,), J (m, n), H (m, n, n))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def glu_derivatives(W, V, x):
    W, V, x = (np.asarray(t, dtype=float) for t in (W, V, x))
    a = W @ x
    s = 1 / (1 + np.exp(-(V @ x)))
    s1 = s * (1 - s)                 # sigma'
    s2 = s1 * (1 - 2 * s)            # sigma''
    F = a * s
    J = s[:, None] * W + (a * s1)[:, None] * V                   # product rule, row by row
    WV = np.einsum('ij,ik->ijk', W, V)
    H = (s1[:, None, None] * (WV + WV.transpose(0, 2, 1))
         + (a * s2)[:, None, None] * np.einsum('ij,ik->ijk', V, V))
    return F, J, H
'''

HINTS = [
    'Each output depends on $x$ only through two scalars, $a_i = w_i^\\top x$ and $t_i = v_i^\\top x$, where $w_i, v_i$ are rows. Differentiate $a_i\\,\\sigma(t_i)$ with the product rule.',
    'Gradient of output $i$: $\\sigma(t_i)\\, w_i + a_i\\sigma\'(t_i)\\, v_i$. Differentiate again: the first term gives $\\sigma\'(t_i)\\, w_i v_i^\\top$, the second gives $\\sigma\'(t_i)\\, v_i w_i^\\top + a_i\\sigma\'\'(t_i)\\, v_i v_i^\\top$.',
    'Batch the outer products with `np.einsum("ij,ik->ijk", W, V)`, add its transpose over the last two axes, and scale each slice with `[:, None, None]`.',
]

_r = np.random.default_rng(41)
TESTS = [
    (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), np.array([2.0, 0.0])),
    (np.array([[1.0, 1.0], [0.0, 2.0]]), np.zeros((2, 2)), np.array([1.0, -1.0])),
    (np.array([[1.0, -1.0, 0.5]]), np.array([[2.0, 0.0, 1.0]]), np.zeros(3)),
    (np.array([[1.0, 2.0]]), np.array([[10.0, 10.0]]), np.array([3.0, 3.0])),
    (_r.normal(size=(3, 4)), _r.normal(size=(3, 4)), _r.normal(size=4)),
    (_r.normal(size=(2, 3)) * 2, _r.normal(size=(2, 3)) * 2, _r.normal(size=3)),
]

EXPLANATION = r"""
## Reduce to scalars
Output $i$ is $g(a_i, t_i) = a_i\,\sigma(t_i)$ with $a_i = w_i^\top x$ and $t_i = v_i^\top x$ both linear in $x$. For any function of linear forms, the Hessian is a combination of outer products of the rows:

$$H_i = g_{aa}\, w_iw_i^\top + g_{at}\,(w_iv_i^\top + v_iw_i^\top) + g_{tt}\, v_iv_i^\top$$

Here $g_{aa} = 0$ (linear in $a$), $g_{at} = \sigma'(t_i)$ and $g_{tt} = a_i\sigma''(t_i)$. Each $H_i$ has rank at most 2.

## Why the tensor matters
A second-order Taylor expansion of a vector-valued function needs one Hessian per output: $F(x+\delta) \approx F(x) + J\delta + \tfrac12\,\delta^\top H_i \delta$ for each $i$. Gauss-Newton methods drop exactly this tensor and keep $J^\top J$; this problem shows what's being dropped. Note that with $V = 0$ (test 2) the gate is constant at $\tfrac12$, $F$ is linear and $H$ vanishes.

## Complexity
$O(mn^2)$ to fill the tensor, the size of the output.

## Common bugs
- Forgetting the cross term appears twice (once from each factor in the product rule), so $H_i$ isn't symmetric.
- Writing $\sigma''$ as $\sigma(1-\sigma)^2$.
- Swapping $J$'s axes: rows are outputs.

## Interview follow-ups
- Contract the tensor with a direction: compute $\sum_{jk} H_{ijk}\delta_j\delta_k$ for all $i$ without forming $H$.
- SwiGLU uses $\mathrm{silu}(Vx) \odot Wx$. Which of $g_{aa}, g_{at}, g_{tt}$ change?
"""
