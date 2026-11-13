import numpy as np

META = dict(title='Batched outer products', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:04.04', 'air-fanout-v1:01.03'], related=['dot-product', 'matrix-rank', 'svd-components'])

STATEMENT = r"""
The outer product of $u \in \mathbb{R}^m$ and $v \in \mathbb{R}^n$ is the $m \times n$ matrix

$$(u v^\top)_{ij} = u_i\, v_j$$

Every such matrix has rank at most 1, and sums of them build every other matrix (that's what the SVD does). They also appear in backprop: the gradient of a linear layer's weights for one sample is the outer product of the upstream gradient and the input.

Write `outer(u, v)` that computes outer products for a **batch** of vector pairs.

## Inputs
- `u`: float array of shape `(..., m)`, `v`: float array of shape `(..., n)`, with **identical** leading shapes (possibly none).

## Output
- Float array of shape `(..., m, n)` where `out[..., i, j] = u[..., i] * v[..., j]`.
- For 1-D inputs this is the ordinary `(m, n)` outer product.

## Note
`np.outer` flattens its inputs, so it gives the wrong answer for batches. Use broadcasting.
"""

STARTER = '''import numpy as np

def outer(u, v):
    """u: (..., m), v: (..., n), same leading shape. Return (..., m, n) outer products."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def outer(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    # (..., m, 1) * (..., 1, n) -> (..., m, n)
    return u[..., :, None] * v[..., None, :]
'''

HINTS = [
    'Entry `(i, j)` only needs `u[i]` and `v[j]`. Can you shape `u` as a column and `v` as a row and let broadcasting fill the grid?',
    'Add a trailing axis to `u` and a second-to-last axis to `v`. The ellipsis `...` keeps any batch axes in front untouched.',
    '`u[..., :, None] * v[..., None, :]`. Or `np.einsum("...i,...j->...ij", u, v)`.',
]

_r = np.random.default_rng(21)
TESTS = [
    (np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])),
    (np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([[1.0, -1.0], [3.0, 4.0]])),
    (np.array([2.0]), np.array([-3.0])),
    (np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0])),
    (np.array([1e154, -1e154]), np.array([1e154, 2.0])),
    (_r.normal(size=(3, 4)), _r.normal(size=(3, 2))),
    (_r.normal(size=(2, 2, 3)), _r.normal(size=(2, 2, 5))),
]

EXPLANATION = r"""
## Why broadcasting works
`u[..., :, None]` has shape `(..., m, 1)` and `v[..., None, :]` has shape `(..., 1, n)`. Multiplying stretches both size-1 axes, so position `[..., i, j]` holds `u[..., i] * v[..., j]`. The leading axes are equal, so they pair up one-to-one: batch item $b$ only ever meets batch item $b$.

## Rank one
Every row of $u v^\top$ is a multiple of $v^\top$ (row $i$ is $u_i v^\top$), so the matrix has rank 1 unless $u$ or $v$ is zero. A product $uv^\top x = u\,(v\cdot x)$ costs $O(m + n)$ without forming the matrix, which is why low-rank updates (LoRA, Sherman–Morrison) are cheap.

## Complexity
$O(Bmn)$ time and memory for $B$ pairs; that's the size of the output, so it's optimal.

## Common bugs
- `np.outer(u, v)` on batches: it flattens to `(B*m, B*n)` and mixes items across the batch.
- `u[:, None] * v[None, :]` on 2-D batches: that broadcasts the **batch** axis against the vector axis.
- Confusing outer and inner products: `u @ v` is a scalar, `u[:, None] @ v[None, :]` is the matrix.

## Interview follow-ups
- What is $\operatorname{tr}(u v^\top)$? ($u \cdot v$, the inner product.)
- For a linear layer $y = Wx$ over a batch, why is $\partial L / \partial W = \sum_b g_b x_b^\top$, and how do you compute it in one call? (`G.T @ X`: a sum of outer products is a matrix product.)
"""
