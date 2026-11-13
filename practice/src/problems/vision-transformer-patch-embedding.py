import numpy as np

META = dict(title='ViT input: patches, projection, class token, positions', tags=['Transformers', 'Computer Vision', 'Embeddings'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:03.04', 'ieng:26.03'],
            related=['reshaping-arrays', '2d-convolution', 'transformer-encoder-block'])

STATEMENT = r"""
A Vision Transformer turns an image into a token sequence: cut it into non-overlapping $P \times P$ patches, project each flattened patch to width $D$, prepend a learned **class token**, and add learned position embeddings. The projection is usually stored as a convolution weight, since a $P \times P$ conv with stride $P$ computes exactly the same thing.

Write `vit_embed(images, W, b, cls, pos)`.

## Inputs
- `images`: float `(B, C, H, W_img)` with $H$ and $W_\text{img}$ multiples of $P$ (they need not be equal).
- `W`: float `(D, C, P, P)`, a conv-style projection weight (this fixes $P$). `b`: `(D,)`.
- `cls`: `(D,)` class token. `pos`: `(1 + N, D)` position embeddings, where $N = (H/P)(W_\text{img}/P)$; row 0 belongs to the class token.

## Computation
1. Patch $(r, c)$ of image $n$ is `images[n, :, r*P:(r+1)*P, c*P:(c+1)*P]`. Patches are ordered row-major over the grid: $k = r \cdot (W_\text{img}/P) + c$.
2. Token $k$: $e_k = \sum_{\text{ch}, u, v} W_{d,\text{ch},u,v}\,\text{patch}_k[\text{ch}, u, v] + b_d$ for each $d$, i.e. flatten the patch in `(C, P, P)` order and multiply by `W.reshape(D, -1).T`.
3. Sequence: `[cls, e_0, ..., e_{N-1}]` of shape `(1 + N, D)`, plus `pos` (the class token gets `pos[0]`, patch $k$ gets `pos[k + 1]`).

## Output
Float `(B, 1 + N, D)`. No loops over patches or pixels: reshapes, transposes and one matrix product.
"""

STARTER = '''import numpy as np

def vit_embed(images, W, b, cls, pos):
    """images: (B, C, H, W); W: (D, C, P, P). Return (B, 1 + N, D)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def vit_embed(images, W, b, cls, pos):
    x, W = np.asarray(images, dtype=float), np.asarray(W, dtype=float)
    B, C, H, Wi = x.shape
    D, _, P, _ = W.shape
    gh, gw = H // P, Wi // P
    # (B, C, gh, P, gw, P) -> (B, gh, gw, C, P, P): grid axes first, each patch in (C, P, P) order
    patches = x.reshape(B, C, gh, P, gw, P).transpose(0, 2, 4, 1, 3, 5).reshape(B, gh * gw, C * P * P)
    tokens = patches @ W.reshape(D, -1).T + b                         # (B, N, D)
    cls_tok = np.broadcast_to(np.asarray(cls, dtype=float), (B, 1, D))
    return np.concatenate([cls_tok, tokens], axis=1) + pos
'''

HINTS = [
    'Split each spatial axis into (grid index, offset inside the patch): `reshape(B, C, H/P, P, W/P, P)`.',
    'Transpose to `(B, H/P, W/P, C, P, P)` so the grid comes first (row-major patch order) and each patch is laid out as `(C, P, P)`, then reshape to `(B, N, C*P*P)`.',
    'Project with `patches @ W.reshape(D, -1).T + b`, which matches the conv weight\'s own `(C, P, P)` flatten order. Prepend `cls` broadcast to `(B, 1, D)` and add `pos`.',
]

_r = np.random.default_rng(103)
def _case(B, C, H, Wi, P, D):
    N = (H // P) * (Wi // P)
    return (_r.normal(size=(B, C, H, Wi)), _r.normal(size=(D, C, P, P)) / P, _r.normal(size=D) * 0.1,
            _r.normal(size=D), _r.normal(size=(1 + N, D)) * 0.1)
TESTS = [
    (np.arange(4.0).reshape(1, 1, 2, 2), np.array([[[[1.0]]], [[[-1.0]]]]), np.zeros(2), np.array([9.0, 9.0]), np.zeros((5, 2))),
    (np.arange(8.0).reshape(1, 2, 2, 2), np.ones((1, 2, 2, 2)), np.array([0.5]), np.zeros(1), np.array([[0.0], [1.0]])),
    _case(2, 3, 8, 8, 4, 5),
    _case(1, 3, 6, 9, 3, 4),
    _case(2, 1, 4, 8, 2, 3),
    _case(1, 2, 5, 5, 5, 6),
    _case(3, 4, 4, 4, 1, 2),
]

EXPLANATION = r"""
## Patchify = strided conv
A $P \times P$ convolution with stride $P$ visits each patch exactly once and takes a dot product between the patch and each of its $D$ filters: precisely a linear projection of the flattened patch. That's why ViT code stores the projection as `nn.Conv2d(C, D, kernel_size=P, stride=P)`, and why your flatten order must be `(C, P, P)` to reuse those weights.

## The reshape trick
`reshape(B, C, gh, P, gw, P)` exposes grid and in-patch coordinates as separate axes without moving data. The transpose to `(B, gh, gw, C, P, P)` then decides two orders at once: patches row-major over the grid, and each patch channel-first. Get either wrong and the output has the right shape and wrong values (test 4 has a non-square grid, which exposes swapped grid axes).

## Class token and positions
The class token is a learned vector that attends to all patches; its final state feeds the classifier head. Position embeddings are learned per grid cell, so fine-tuning at a higher resolution requires interpolating `pos` to the new grid (2D bilinear on rows 1..N, leaving row 0 alone).

## Complexity
$O(B N C P^2 D) = O(B\,HW\,C D)$, the same as one dense layer per pixel group; sequence length is $N = HW/P^2$, so halving $P$ quadruples attention cost.

## Common bugs
- `x.reshape(B, N, -1)` directly: that cuts along rows of pixels, not patches.
- Flattening patches as `(P, P, C)` while using the conv weight's `(C, P, P)` order.
- Adding `pos` before prepending the class token, which shifts every position by one.

## Interview follow-ups
- How many tokens does ViT-B/16 see for a 224x224 image, and for 384x384?
- Why do ViTs need more data (or stronger augmentation) than CNNs? What inductive biases are missing?
"""
