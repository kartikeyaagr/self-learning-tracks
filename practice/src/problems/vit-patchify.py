import numpy as np

META = dict(title='ViT patch embedding with a class token', tags=['Computer Vision', 'Transformers'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:03.04'], related=['reshaping-arrays', '2d-sinusoidal-positional-embedding', 'vit-multi-head-self-attention'])

STATEMENT = r"""
A Vision Transformer turns an image into a token sequence in three steps: cut it into non-overlapping `p x p` patches, project each flattened patch to the model width `D` with a shared linear layer, and prepend a learnable **class token**. A learned position embedding is then added to every token.

Write `patch_embed(images, p, w, b, cls, pos)`.

## Input
- `images`: float array `(B, H, W, C)`, channels last, with `H` and `W` multiples of `p`. Let `N = (H/p) * (W/p)`.
- `w`: `(p*p*C, D)` projection weights, `b`: `(D,)` bias.
- `cls`: `(D,)` class token. `pos`: `(N + 1, D)` position embeddings (row 0 belongs to the class token).

## Patch order and flattening
- Patches are numbered **row-major over the patch grid**: left to right along the top strip of patches, then the next strip.
- Each patch is flattened in `(row inside patch, column inside patch, channel)` order, i.e. `patch.reshape(-1)` of its `(p, p, C)` block.

## Output
A `float64` array `(B, N + 1, D)`: token 0 is `cls + pos[0]` for every image, and token `k >= 1` is `flat_patch_{k-1} @ w + b + pos[k]`.
"""

STARTER = '''import numpy as np

def patch_embed(images, p, w, b, cls, pos):
    """images: (B, H, W, C). Return (B, N + 1, D) tokens: class token first, then projected patches, plus pos."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def patch_embed(images, p, w, b, cls, pos):
    x = np.asarray(images, dtype=np.float64)
    B, H, W, C = x.shape
    gh, gw = H // p, W // p
    # (B, gh, p, gw, p, C) -> (B, gh, gw, p, p, C): bring the two patch-grid axes together, then flatten.
    patches = x.reshape(B, gh, p, gw, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, gh * gw, p * p * C)
    tokens = patches @ w + b                                   # (B, N, D)
    cls_tok = np.broadcast_to(cls, (B, 1, len(cls)))
    return np.concatenate([cls_tok, tokens], axis=1) + pos     # pos broadcasts over the batch
'''

HINTS = [
    'A reshape alone cannot cut patches: you have to split `H` into `(H/p, p)` and `W` into `(W/p, p)` and then move the axes so both grid axes come before both in-patch axes.',
    '`x.reshape(B, gh, p, gw, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, N, p*p*C)` gives row-major patches flattened as `(row, col, channel)`.',
    'Project with `patches @ w + b`, prepend `cls` broadcast to `(B, 1, D)` with `np.concatenate(axis=1)`, then add `pos` (shape `(N+1, D)` broadcasts over `B`).',
]

_r = np.random.default_rng(35)
_img = np.arange(16, dtype=float).reshape(1, 4, 4, 1)
TESTS = [
    (_img, 2, np.eye(4), np.zeros(4), np.zeros(4), np.zeros((5, 4))),
    (_img, 2, np.ones((4, 1)), np.array([0.5]), np.array([-1.0]), np.arange(5, dtype=float).reshape(5, 1)),
    (np.arange(24, dtype=float).reshape(1, 2, 4, 3), 2, np.eye(12), np.zeros(12), np.ones(12), np.zeros((3, 12))),
    (_r.normal(size=(2, 6, 4, 3)), 2, _r.normal(size=(12, 5)), _r.normal(size=5), _r.normal(size=5), _r.normal(size=(7, 5))),
    (_r.normal(size=(3, 8, 8, 2)), 4, _r.normal(size=(32, 6)), _r.normal(size=6), _r.normal(size=6), _r.normal(size=(5, 6))),
    (_r.normal(size=(1, 3, 3, 1)), 3, _r.normal(size=(9, 2)), np.zeros(2), _r.normal(size=2), _r.normal(size=(2, 2))),
]

EXPLANATION = r"""
## Patchify plus linear equals a strided convolution
Projecting each flattened patch with a shared matrix is exactly a convolution with kernel size $p$ and stride $p$ whose `D` filters are the columns of `w` (reshaped to `p x p x C`). That's how ViTs are implemented in practice: `Conv2d(C, D, kernel_size=p, stride=p)` followed by a flatten.

## Why the class token
The class token attends to every patch through the encoder and its final state is fed to the classifier head. It gives the model a dedicated slot for a global summary. Many newer models drop it and average the patch tokens instead.

## Complexity
$O(B N p^2 C D)$ for the projection. The sequence length $N = HW/p^2$ is what drives attention's $O(N^2)$ cost: halving $p$ quadruples $N$.

## Common bugs
- `x.reshape(B, N, -1)` without the transpose: each "patch" becomes a strip of pixels.
- Flattening in `(channel, row, col)` order (the PyTorch conv-weight layout) while the weights expect `(row, col, channel)`. Either convention works, but weights and patches must agree.
- Adding `pos` before prepending the class token, which leaves the shapes mismatched by one.

## Interview follow-ups
- The model was trained at 224 px with $p = 16$; you fine-tune at 384 px. What do you do with `pos`? (Reshape the patch part to its 14x14 grid, interpolate it to 24x24, keep the class row.)
- How do hybrid ViTs change this step? (They patchify a CNN feature map instead of raw pixels.)
"""
