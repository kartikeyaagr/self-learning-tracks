import numpy as np

META = dict(title='Spatial self-attention block for feature maps', tags=['U-Net', 'Attention'], diff='Medium', runtime='numpy',
            lessons=['aieng:04.02', 'air-fanout-v1:06.02'],
            related=['scaled-dot-product-attention', 'softmax-rows', 'time-conditioned-residual-block'])

STATEMENT = r"""
Convolutions only see a neighbourhood. At the low-resolution levels of a diffusion U-Net, a self-attention block lets every pixel look at every other pixel, so distant parts of an image stay consistent. Each spatial position becomes a token whose features are its $C$ channels.

1. Normalise: $h = \mathrm{GN}(x)$, GroupNorm with `groups` contiguous channel groups, statistics over (group channels, $H$, $W$), population variance, $\epsilon = 10^{-5}$, no affine.
2. Flatten to tokens: $(B, C, H, W) \to (B, N, C)$ with $N = HW$, token $n = y \cdot W + x$ (row-major over pixels).
3. Single-head attention with row-vector weights: $Q = hW_q$, $K = hW_k$, $V = hW_v$, then
$$A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{C}}\right), \qquad O = (AV)\,W_o$$
with the softmax over keys (the last axis).
4. Reshape $O$ back to $(B, C, H, W)$ and return $x + O$.

Write `attn_block(x, Wq, Wk, Wv, Wo, groups)`.

## Inputs
- `x`: `(B, C, H, W)` float array.
- `Wq`, `Wk`, `Wv`, `Wo`: `(C, C)`, applied as `tokens @ W`.
- `groups`: positive integer dividing $C$.

## Output
`(B, C, H, W)` float array. Use a numerically stable softmax (subtract the row max).
"""

STARTER = '''import numpy as np

def attn_block(x, Wq, Wk, Wv, Wo, groups):
    """x: (B, C, H, W). Return x + self-attention over the H*W positions, same shape."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def group_norm(h, groups, eps=1e-5):
    B, C, H, W = h.shape
    g = h.reshape(B, groups, C // groups, H, W)
    g = (g - g.mean(axis=(2, 3, 4), keepdims=True)) / np.sqrt(g.var(axis=(2, 3, 4), keepdims=True) + eps)
    return g.reshape(B, C, H, W)

def attn_block(x, Wq, Wk, Wv, Wo, groups):
    x = np.asarray(x, dtype=float)
    B, C, H, W = x.shape
    tok = group_norm(x, groups).reshape(B, C, H * W).transpose(0, 2, 1)   # (B, N, C), row-major pixels
    q, k, v = tok @ Wq, tok @ Wk, tok @ Wv
    s = q @ k.transpose(0, 2, 1) / np.sqrt(C)                             # (B, N, N)
    a = np.exp(s - s.max(axis=-1, keepdims=True))
    a /= a.sum(axis=-1, keepdims=True)
    o = (a @ v) @ Wo                                                      # (B, N, C)
    return x + o.transpose(0, 2, 1).reshape(B, C, H, W)
'''

HINTS = [
    'Pixels are the tokens and channels are the features. `h.reshape(B, C, H*W).transpose(0, 2, 1)` gives `(B, N, C)` in row-major pixel order.',
    'With tokens in rows, every projection is a right-multiplication `tok @ W`, and the score matrix is `q @ k.transpose(0, 2, 1) / sqrt(C)`, shape `(B, N, N)`.',
    'After `(a @ v) @ Wo`, undo the flattening with `.transpose(0, 2, 1).reshape(B, C, H, W)`. Transposing first matters: reshaping `(B, N, C)` straight to `(B, C, H, W)` scrambles channels and pixels.',
]

_r = np.random.default_rng(17)
_W = lambda C: _r.normal(size=(C, C)) / np.sqrt(C)
_x1 = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]])
TESTS = [
    (_x1, np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2)), 1),
    (_x1, np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2), np.eye(2), 2),
    (_r.normal(size=(1, 2, 2, 2)), np.eye(2), np.eye(2), np.eye(2), np.eye(2), 1),
    (_r.normal(size=(2, 4, 3, 2)), _W(4), _W(4), _W(4), _W(4), 2),
    (_r.normal(size=(1, 4, 1, 5)) * 4, _W(4) * 3, _W(4) * 3, _W(4), _W(4), 4),
    (_r.normal(size=(2, 6, 2, 3)), _W(6), _W(6), _W(6), _W(6), 3),
]

EXPLANATION = r"""
## Why it works
After flattening, this is ordinary self-attention over $N = HW$ tokens. Each output pixel is a convex combination of all value vectors, weighted by query-key similarity, so information moves across the whole image in one layer. The residual connection plus a zero-initialised $W_o$ (test 1) makes the block start as the identity.

With zero query and key weights (test 2) every score is $0$, the softmax is uniform, and each pixel receives the mean of the normalised feature map.

## Complexity
$O(B\,N^2\,C + B\,N\,C^2)$ time and $O(B\,N^2)$ memory for the attention matrix. That quadratic cost in pixels is why U-Nets only attend at $16\times16$ or $8\times8$ resolution.

## Common bugs
- Reshaping `(B, N, C)` straight to `(B, C, H, W)` without transposing back. Shapes match, values are scrambled.
- Scaling by $\sqrt{N}$ instead of $\sqrt{C}$ (the key dimension).
- Softmax over the query axis instead of the key axis.

## Interview follow-ups
- How would you add text conditioning here? (Cross-attention: queries from pixels, keys and values from text tokens.)
- How does multi-head attention change the shapes and the cost?
"""
