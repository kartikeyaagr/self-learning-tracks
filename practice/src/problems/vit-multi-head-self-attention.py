import numpy as np

META = dict(title='ViT pre-norm attention sublayer', tags=['Attention', 'Computer Vision', 'Transformers'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:06.02', 'aieng:05.02'], related=['scaled-dot-product-attention-2', 'layer-norm', 'vit-patchify'])

STATEMENT = r"""
Each ViT encoder block starts with a **pre-norm** multi-head self-attention sublayer: LayerNorm, a fused QKV projection, attention over all tokens (no causal mask, since image patches have no order to protect), an output projection, and a residual connection.

Write `vit_attention(x, ln_g, ln_b, w_qkv, b_qkv, w_o, b_o, num_heads, eps=1e-6)` and return `(y, attn)`.

## Steps
1. **LayerNorm** over the last axis: $\hat x = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot g + \beta$, with $\mu$, $\sigma^2$ the mean and the biased (divide by `D`) variance of each token.
2. **QKV**: `qkv = x_hat @ w_qkv + b_qkv` of shape `(B, N, 3D)`. The first `D` columns are Q, the next `D` are K, the last `D` are V.
3. **Heads**: with `h = num_heads` and `d = D / h`, head `m` uses columns `m*d : (m+1)*d` of each of Q, K, V.
4. **Attention** per head: $A = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt d}\right)$ along the key axis, output $A V$.
5. **Merge**: concatenate the heads' outputs along the feature axis in head order to `(B, N, D)`, then `out @ w_o + b_o`.
6. **Residual**: `y = x + (that projection)`.

## Input
- `x`: float `(B, N, D)`. `ln_g`, `ln_b`: `(D,)`. `w_qkv`: `(D, 3D)`, `b_qkv`: `(3D,)`. `w_o`: `(D, D)`, `b_o`: `(D,)`. `D` is divisible by `num_heads`.

## Output
- `y`: `float64` `(B, N, D)`.
- `attn`: `float64` `(B, h, N, N)`, the attention probabilities (each row sums to 1).
"""

STARTER = '''import numpy as np

def vit_attention(x, ln_g, ln_b, w_qkv, b_qkv, w_o, b_o, num_heads, eps=1e-6):
    """x: (B, N, D). Return (y (B, N, D), attn (B, heads, N, N)) for x + MHSA(LayerNorm(x))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def vit_attention(x, ln_g, ln_b, w_qkv, b_qkv, w_o, b_o, num_heads, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    B, N, D = x.shape
    h, d = num_heads, D // num_heads

    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)                                    # biased variance
    xn = (x - mu) / np.sqrt(var + eps) * ln_g + ln_b

    qkv = xn @ w_qkv + b_qkv                                          # (B, N, 3D)
    # (B, N, 3, h, d) -> (3, B, h, N, d): split Q/K/V, then heads
    q, k, v = qkv.reshape(B, N, 3, h, d).transpose(2, 0, 3, 1, 4)

    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d)                      # (B, h, N, N)
    s = s - s.max(-1, keepdims=True)                                  # stable softmax over keys
    attn = np.exp(s)
    attn /= attn.sum(-1, keepdims=True)

    out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D)           # merge heads in head order
    return x + out @ w_o + b_o, attn
'''

HINTS = [
    'Do it as a pipeline of shapes: `(B, N, D)` -> LayerNorm -> `(B, N, 3D)` -> `(3, B, h, N, d)` -> scores `(B, h, N, N)` -> `(B, h, N, d)` -> `(B, N, D)`.',
    'Because the columns are laid out as `[Q | K | V]` and each of those as `[head 0 | head 1 | ...]`, `qkv.reshape(B, N, 3, h, d)` puts the Q/K/V choice on axis 2 and the head on axis 3. Transpose to `(2, 0, 3, 1, 4)` and unpack.',
    'Scores: `q @ k.transpose(0, 1, 3, 2) / sqrt(d)`; subtract the row max before `exp`. Merge with `(attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D)`, project, and add `x`.',
]

_r = np.random.default_rng(36)
def _params(D, s=0.4):
    return (1 + 0.1 * _r.normal(size=D), 0.1 * _r.normal(size=D), s * _r.normal(size=(D, 3 * D)), 0.1 * _r.normal(size=3 * D),
            s * _r.normal(size=(D, D)), 0.1 * _r.normal(size=D))
TESTS = [
    (np.array([[[1.0, -1.0], [2.0, 0.0]]]), np.ones(2), np.zeros(2), np.hstack([np.eye(2)] * 3), np.zeros(6), np.eye(2), np.zeros(2), 1),
    (np.array([[[1.0, 0.0, 0.0, 1.0], [0.0, 2.0, 1.0, 0.0], [3.0, 1.0, 0.0, 0.0]]]), np.ones(4), np.zeros(4),
     np.hstack([np.eye(4)] * 3), np.zeros(12), np.eye(4), np.zeros(4), 2),
    (_r.normal(size=(2, 5, 8)), *_params(8), 2),
    (_r.normal(size=(1, 7, 12)), *_params(12), 3),
    (_r.normal(size=(3, 4, 6)), *_params(6, 0.8), 1),
    (_r.normal(size=(2, 1, 4)), *_params(4), 4),
    (_r.normal(size=(1, 6, 8)) * 5, *_params(8, 1.5), 4, 1e-5),
]

EXPLANATION = r"""
## Shapes are the whole problem
The fused projection computes Q, K and V in one GEMM. Reading its `3D` output columns as `(3, h, d)` exposes both splits at once, and one transpose moves the head axis next to the batch so every head runs as an independent batched matmul. Merging is the inverse: move heads back next to the feature axis and flatten, which concatenates heads in order.

## Why pre-norm and the residual
Normalising the sublayer's input (rather than its output, as the original Transformer did) keeps the residual stream un-normalised, so gradients flow through `x + ...` unchanged; ViTs train stably without learning-rate warm-up tricks this way. The residual means a block that outputs zeros is the identity, so stacking more blocks can't make the initial function worse.

## Complexity
Projections: $O(BND^2)$. Attention: $O(BN^2D)$ time and $O(BhN^2)$ memory for `attn`. For a 224 px image with 16 px patches, $N = 197$, so attention is cheap; at high resolution $N^2$ dominates, which motivates windowed attention.

## Common bugs
- Splitting heads with `reshape(B, N, h, 3, d)`: wrong interleaving of Q, K, V columns.
- Merging with `reshape(B, N, D)` **without** the transpose back, which mixes tokens and heads.
- Scaling by $\sqrt D$ instead of $\sqrt d$.
- Unbiased variance in LayerNorm, or forgetting the residual.

## Interview follow-ups
- Return the attention from the class token to every patch, averaged over heads, and reshape it to the patch grid. That's a basic attention map visualisation.
- What changes in the backward pass if you use FlashAttention? (`attn` is never materialised; it's recomputed blockwise from saved row statistics.)
"""
