import numpy as np

META = dict(title='DiT: patchify plus one adaLN-Zero block', tags=['Diffusion Models', 'Transformers', 'Conditioning'], diff='Hard',
            runtime='numpy', lessons=['aieng:04.02'],
            related=['reshaping-arrays', 'layer-norm', 'scaled-dot-product-attention', 'spatial-self-attention-block'])

STATEMENT = r"""
Diffusion Transformers (DiT) replace the U-Net with a plain Transformer over image patches. The conditioning (timestep plus class) does not enter as tokens; it **modulates** every block through adaptive LayerNorm, with gates that start at zero so each block begins as the identity.

## 1. Patchify and embed
`x` is `(B, C, H, W)` and the patch size `p` divides $H$ and $W$. Patches are taken in row-major order, patch $n = i\cdot(W/p) + j$ for patch row $i$ and column $j$. Each patch `x[b, :, i*p:(i+1)*p, j*p:(j+1)*p]` is flattened in **(row in patch, column in patch, channel)** order, i.e. transposed to `(p, p, C)` and then reshaped to a vector of length $p^2C$. Tokens: $T = \text{patches}\,W_{patch} + b_{patch}$, shape `(B, N, D)`.

## 2. Modulation
$m = \mathrm{SiLU}(c)\,W_{mod} + b_{mod}$, shape `(B, 6D)`, split into six consecutive `(B, D)` chunks in this order: $\text{shift}_1, \text{scale}_1, \text{gate}_1, \text{shift}_2, \text{scale}_2, \text{gate}_2$. Each is broadcast over the $N$ tokens of its example.

## 3. Block
With $\mathrm{LN}$ a LayerNorm over the last axis (population variance, $\epsilon = 10^{-6}$, no learned affine):

$$h = \mathrm{LN}(T)\odot(1 + \text{scale}_1) + \text{shift}_1, \qquad T \leftarrow T + \text{gate}_1 \odot \mathrm{Attn}(h)$$
$$h = \mathrm{LN}(T)\odot(1 + \text{scale}_2) + \text{shift}_2, \qquad T \leftarrow T + \text{gate}_2 \odot \mathrm{MLP}(h)$$

- $\mathrm{Attn}(h) = \mathrm{softmax}\!\big(QK^\top/\sqrt{D}\big)V\,W_o$ with $Q = hW_q$, $K = hW_k$, $V = hW_v$, single head, softmax over keys, no mask.
- $\mathrm{MLP}(h) = \mathrm{GELU}(hW_1 + b_1)\,W_2 + b_2$ with the tanh approximation $\mathrm{GELU}(z) = \tfrac12 z\big(1 + \tanh(\sqrt{2/\pi}\,(z + 0.044715 z^3))\big)$.

Write `dit_block(x, c, params, p)` and return the final tokens, shape `(B, N, D)`.

## `params` (dict of arrays)
| key | shape |
| `W_patch`, `b_patch` | `(p*p*C, D)`, `(D,)` |
| `W_mod`, `b_mod` | `(D, 6D)`, `(6D,)` |
| `Wq`, `Wk`, `Wv`, `Wo` | `(D, D)` each |
| `W1`, `b1`, `W2`, `b2` | `(D, M)`, `(M,)`, `(M, D)`, `(D,)` |

`c` is `(B, D)`.
"""

STARTER = '''import numpy as np

def dit_block(x, c, params, p):
    """x: (B, C, H, W), c: (B, D). Patchify with patch size p, embed, apply one adaLN-Zero block. Return (B, N, D)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def patchify(x, p):
    B, C, H, W = x.shape
    g = x.reshape(B, C, H // p, p, W // p, p)          # (B, C, i, row, j, col)
    g = g.transpose(0, 2, 4, 3, 5, 1)                  # (B, i, j, row, col, C)
    return g.reshape(B, (H // p) * (W // p), p * p * C)

def layer_norm(z, eps=1e-6):
    mu = z.mean(axis=-1, keepdims=True)
    return (z - mu) / np.sqrt(z.var(axis=-1, keepdims=True) + eps)

def gelu(z):
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))

def attention(h, P):
    q, k, v = h @ P['Wq'], h @ P['Wk'], h @ P['Wv']
    s = q @ k.transpose(0, 2, 1) / np.sqrt(h.shape[-1])
    a = np.exp(s - s.max(axis=-1, keepdims=True))
    a /= a.sum(axis=-1, keepdims=True)
    return (a @ v) @ P['Wo']

def dit_block(x, c, params, p):
    P = {k: np.asarray(v, dtype=float) for k, v in params.items()}
    x, c = np.asarray(x, dtype=float), np.asarray(c, dtype=float)
    T = patchify(x, p) @ P['W_patch'] + P['b_patch']               # (B, N, D)
    m = (c / (1.0 + np.exp(-c))) @ P['W_mod'] + P['b_mod']          # SiLU(c) projection, (B, 6D)
    sh1, sc1, g1, sh2, sc2, g2 = (z[:, None, :] for z in np.split(m, 6, axis=1))
    h = layer_norm(T) * (1.0 + sc1) + sh1
    T = T + g1 * attention(h, P)
    h = layer_norm(T) * (1.0 + sc2) + sh2
    T = T + g2 * (gelu(h @ P['W1'] + P['b1']) @ P['W2'] + P['b2'])
    return T
'''

HINTS = [
    'Patchify with one reshape and one transpose: `(B, C, H, W) -> (B, C, H/p, p, W/p, p)`, move axes to `(B, H/p, W/p, p, p, C)`, then flatten the last three and merge the two patch-grid axes.',
    '`np.split(m, 6, axis=1)` gives the six modulation chunks in order. Add a token axis (`[:, None, :]`) so each `(B, D)` chunk broadcasts over `(B, N, D)`.',
    'The block is two identical sub-steps: modulated LayerNorm, a sub-layer (attention, then MLP), and a gated residual add. With `W_mod` and `b_mod` zero every gate is 0, so the output equals the patch embeddings: use that to test your patchify.',
]

_r = np.random.default_rng(79)

def _params(C, p, D, M, zero=False):
    P = {'W_patch': _r.normal(size=(p * p * C, D)) / p, 'b_patch': _r.normal(size=D) * 0.1,
         'W_mod': _r.normal(size=(D, 6 * D)) / np.sqrt(D), 'b_mod': _r.normal(size=6 * D) * 0.1,
         'Wq': _r.normal(size=(D, D)) / np.sqrt(D), 'Wk': _r.normal(size=(D, D)) / np.sqrt(D),
         'Wv': _r.normal(size=(D, D)) / np.sqrt(D), 'Wo': _r.normal(size=(D, D)) / np.sqrt(D),
         'W1': _r.normal(size=(D, M)) / np.sqrt(D), 'b1': _r.normal(size=M) * 0.1,
         'W2': _r.normal(size=(M, D)) / np.sqrt(M), 'b2': _r.normal(size=D) * 0.1}
    if zero:
        P['W_mod'], P['b_mod'] = np.zeros((D, 6 * D)), np.zeros(6 * D)
    return P

_eye_patch = dict(_params(1, 2, 4, 4, zero=True), W_patch=np.eye(4), b_patch=np.zeros(4))
TESTS = [
    (np.arange(16.0).reshape(1, 1, 4, 4), np.zeros((1, 4)), _eye_patch, 2),
    (np.arange(8.0).reshape(1, 2, 2, 2), np.zeros((1, 8)), dict(_params(2, 1, 8, 4, zero=True), W_patch=np.eye(2, 8), b_patch=np.zeros(8)), 1),
    (_r.normal(size=(2, 3, 4, 4)), _r.normal(size=(2, 6)), _params(3, 2, 6, 12, zero=True), 2),
    (_r.normal(size=(2, 3, 4, 4)), _r.normal(size=(2, 6)), _params(3, 2, 6, 12), 2),
    (_r.normal(size=(1, 2, 6, 4)), _r.normal(size=(1, 8)), _params(2, 2, 8, 16), 2),
    (_r.normal(size=(3, 1, 4, 8)), _r.normal(size=(3, 4)), _params(1, 4, 4, 8), 4),
]

EXPLANATION = r"""
## Why adaLN-Zero
Conditioning by modulation is cheap (no extra tokens, no cross-attention) and acts on every block. The scale is written as $1 + \text{scale}$ so a zero modulation means "plain LayerNorm". Zero-initialising the final layer of the modulation MLP sets every **gate** to $0$, so each residual branch contributes nothing at the start and the deep network begins as an identity map over patch embeddings, which stabilises training. Tests 1-3 use that initial state; test 1 also checks the patch order against `np.arange`.

## Patch order
Any fixed order works for training, but the un-patchify at the output must be its exact inverse, and pretrained weights assume one convention. The row-major patch grid with (row, column, channel) inside each patch is the common one.

## Complexity
Patchify is a reshape. The block costs $O(BN^2D + BND^2 + BNDM)$; with $N = (H/p)^2$, halving $p$ multiplies attention cost by 16, which is why DiT quality-vs-compute is reported per patch size.

## Common bugs
- Flattening patches as `(C, p, p)` while the weights expect `(p, p, C)`, or reshaping straight to `(B, N, p*p*C)` without the transpose (test 2 uses $p = 1$, where only the channel ordering matters).
- Splitting the modulation vector in a different order than the weights expect.
- Applying `gate` to the normalised input instead of to the sub-layer output.

## Interview follow-ups
- How would you un-patchify the final `(B, N, p*p*C_out)` tokens back to an image?
- Compare adaLN with cross-attention for conditioning on a text prompt. Why do text-to-image DiTs often use both?
"""
