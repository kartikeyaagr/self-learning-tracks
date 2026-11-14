import numpy as np

META = dict(title='Vision Transformer: images to class logits', tags=['Transformers', 'Computer Vision'], diff='Hard',
            runtime='numpy', lessons=['dlcore:04.05'],
            related=['vit-patchify', 'vision-transformer-patch-embedding', 'vit-multi-head-self-attention', 'transformer-encoder-block'])

STATEMENT = r"""
A Vision Transformer cuts an image into fixed-size patches, treats each flattened patch as a token, prepends a learned `[CLS]` token and runs a standard pre-LN Transformer encoder. The class prediction is read from the `[CLS]` position.

Write `vit_logits(images, p, patch, n_heads)` returning `(B, K)` logits.

## Inputs
- `images`: `(B, C, H, W)` floats; `H` and `W` are multiples of `patch`.
- `p['We']` `(C·P·P, D)`, `p['be']` `(D,)`: patch projection. `p['cls']` `(D,)`. `p['pos']` `(N+1, D)` learned positions, $N = HW/P^2$.
- `p['blocks']`: list of dicts with `ln1`, `ln2` (gamma, beta), `Wq Wk Wv Wo` `(D, D)`, `bq bk bv bo` `(D,)`, `W1` `(D, F)`, `b1`, `W2` `(F, D)`, `b2`.
- `p['ln_f']` (gamma, beta), `p['Wh']` `(D, K)`, `p['bh']` `(K,)`.

## Model
1. **Patches** in row-major order over the grid; patch $(i, j)$ is `images[b, :, iP:(i+1)P, jP:(j+1)P]` flattened in `(C, P, P)` order.
2. Tokens $= [\text{cls};\ \text{patches}\,W_e + b_e] + \text{pos}$, shape `(B, N+1, D)`.
3. Each block: $z = z + \mathrm{MSA}(\mathrm{LN}_1(z))$, then $z = z + \mathrm{MLP}(\mathrm{LN}_2(z))$, where MSA has biases on all four projections, no mask, $1/\sqrt{d_h}$ scaling and contiguous head blocks, and $\mathrm{MLP}(u) = \mathrm{GELU}(uW_1 + b_1)W_2 + b_2$ with the tanh-approximate GELU.
4. Logits $= \mathrm{LN}_f(z_{:,0})\,W_h + b_h$.

LayerNorm: population variance, $\epsilon = 10^{-6}$.
"""

STARTER = '''import numpy as np

def vit_logits(images, p, patch, n_heads):
    """Return (B, K) class logits."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _ln(z, gb, eps=1e-6):
    g, b = gb
    mu = z.mean(-1, keepdims=True)
    return g * (z - mu) / np.sqrt(z.var(-1, keepdims=True) + eps) + b

def _gelu(u):
    return 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))

def _patches(x, P):
    B, C, H, W = x.shape
    g = x.reshape(B, C, H // P, P, W // P, P).transpose(0, 2, 4, 1, 3, 5)   # (B, gh, gw, C, P, P)
    return g.reshape(B, (H // P) * (W // P), C * P * P)

def _msa(z, L, H):
    B, T, D = z.shape
    dh = D // H
    heads = lambda W, b: (z @ W + b).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    Q, K, V = heads(L['Wq'], L['bq']), heads(L['Wk'], L['bk']), heads(L['Wv'], L['bv'])
    S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)
    A = np.exp(S - S.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    return (A @ V).transpose(0, 2, 1, 3).reshape(B, T, D) @ L['Wo'] + L['bo']

def vit_logits(images, p, patch, n_heads):
    x = np.asarray(images, dtype=float)
    B = x.shape[0]
    tok = _patches(x, patch) @ p['We'] + p['be']
    cls = np.broadcast_to(p['cls'], (B, 1, tok.shape[-1]))
    z = np.concatenate([cls, tok], axis=1) + p['pos']
    for L in p['blocks']:
        z = z + _msa(_ln(z, L['ln1']), L, n_heads)
        z = z + _gelu(_ln(z, L['ln2']) @ L['W1'] + L['b1']) @ L['W2'] + L['b2']
    return _ln(z[:, 0], p['ln_f']) @ p['Wh'] + p['bh']
'''

HINTS = [
    'Patchify with one reshape and one transpose: `(B, C, H/P, P, W/P, P)` to `(B, H/P, W/P, C, P, P)`, then flatten the last three axes.',
    'The `[CLS]` vector is the same for every image: broadcast it to `(B, 1, D)` and concatenate before adding the position table, which has $N + 1$ rows.',
    'Only `z[:, 0]` reaches the head, after the final LayerNorm.',
]

_r = np.random.default_rng(2010)
def _lnp(D):
    return (1 + 0.1 * _r.normal(size=D), 0.1 * _r.normal(size=D))
def _model(C, P, N, D, F, K, nb):
    s = 1 / np.sqrt(D)
    blocks = []
    for _ in range(nb):
        L = {k: _r.normal(size=(D, D)) * s for k in ('Wq', 'Wk', 'Wv', 'Wo')}
        L.update({k: 0.1 * _r.normal(size=D) for k in ('bq', 'bk', 'bv', 'bo')})
        L.update(W1=_r.normal(size=(D, F)) * s, b1=0.1 * _r.normal(size=F), W2=_r.normal(size=(F, D)) / np.sqrt(F),
                 b2=0.1 * _r.normal(size=D), ln1=_lnp(D), ln2=_lnp(D))
        blocks.append(L)
    return {'We': _r.normal(size=(C * P * P, D)) / np.sqrt(C * P * P), 'be': 0.1 * _r.normal(size=D),
            'cls': _r.normal(size=D) * 0.5, 'pos': _r.normal(size=(N + 1, D)) * 0.1, 'blocks': blocks,
            'ln_f': _lnp(D), 'Wh': _r.normal(size=(D, K)) * s, 'bh': 0.1 * _r.normal(size=K)}

_m1 = _model(1, 2, 4, 4, 8, 3, 1)
_m2 = _model(3, 2, 6, 8, 16, 5, 2)
_m3 = _model(2, 3, 4, 6, 12, 4, 1)
TESTS = [
    (_r.normal(size=(1, 1, 4, 4)), _m1, 2, 1),
    (_r.normal(size=(2, 1, 4, 4)), _m1, 2, 2),
    (_r.normal(size=(2, 3, 4, 6)), _m2, 2, 2),
    (_r.normal(size=(3, 3, 4, 6)), _m2, 2, 4),
    (_r.normal(size=(2, 2, 6, 6)), _m3, 3, 3),
    (np.zeros((1, 2, 6, 6)), _m3, 3, 2),
]

EXPLANATION = r"""
## What the paper changed
Convolutions build in locality and translation equivariance. ViT drops almost all of that: the only image-specific step is cutting patches, and positions are a learned table with no 2-D structure. With enough pre-training data (the paper used JFT-300M) this matched or beat strong CNNs; on ImageNet alone it underfit the inductive bias gap, which later work (DeiT) closed with augmentation and distillation.

## The pieces
- **Patch embedding** is exactly a convolution with kernel = stride = $P$ and $D$ output channels; the reshape-and-matmul form makes that explicit.
- **`[CLS]`** borrows BERT's trick: a token with no content whose final state summarises the image. Global average pooling of the patch tokens works about as well.
- **Pre-LN** blocks with a final LayerNorm, the GPT-2 arrangement.

## Complexity
$N = HW/P^2$ tokens, so attention costs $O(N^2 D)$ per block: halving the patch size multiplies attention cost by 16. This is why ViTs use $P = 14$ or 16 at 224×224 (196 or 256 tokens).

## Common bugs
- Flattening patches in `(P, P, C)` order when the weights expect `(C, P, P)`; both "work" but don't match given weights.
- Adding positions before concatenating `[CLS]` (the table has $N+1$ rows for a reason).
- Reading the logits from the mean over tokens instead of position 0.

## Interview follow-ups
- How would you run this model at a higher resolution than it was trained on? (Interpolate the position table in 2-D.)
- Why did ViT need far more data than ResNet to reach the same accuracy?
"""
