import numpy as np

META = dict(title='Diffusion Transformer forward pass: tokens to noise prediction', tags=['Diffusion Models', 'Transformers'],
            diff='Hard', runtime='numpy', lessons=['ieng:26.08'],
            related=['dit-patchify-and-adaln-zero-block', 'sinusoidal-timestep-embedding', 'vit-patchify', 'classifier-free-guidance'])

STATEMENT = r"""
A Diffusion Transformer (DiT) replaces the U-Net denoiser with a ViT-style transformer over latent patches. The timestep and class label enter every block through **adaptive LayerNorm**: they produce per-block shifts, scales and residual gates. Implement the full forward pass for one latent.

Write `dit_forward(x, t, y, p, patch, n_heads)` returning `(eps, sigma)`, each `(C, H, W)`.

## Conditioning vector
- Frequencies $f_i = \exp(-\ln(10^4)\, i/h)$ for $i < h$, with $h$ = `p['W_t1'].shape[0] // 2`; $e = [\cos(tf);\ \sin(tf)]$.
- $c = \mathrm{SiLU}(eW_{t1} + b_{t1})W_{t2} + b_{t2} + \text{y\_table}[y]$.

## Tokens
Patches of `x` `(C, H, W)` in row-major grid order, each flattened in `(C, P, P)` order; tokens $= \text{patches}\,W_e + b_e + \text{pos}$ `(N, D)`.

## Block (adaLN-Zero), for each dict in `p['blocks']`
- $[\text{sh}_1, \text{sc}_1, g_1, \text{sh}_2, \text{sc}_2, g_2] = $ split of $\mathrm{SiLU}(c)W_{ada} + b_{ada}$ into six $D$-vectors.
- $h = h + g_1\odot\mathrm{Attn}(\mathrm{LN}(h)\odot(1 + \text{sc}_1) + \text{sh}_1)$, then $h = h + g_2\odot\mathrm{MLP}(\mathrm{LN}(h)\odot(1 + \text{sc}_2) + \text{sh}_2)$.
- LN has no learned affine, $\epsilon = 10^{-6}$. Attn: `qkv = u @ W_qkv + b_qkv` split $[Q|K|V]$, contiguous heads, $1/\sqrt{d_h}$, no mask, `@ W_proj + b_proj`. MLP: $\mathrm{GELU}_{tanh}(uW_1 + b_1)W_2 + b_2$.

## Final layer and output
$[\text{sh}, \text{sc}] = \mathrm{SiLU}(c)W_f + b_f$; $o = (\mathrm{LN}(h)\odot(1 + \text{sc}) + \text{sh})W_{out} + b_{out}$, each row of width $2CP^2$. Reshape each row to `(2C, P, P)` and place it back at its patch position to get `(2C, H, W)`. The first $C$ channels are `eps`, the rest `sigma` (the learned-variance output).
"""

STARTER = '''import numpy as np

def dit_forward(x, t, y, p, patch, n_heads):
    """Return (eps (C, H, W), sigma (C, H, W))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _silu(u):
    return u / (1 + np.exp(-u))

def _ln(u, eps=1e-6):
    return (u - u.mean(-1, keepdims=True)) / np.sqrt(u.var(-1, keepdims=True) + eps)

def _gelu(u):
    return 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))

def _attn(u, B, H):
    N, D = u.shape
    dh = D // H
    q, k, v = np.split(u @ B['W_qkv'] + B['b_qkv'], 3, axis=-1)
    sp = lambda a: a.reshape(N, H, dh).transpose(1, 0, 2)
    q, k, v = sp(q), sp(k), sp(v)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    return (a @ v).transpose(1, 0, 2).reshape(N, D) @ B['W_proj'] + B['b_proj']

def dit_forward(x, t, y, p, patch, n_heads):
    x = np.asarray(x, dtype=float)
    C, Hh, Ww = x.shape
    P, gh, gw = patch, Hh // patch, Ww // patch
    half = p['W_t1'].shape[0] // 2
    f = np.exp(-np.log(10000.0) * np.arange(half) / half)
    e = np.concatenate([np.cos(t * f), np.sin(t * f)])
    c = _silu(e @ p['W_t1'] + p['b_t1']) @ p['W_t2'] + p['b_t2'] + p['y_table'][y]
    sc_in = _silu(c)
    tok = x.reshape(C, gh, P, gw, P).transpose(1, 3, 0, 2, 4).reshape(gh * gw, C * P * P)
    h = tok @ p['W_e'] + p['b_e'] + p['pos']
    for B in p['blocks']:
        sh1, s1, g1, sh2, s2, g2 = np.split(sc_in @ B['W_ada'] + B['b_ada'], 6)
        h = h + g1 * _attn(_ln(h) * (1 + s1) + sh1, B, n_heads)
        h = h + g2 * (_gelu((_ln(h) * (1 + s2) + sh2) @ B['W1'] + B['b1']) @ B['W2'] + B['b2'])
    sh, s = np.split(sc_in @ p['W_f'] + p['b_f'], 2)
    o = (_ln(h) * (1 + s) + sh) @ p['W_out'] + p['b_out']
    out = o.reshape(gh, gw, 2 * C, P, P).transpose(2, 0, 3, 1, 4).reshape(2 * C, Hh, Ww)
    return out[:C], out[C:]
'''

HINTS = [
    'Compute the conditioning vector $c$ once; every block and the final layer read $\\mathrm{SiLU}(c)$ through their own linear map.',
    'Patchify: `x.reshape(C, gh, P, gw, P).transpose(1, 3, 0, 2, 4)`; unpatchify is the inverse with $2C$ channels: `o.reshape(gh, gw, 2C, P, P).transpose(2, 0, 3, 1, 4)`.',
    'adaLN modulation is `LN(h) * (1 + scale) + shift`, and the gate multiplies the sublayer output before the residual add. If the gates are zero, a block is the identity (the "Zero" in adaLN-Zero).',
]

_r = np.random.default_rng(2212)
def _block(D, F):
    n = lambda *s: _r.normal(size=s)
    return dict(W_ada=n(D, 6 * D) * 0.3, b_ada=0.1 * n(6 * D), W_qkv=n(D, 3 * D) / np.sqrt(D), b_qkv=0.1 * n(3 * D),
                W_proj=n(D, D) / np.sqrt(D), b_proj=0.1 * n(D), W1=n(D, F) / np.sqrt(D), b1=0.1 * n(F),
                W2=n(F, D) / np.sqrt(F), b2=0.1 * n(D))
def _p(C, P, N, D, F, Ft, K, nb, zero=False):
    n = lambda *s: _r.normal(size=s)
    blocks = [_block(D, F) for _ in range(nb)]
    if zero:
        for B in blocks:
            B['W_ada'][:] = 0.0; B['b_ada'][:] = 0.0
    return dict(W_t1=n(Ft, D) / np.sqrt(Ft), b_t1=0.1 * n(D), W_t2=n(D, D) / np.sqrt(D), b_t2=0.1 * n(D), y_table=n(K + 1, D) * 0.5,
                W_e=n(C * P * P, D) / np.sqrt(C * P * P), b_e=0.1 * n(D), pos=0.1 * n(N, D), blocks=blocks,
                W_f=n(D, 2 * D) * 0.3, b_f=0.1 * n(2 * D), W_out=n(D, 2 * C * P * P) / np.sqrt(D), b_out=0.1 * n(2 * C * P * P))
_p1 = _p(1, 2, 4, 4, 8, 4, 3, 1)
_p2 = _p(2, 2, 6, 8, 16, 8, 5, 2)
_p3 = _p(2, 2, 6, 8, 16, 8, 5, 2, zero=True)
TESTS = [
    (_r.normal(size=(1, 4, 4)), 10, 0, _p1, 2, 1),
    (_r.normal(size=(1, 4, 4)), 999, 3, _p1, 2, 2),
    (_r.normal(size=(2, 4, 6)), 500, 2, _p2, 2, 2),
    (_r.normal(size=(2, 4, 6)), 0, 5, _p2, 2, 4),
    (_r.normal(size=(2, 4, 6)), 250, 1, _p3, 2, 2),
]

EXPLANATION = r"""
## What DiT showed
A plain transformer over latent patches, with no convolutions, scales cleanly as a diffusion denoiser: sample quality (FID) improves steadily with more GFLOPs per forward pass, whether from more depth, more width or smaller patches. DiT-XL/2 set a new state of the art on class-conditional ImageNet 256×256.

## Why adaLN-Zero
The paper compared in-context conditioning (extra tokens), cross-attention and adaptive LayerNorm. Regressing per-block shift, scale and gate from $c$ was the cheapest and best, and initialising the regressors to zero makes every block start as the identity (test 5: with zero modulation each block passes $h$ through unchanged), which stabilises training of deep stacks.

## Outputs
The model predicts both the noise and a per-pixel interpolation coefficient for the variance (the "learned sigma" from improved DDPM), hence $2C$ output channels. The extra class row in `y_table` is the null label used for classifier-free guidance.

## Complexity
$N = HW/P^2$ tokens; attention is $O(N^2D)$ per block, so halving the patch size quadruples $N$ and multiplies attention cost by 16.

## Common bugs
- Using $\text{scale}$ instead of $1 + \text{scale}$, which zeroes the normalised input at initialisation.
- Gating the residual sum instead of the sublayer output.
- Unpatchifying in a different order than patchifying.

## Interview follow-ups
- How would you add text conditioning to this model (as later DiT-based text-to-image models do)?
- Count the adaLN parameters per block for $D = 1152$ and compare with the attention parameters.
"""
