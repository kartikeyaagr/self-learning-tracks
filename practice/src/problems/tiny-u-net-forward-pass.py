import numpy as np

META = dict(title='Forward pass of a tiny one-level U-Net', tags=['U-Net', 'Diffusion Models', 'Convolutional Networks'], diff='Hard',
            runtime='numpy', lessons=[],
            related=['sinusoidal-timestep-embedding', 'time-conditioned-residual-block', 'spatial-self-attention-block'])

STATEMENT = r"""
A diffusion U-Net maps a noisy image and a timestep to a same-shaped prediction. It goes down in resolution to see context, comes back up, and concatenates a **skip** connection so fine detail is not lost. This one has a single level and 1x1 convolutions, but the plumbing is the real thing.

Given `x` of shape `(B, C, H, W)` (with $H$ and $W$ even) and timesteps `t` of shape `(B,)`:

1. **Embed time.** $e$ = sinusoidal embedding of `t` with $E$ = `params["W_t"].shape[1]` dims (even): frequencies $\omega_i = \exp(-\ln(10000)\, i / (E/2))$ for $i < E/2$, layout $[\cos(t\omega) \mid \sin(t\omega)]$.
2. **Stem.** $h = \mathrm{conv}(x;\ W_{in}, b_{in}) + \big(\mathrm{SiLU}(e)\,W_t^\top + b_t\big)$, the time term broadcast over all pixels.
3. **Skip.** $s = \mathrm{SiLU}(h)$, shape `(B, F, H, W)`.
4. **Down.** $d$ = 2x2 average pooling of $s$ with stride 2, shape `(B, F, H/2, W/2)`.
5. **Middle.** $m = \mathrm{SiLU}(\mathrm{conv}(d;\ W_{mid}, b_{mid}))$.
6. **Up.** $u$ = nearest-neighbour 2x upsampling of $m$: `u[..., i, j] = m[..., i // 2, j // 2]`.
7. **Merge.** $c$ = concatenation of $[u, s]$ along the channel axis, **upsampled first**: shape `(B, 2F, H, W)`.
8. **Head.** return $\mathrm{conv}(c;\ W_{out}, b_{out})$, shape `(B, C, H, W)`.

Here $\mathrm{conv}(h; W, b)$ is a 1x1 convolution `out[b, o, i, j] = sum_k W[o, k] * h[b, k, i, j] + b[o]`, and $\mathrm{SiLU}(z) = z / (1 + e^{-z})$.

Write `tiny_unet(x, t, params)`.

## `params` (dict of arrays)
| key | shape |
| `W_in`, `b_in` | `(F, C)`, `(F,)` |
| `W_t`, `b_t` | `(F, E)`, `(F,)` |
| `W_mid`, `b_mid` | `(F, F)`, `(F,)` |
| `W_out`, `b_out` | `(C, 2F)`, `(C,)` |
"""

STARTER = '''import numpy as np

def tiny_unet(x, t, params):
    """x: (B, C, H, W) with even H, W; t: (B,). Return the (B, C, H, W) prediction."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def silu(z):
    return z / (1.0 + np.exp(-z))

def conv1x1(h, W, b):
    return np.einsum('ok,bkij->boij', W, h) + b[None, :, None, None]

def embed(t, E):
    half = E // 2
    args = np.asarray(t, dtype=float)[:, None] * np.exp(-np.log(10000.0) * np.arange(half) / half)
    return np.concatenate([np.cos(args), np.sin(args)], axis=1)

def tiny_unet(x, t, params):
    x = np.asarray(x, dtype=float)
    p = {k: np.asarray(v, dtype=float) for k, v in params.items()}
    B, C, H, W = x.shape
    e = embed(t, p['W_t'].shape[1])
    tproj = silu(e) @ p['W_t'].T + p['b_t']                              # (B, F)
    s = silu(conv1x1(x, p['W_in'], p['b_in']) + tproj[:, :, None, None])  # skip, full resolution
    F = s.shape[1]
    d = s.reshape(B, F, H // 2, 2, W // 2, 2).mean(axis=(3, 5))           # 2x2 average pool
    m = silu(conv1x1(d, p['W_mid'], p['b_mid']))
    u = m.repeat(2, axis=2).repeat(2, axis=3)                             # nearest-neighbour upsample
    c = np.concatenate([u, s], axis=1)                                    # upsampled first, then skip
    return conv1x1(c, p['W_out'], p['b_out'])
'''

HINTS = [
    'Write small helpers (`silu`, a 1x1 `conv` via `einsum`, the time `embed`) and then follow the eight steps literally. Print shapes after each one.',
    'Average pooling with a 2x2 window and stride 2 is a reshape: `(B, F, H, W) -> (B, F, H/2, 2, W/2, 2)`, then `.mean(axis=(3, 5))`. Nearest upsampling is `.repeat(2, axis=2).repeat(2, axis=3)`.',
    'The concatenation order fixes which half of `W_out` sees which features: `np.concatenate([u, s], axis=1)`. The time projection is `(B, F)` and needs `[:, :, None, None]` before it is added to the stem.',
]

_r = np.random.default_rng(41)

def _params(C, F, E):
    return {'W_in': _r.normal(size=(F, C)), 'b_in': _r.normal(size=F) * 0.1,
            'W_t': _r.normal(size=(F, E)) / np.sqrt(E), 'b_t': _r.normal(size=F) * 0.1,
            'W_mid': _r.normal(size=(F, F)) / np.sqrt(F), 'b_mid': _r.normal(size=F) * 0.1,
            'W_out': _r.normal(size=(C, 2 * F)) / np.sqrt(2 * F), 'b_out': _r.normal(size=C) * 0.1}

_simple = {'W_in': np.array([[1.0]]), 'b_in': np.array([0.0]), 'W_t': np.zeros((1, 2)), 'b_t': np.array([0.0]),
           'W_mid': np.array([[1.0]]), 'b_mid': np.array([0.0]), 'W_out': np.array([[1.0, 0.0]]), 'b_out': np.array([0.0])}
_skip_only = dict(_simple, W_out=np.array([[0.0, 1.0]]), b_out=np.array([0.5]))
TESTS = [
    (np.array([[[[1.0, 3.0], [-1.0, 1.0]]]]), np.array([0]), _simple),
    (np.array([[[[1.0, 3.0], [-1.0, 1.0]]]]), np.array([0]), _skip_only),
    (_r.normal(size=(2, 3, 4, 4)), np.array([0, 999]), _params(3, 4, 6)),
    (_r.normal(size=(1, 1, 2, 6)), np.array([250]), _params(1, 2, 4)),
    (_r.normal(size=(3, 2, 4, 2)), np.array([10, 20, 30]), _params(2, 5, 8)),
    (np.zeros((1, 2, 2, 2)), np.array([500]), _params(2, 3, 4)),
]

EXPLANATION = r"""
## Why the shape of a U-Net
The encoder path halves resolution so each feature summarises a larger region; the decoder path brings it back to full size. Upsampling alone cannot recover detail lost in pooling, so the full-resolution features $s$ are concatenated in, and the head learns how much to take from each. Test 2 routes the output only through the skip half of $W_{out}$; test 1 only through the coarse half, where each $2\times2$ block of the output is constant.

## Where the time goes
The timestep enters once here, as a per-channel shift in the stem. Real U-Nets add a projected embedding inside **every** residual block, at every resolution, because each level needs to know the noise level. Test 6 feeds an all-zero image, so the output depends on `t` and the biases alone.

## Complexity
$O(B\,HW\,(FC + F^2/4 + 2FC) + B\,FE)$: linear in the number of pixels, because every convolution here is 1x1.

## Common bugs
- Concatenating `[s, u]` instead of `[u, s]`: shapes match but `W_out` multiplies the wrong features.
- Pooling with `reshape(B, F, 2, H//2, 2, W//2)`, which averages pixels half an image apart instead of neighbours.
- Upsampling with `np.tile`, which repeats the whole map instead of each pixel.

## Interview follow-ups
- Why do diffusion U-Nets use additive or concatenative skips rather than a plain autoencoder?
- How would you make the network accept a class label too? (Add a learned class embedding to the time embedding.)
"""
