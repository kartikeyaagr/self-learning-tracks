import numpy as np

META = dict(title='Multi-head latent attention with decoupled RoPE', tags=['Attention', 'Inference', 'Transformers'],
            diff='Hard', runtime='torch', lessons=['ieng:20.03', 'ieng:03.06'],
            related=['multi-query-attention-mqa', 'rotary-position-embedding-rope', 'kv-cache-memory-for-mha-mqa-gqa-and-mla'])

STATEMENT = r"""
Multi-head latent attention (MLA) caches one small **latent** vector per token instead of per-head keys and values. Keys and values for every head are re-expanded from that latent. RoPE does not commute with the up-projection, so position information travels in a separate, small key that all heads share.

Write `mla(x, W_dkv, W_uk, W_uv, W_kr, W_q, W_o, n_heads, d_rope, base=10000.0)` in PyTorch: causal self-attention over a full sequence (prefill), returning the output and the two tensors an engine would cache.

## Shapes (all float64; projections are $XW$, no biases)
- `x`: `(B, T, D)`. $H$ = `n_heads`, $d_r$ = `d_rope` (even), $d_c$ = latent width, $d_h$ = per-head content width.
- `W_dkv`: `(D, d_c)`, `W_uk`: `(d_c, H d_h)`, `W_uv`: `(d_c, H d_h)`, `W_kr`: `(D, d_r)`, `W_q`: `(D, H (d_h + d_r))`, `W_o`: `(H d_h, D)`.

## Computation
1. Latent: $c = x W_{dkv}$, `(B, T, d_c)`. Content keys $k^C = c\,W_{uk}$ and values $v = c\,W_{uv}$, split into heads of width $d_h$ (head $h$ = columns $h d_h : (h+1) d_h$).
2. Shared position key: $k^R = \mathrm{RoPE}(x W_{kr})$, `(B, T, d_r)`, one for all heads.
3. Queries: reshape $x W_q$ to `(B, T, H, d_h + d_r)`. In each head the first $d_h$ features are $q^C$, the last $d_r$ are $q^R$, and $q^R \leftarrow \mathrm{RoPE}(q^R)$.
4. Scores for head $h$: $\;s_{ij} = \bigl(q^C_{i} \cdot k^C_{j} + q^R_{i} \cdot k^R_{j}\bigr) / \sqrt{d_h + d_r}$, causal ($j \le i$), softmax over $j$, output $\sum_j a_{ij} v_j$.
5. Concatenate heads in order to `(B, T, H d_h)` and multiply by $W_o$.

**RoPE** here uses the "half" layout: for a vector $u$ of width $d_r$ at position $m$ (tokens are at positions $0, \dots, T-1$), pair $i$ is features $(i, i + d_r/2)$, rotated by angle $m\,\theta_i$ with $\theta_i = \text{base}^{-2i/d_r}$:
$$(u_i, u_{i+d_r/2}) \mapsto (u_i \cos m\theta_i - u_{i+d_r/2} \sin m\theta_i,\;\; u_i \sin m\theta_i + u_{i+d_r/2} \cos m\theta_i).$$

## Output
A tuple `(y, c, k_rope)`: `y` is `(B, T, D)`, `c` is `(B, T, d_c)`, `k_rope` is `(B, T, d_r)` (after RoPE).

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def mla(x, W_dkv, W_uk, W_uv, W_kr, W_q, W_o, n_heads, d_rope, base=10000.0):
    """Return (y (B,T,D), latent cache c (B,T,dc), rotated shared key k_rope (B,T,d_rope))."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import torch

def rope_half(u, base):
    """Rotate the last axis of u (..., T, d) with the half layout; positions are 0..T-1."""
    T, d = u.shape[-2], u.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=u.dtype) / d)          # (d/2,)
    ang = torch.arange(T, dtype=u.dtype)[:, None] * theta                # (T, d/2)
    cos, sin = ang.cos(), ang.sin()
    a, b = u[..., : d // 2], u[..., d // 2:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)

def mla(x, W_dkv, W_uk, W_uv, W_kr, W_q, W_o, n_heads, d_rope, base=10000.0):
    B, T, _ = x.shape
    H = n_heads
    dh = W_uk.shape[1] // H

    c = x @ W_dkv                                                         # (B, T, dc): the cache
    kC = (c @ W_uk).view(B, T, H, dh).transpose(1, 2)                     # (B, H, T, dh)
    v = (c @ W_uv).view(B, T, H, dh).transpose(1, 2)
    kR = rope_half(x @ W_kr, base)                                        # (B, T, dr): also cached

    q = (x @ W_q).view(B, T, H, dh + d_rope).transpose(1, 2)              # (B, H, T, dh + dr)
    qC, qR = q[..., :dh], rope_half(q[..., dh:], base)

    # content part per head + position part against the single shared key (broadcast over heads)
    scores = (qC @ kC.transpose(-2, -1) + qR @ kR.unsqueeze(1).transpose(-2, -1)) / math.sqrt(dh + d_rope)
    future = torch.ones(T, T, dtype=torch.bool).triu(1)
    attn = torch.softmax(scores.masked_fill(future, float('-inf')), dim=-1)
    o = (attn @ v).transpose(1, 2).reshape(B, T, H * dh)
    return o @ W_o, c, kR
'''

HINTS = [
    'Work out the per-head content width first: `dh = W_uk.shape[1] // n_heads`. Then the query projection reshapes to `(B, T, H, dh + d_rope)`.',
    'Write one helper that applies half-layout RoPE to the last axis of any `(..., T, d)` tensor. Use it on `x @ W_kr` (shape `(B, T, d_r)`) and on the last `d_rope` query features (shape `(B, H, T, d_r)`).',
    'The score is a sum of two matmuls: `qC @ kC^T` per head and `qR @ kR^T` where `kR.unsqueeze(1)` broadcasts the shared key over heads. Divide by `sqrt(dh + d_rope)`, not `sqrt(dh)`.',
]

_r = np.random.default_rng(42)


def _case(B, T, D, H, dh, dr, dc, s=0.3):
    n = lambda *sh: _r.normal(size=sh) * s
    return (_r.normal(size=(B, T, D)), n(D, dc), n(dc, H * dh), n(dc, H * dh), n(D, dr), n(D, H * (dh + dr)),
            n(H * dh, D), H, dr)


_one = (np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.eye(2), np.eye(2), np.eye(2), np.eye(2),
        np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]), np.eye(2), 1, 2)
TESTS = [
    _one,
    (np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]), *_one[1:]),
    _case(2, 4, 8, 2, 3, 2, 4),
    _case(1, 6, 6, 3, 2, 4, 3),
    _case(2, 1, 4, 2, 2, 2, 2),
    _case(1, 5, 16, 4, 4, 4, 6),
    (*_case(2, 3, 8, 2, 4, 2, 5)[:-1], 2, 500.0),
]

EXPLANATION = r"""
## Why it works
Without RoPE, the content score $q^C_i \cdot k^C_j = (x_i W^C_q)(c_j W_{uk})^\top$ only ever touches $c_j$, so the cache can hold $c_j$ alone. RoPE inserts a position-dependent rotation between $q$ and $k$ that cannot be folded into $W_{uk}$, so MLA moves position into a separate $d_r$-wide key that is rotated once and shared by every head. Its score adds to the content score, and the scale uses the full query width $d_h + d_r$.

## Cache size
Per token per layer MLA stores $d_c + d_r$ numbers, versus $2 H d_h$ for MHA. With DeepSeek-V2's sizes ($d_c = 512$, $d_r = 64$, $H = 128$, $d_h = 128$) that is 576 versus 32,768: about 57 times smaller, and smaller than GQA with 8 KV heads ($2 \cdot 8 \cdot 128 = 2048$).

## Weight absorption at decode time
Because $q^C_i \cdot k^C_j = (q^C_i W_{uk,h}^\top) \cdot c_j$, a decode kernel can multiply the query by $W_{uk,h}^\top$ once and attend directly over the latents, never materialising per-head keys. Likewise $W_{uv,h}$ folds into $W_o$. Decode then reads only $d_c + d_r$ numbers per cached token, which is the bandwidth win MLA was designed for. (This prefill version expands keys explicitly, which is cheaper when there are many queries.)

## Common bugs
- Applying RoPE to the content part, or applying it to $c$ before up-projection: positions get mixed into every head's keys and absorption breaks.
- Scaling by $\sqrt{d_h}$ only.
- Splitting the query as `[rope | content]` instead of `[content | rope]`; both are valid designs, but the checkpoint decides, and here content comes first.

## Interview follow-ups
- Derive the absorbed decode computation and its FLOPs per token. When is absorption slower than expanding keys?
- Why can't MLA's latent cache simply be quantised the same way per-head KV caches are? What changes in the error analysis?
"""
