import numpy as np

META = dict(title='DeepSeek-V3 block: latent attention plus a shared-expert MoE', tags=['Transformers', 'Attention'],
            diff='Hard', runtime='numpy', lessons=['aieng:06.02', 'ieng:19.02'],
            related=['multi-head-latent-attention-mla', 'aux-loss-free-moe-routing', 'moe-token-dispatch-and-expert-aggregation',
                     'kv-cache-memory-for-mha-mqa-gqa-and-mla'])

STATEMENT = r"""
A DeepSeek-V3 layer combines two ideas. **Multi-head latent attention (MLA)** caches one small latent vector per token instead of full keys and values, plus a single shared rotary key. **DeepSeekMoE** replaces the dense FFN with one always-on shared expert and many small routed experts. Implement one full block for a single sequence.

Write `deepseek_block(x, p, n_heads, k)` returning the `(T, D)` output.

## Block
$h = x + \mathrm{MLA}(\mathrm{RMS}_{attn}(x))$, then $y = h + \mathrm{MoE}(\mathrm{RMS}_{ffn}(h))$, with $\mathrm{RMS}_w(z) = w\odot z/\sqrt{\mathrm{mean}(z^2) + 10^{-6}}$.

## MLA (for input $z$, $H$ heads)
- Queries: $c_q = \mathrm{RMS}_{q\_norm}(zW_{dq})$; $q^C = c_qW_{uq}$ split into $H$ heads of width $d_n$; $q^R = \mathrm{RoPE}(c_qW_{qr})$ split into $H$ heads of width $d_r$.
- Keys and values: $c_{kv} = \mathrm{RMS}_{kv\_norm}(zW_{dkv})$ (this is what gets cached); $k^C = c_{kv}W_{uk}$ ($H$ heads of $d_n$); $v = c_{kv}W_{uv}$ ($H$ heads of $d_v$); $k^R = \mathrm{RoPE}(zW_{kr})$, **one** $d_r$-wide key shared by all heads.
- Head $h$: $q = [q^C_h; q^R_h]$, $k = [k^C_h; k^R]$; scores $qk^\top/\sqrt{d_n + d_r}$, causal mask, softmax, times $v_h$. Concatenate heads and multiply by $W_o$.
- RoPE, rotate-half layout, base $10^4$, positions from 0: $[x_1, x_2] \mapsto [x_1\cos - x_2\sin,\ x_2\cos + x_1\sin]$ with angle $t\cdot 10000^{-2i/d_r}$ for $i < d_r/2$. It is applied per head vector.

## MoE (for input $u$)
- Expert $e$ is a SwiGLU MLP: $\mathrm{FFN}_e(u) = (\mathrm{SiLU}(uW^e_1)\odot uW^e_3)W^e_2$. The shared expert has the same form.
- Router: $s = \sigma(u\,W_r)$ over the $E$ routed experts; choose the top `k` by $s + b$ (`p['router_bias']`); gates are the chosen $s$ values normalised to sum to 1 (no bias).
- $\mathrm{MoE}(u) = \mathrm{FFN}_{shared}(u) + \sum_{j} g_j\,\mathrm{FFN}_{e_j}(u)$.

## Parameter keys
`attn_norm`, `ffn_norm`, `q_norm`, `kv_norm`, `W_dq` `(D, d_cq)`, `W_uq` `(d_cq, H·d_n)`, `W_qr` `(d_cq, H·d_r)`, `W_dkv` `(D, d_c)`, `W_uk` `(d_c, H·d_n)`, `W_uv` `(d_c, H·d_v)`, `W_kr` `(D, d_r)`, `W_o` `(H·d_v, D)`, `shared` and each of `experts` = dict `W1`, `W3`, `W2`, `W_r` `(D, E)`, `router_bias` `(E,)`. Read $d_n, d_r, d_v$ from the shapes.
"""

STARTER = '''import numpy as np

def deepseek_block(x, p, n_heads, k):
    """Return the (T, D) output of one MLA + DeepSeekMoE block."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _rms(z, w, eps=1e-6):
    return w * z / np.sqrt((z * z).mean(-1, keepdims=True) + eps)

def _rope(x):
    """x: (..., T, d) rotate-half."""
    T, d = x.shape[-2], x.shape[-1]
    ang = np.arange(T)[:, None] * 10000.0 ** (-np.arange(0, d, 2) / d)
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

def _swiglu(u, e):
    g = u @ e['W1']
    return ((g / (1 + np.exp(-g))) * (u @ e['W3'])) @ e['W2']

def _mla(z, p, H):
    T = z.shape[0]
    dn = p['W_uk'].shape[1] // H
    dr = p['W_kr'].shape[1]
    dv = p['W_uv'].shape[1] // H
    heads = lambda a, w: a.reshape(T, H, w).transpose(1, 0, 2)
    cq = _rms(z @ p['W_dq'], p['q_norm'])
    q = np.concatenate([heads(cq @ p['W_uq'], dn), _rope(heads(cq @ p['W_qr'], dr))], axis=-1)
    ckv = _rms(z @ p['W_dkv'], p['kv_norm'])
    kr = np.broadcast_to(_rope((z @ p['W_kr'])[None]), (H, T, dr))         # one rotary key for all heads
    k = np.concatenate([heads(ckv @ p['W_uk'], dn), kr], axis=-1)
    v = heads(ckv @ p['W_uv'], dv)
    S = q @ k.transpose(0, 2, 1) / np.sqrt(dn + dr)
    S = np.where(np.tril(np.ones((T, T), dtype=bool)), S, -np.inf)
    A = np.exp(S - S.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    return (A @ v).transpose(1, 0, 2).reshape(T, H * dv) @ p['W_o']

def _moe(u, p, k):
    s = 1 / (1 + np.exp(-(u @ p['W_r'])))
    idx = np.argsort(-(s + p['router_bias']), axis=1, kind='stable')[:, :k]
    g = np.take_along_axis(s, idx, axis=1)
    g = g / g.sum(1, keepdims=True)
    out = _swiglu(u, p['shared'])
    for e, ex in enumerate(p['experts']):
        w = (g * (idx == e)).sum(1)                       # gate of expert e per token (0 if not chosen)
        rows = w > 0
        if rows.any():
            out[rows] += w[rows, None] * _swiglu(u[rows], ex)
    return out

def deepseek_block(x, p, n_heads, k):
    x = np.asarray(x, dtype=float)
    h = x + _mla(_rms(x, p['attn_norm']), p, n_heads)
    return h + _moe(_rms(h, p['ffn_norm']), p, k)
'''

HINTS = [
    'Build MLA in three parts: the compressed path ($c_q$, $c_{kv}$ and their up-projections), the rotary path ($q^R$ per head, $k^R$ shared), then concatenate content and rotary parts along the feature axis before the dot product.',
    'The shared rotary key has shape `(T, d_r)`; broadcast it to `(H, T, d_r)` after applying RoPE. The score scale uses $d_n + d_r$, the full query width.',
    'For the MoE, compute every token\'s top-$k$ and gates first, then loop over experts and process only the tokens that chose each one. Add the shared expert for every token.',
]

_r = np.random.default_rng(2024)
def _ex(D, F):
    return dict(W1=_r.normal(size=(D, F)) / np.sqrt(D), W3=_r.normal(size=(D, F)) / np.sqrt(D), W2=_r.normal(size=(F, D)) / np.sqrt(F))
def _model(D, H, dcq, dc, dn, dr, dv, E, F):
    n = lambda *s: _r.normal(size=s)
    return dict(attn_norm=1 + 0.1 * n(D), ffn_norm=1 + 0.1 * n(D), q_norm=1 + 0.1 * n(dcq), kv_norm=1 + 0.1 * n(dc),
                W_dq=n(D, dcq) / np.sqrt(D), W_uq=n(dcq, H * dn) / np.sqrt(dcq), W_qr=n(dcq, H * dr) / np.sqrt(dcq),
                W_dkv=n(D, dc) / np.sqrt(D), W_uk=n(dc, H * dn) / np.sqrt(dc), W_uv=n(dc, H * dv) / np.sqrt(dc),
                W_kr=n(D, dr) / np.sqrt(D), W_o=n(H * dv, D) / np.sqrt(H * dv), shared=_ex(D, F),
                experts=[_ex(D, F) for _ in range(E)], W_r=n(D, E) / np.sqrt(D), router_bias=0.2 * n(E))

_m1 = _model(4, 1, 3, 2, 2, 2, 2, 2, 4)
_m2 = _model(8, 2, 4, 3, 4, 2, 3, 4, 6)
_m3 = _model(8, 4, 6, 4, 2, 4, 2, 6, 4)
TESTS = [
    (_r.normal(size=(3, 4)), _m1, 1, 1),
    (_r.normal(size=(3, 4)), _m1, 1, 2),
    (_r.normal(size=(5, 8)), _m2, 2, 2),
    (_r.normal(size=(5, 8)), _m2, 2, 1),
    (_r.normal(size=(6, 8)), _m3, 4, 3),
    (_r.normal(size=(1, 8)), _m3, 4, 2),
]

EXPLANATION = r"""
## MLA: what gets cached
At inference the cache holds $c_{kv}$ ($d_c$ numbers) and $k^R$ ($d_r$ numbers) per token per layer, instead of $2Hd$ for standard multi-head attention. DeepSeek-V3 uses $d_c = 512$, $d_r = 64$ against $H d = 128 \times 128$, so the cache is roughly 1/57 of MHA's. The up-projections $W_{uk}$ and $W_{uv}$ can be absorbed into $W_{uq}$ and $W_o$ at inference, so attention runs directly against the latent.

## Why a separate rotary key
RoPE inserts a position-dependent rotation between $q$ and $k$, so $W_{uk}$ could no longer be absorbed into the query side. MLA keeps the compressed path position-free and adds a small decoupled RoPE path, shared across heads to keep the cache small.

## DeepSeekMoE
Many small routed experts (256 in V3, 8 active) plus a shared expert that every token uses. The shared expert captures common knowledge so routed experts can specialise. Routing uses sigmoid affinities with bias-only load balancing.

## Complexity
Attention: $O(T^2 H(d_n + d_r))$; MoE: each token runs $k + 1$ small experts, so FLOPs scale with active parameters, not total.

## Common bugs
- Applying RoPE to the content parts or to the latent $c_{kv}$.
- Scaling scores by $\sqrt{d_n}$ instead of $\sqrt{d_n + d_r}$.
- Weighting expert outputs with `s + bias`.
- Forgetting the shared expert.

## Interview follow-ups
- Derive the absorbed form: write $q^C_h k^C_h{}^\top$ in terms of $c_q$ and $c_{kv}$ only.
- Compare MLA's per-token cache with GQA using 8 KV heads of width 128.
"""
