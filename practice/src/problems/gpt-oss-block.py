import numpy as np

META = dict(title='gpt-oss block: sinks, banded attention and a clamped-SwiGLU MoE', tags=['Transformers', 'Attention'],
            diff='Hard', runtime='numpy', lessons=['aieng:06.02', 'aieng:07.02'],
            related=['attention-with-sink-logits', 'causal-grouped-query-attention', 'sparse-moe-top-k-expert-routing', 'mxfp4-dequantize'])

STATEMENT = r"""
The open-weight gpt-oss models alternate **banded** (sliding-window) and **dense** attention layers, give each attention head a **learned sink logit**, and use a top-$k$ mixture of experts whose experts are a **clamped SwiGLU**. Implement one block for a single sequence.

Write `gpt_oss_block(x, p, n_heads, n_kv_heads, window, k)`. If `window` is `None` the layer is dense, otherwise banded. Return `(T, D)`.

## Block
$h = x + \mathrm{Attn}(\mathrm{RMS}_{attn}(x))$, $y = h + \mathrm{MoE}(\mathrm{RMS}_{mlp}(h))$, $\mathrm{RMS}_w(z) = w\odot z/\sqrt{\mathrm{mean}(z^2) + 10^{-5}}$.

## Attention
- `qkv = z @ W_qkv + b_qkv`, columns ordered $[Q\ (H d) \mid K\ (H_{kv} d) \mid V\ (H_{kv} d)]$, heads contiguous; $d$ = `W_o.shape[0] // H`.
- RoPE on $Q$ and $K$, rotate-half layout, base `p['rope_base']`, positions from 0.
- Query head $h$ uses KV head $\lfloor h/(H/H_{kv})\rfloor$. Scores $/\sqrt d$; key $j$ is allowed for query $t$ if $j \le t$ and (banded only) $j > t - \text{window}$.
- Softmax with the head's sink logit `sinks[h]` as an extra denominator column (no value). Output: concatenated heads `@ W_o + b_o`.

## MoE
- Router: `logits = u @ W_router + b_router` `(T, E)`. Pick the top `k`; their weights are the **softmax over the chosen $k$ logits** only.
- Expert $e$: `a = u @ W1[e] + b1[e]` has width $2F$ with gate and linear parts **interleaved**: `g = a[:, 0::2]`, `l = a[:, 1::2]`. Clamp `g` to at most 7 and `l` to $[-7, 7]$. Then $\text{act} = g\,\sigma(1.702\,g)\cdot(l + 1)$, and the expert output is `act @ W2[e] + b2[e]`.
- $\mathrm{MoE}(u)_t = \sum_j w_{t,j}\,\mathrm{expert}_{e_{t,j}}(u_t)$.

Parameter keys: `attn_norm`, `mlp_norm` `(D,)`, `W_qkv`, `b_qkv`, `W_o`, `b_o`, `sinks` `(H,)`, `rope_base`, `W_router` `(D, E)`, `b_router` `(E,)`, `W1` `(E, D, 2F)`, `b1` `(E, 2F)`, `W2` `(E, F, D)`, `b2` `(E, D)`. Inputs have no exact ties among router logits.
"""

STARTER = '''import numpy as np

def gpt_oss_block(x, p, n_heads, n_kv_heads, window, k):
    """Return the (T, D) block output."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _rms(z, w, eps=1e-5):
    return w * z / np.sqrt((z * z).mean(-1, keepdims=True) + eps)

def _rope(x, base):
    T, d = x.shape[-2], x.shape[-1]
    ang = np.arange(T)[:, None] * base ** (-np.arange(0, d, 2) / d)
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

def _attn(z, p, H, Hkv, window):
    T = z.shape[0]
    d = p['W_o'].shape[0] // H
    qkv = z @ p['W_qkv'] + p['b_qkv']
    q = qkv[:, : H * d].reshape(T, H, d).transpose(1, 0, 2)
    k = qkv[:, H * d:(H + Hkv) * d].reshape(T, Hkv, d).transpose(1, 0, 2)
    v = qkv[:, (H + Hkv) * d:].reshape(T, Hkv, d).transpose(1, 0, 2)
    q, k = _rope(q, p['rope_base']), _rope(k, p['rope_base'])
    k, v = np.repeat(k, H // Hkv, axis=0), np.repeat(v, H // Hkv, axis=0)
    t, j = np.arange(T)[:, None], np.arange(T)[None, :]
    ok = j <= t
    if window is not None:
        ok &= j > t - window
    S = np.where(ok, q @ k.transpose(0, 2, 1) / np.sqrt(d), -np.inf)
    full = np.concatenate([S, np.broadcast_to(np.asarray(p['sinks'], dtype=float)[:, None, None], (H, T, 1))], axis=-1)
    w = np.exp(full - full.max(-1, keepdims=True))
    w = w[..., :T] / w.sum(-1, keepdims=True)
    return (w @ v).transpose(1, 0, 2).reshape(T, H * d) @ p['W_o'] + p['b_o']

def _expert(u, p, e):
    a = u @ p['W1'][e] + p['b1'][e]
    g = np.minimum(a[:, 0::2], 7.0)
    l = np.clip(a[:, 1::2], -7.0, 7.0)
    act = g / (1 + np.exp(-1.702 * g)) * (l + 1)
    return act @ p['W2'][e] + p['b2'][e]

def _moe(u, p, k):
    logits = u @ p['W_router'] + p['b_router']
    idx = np.argsort(-logits, axis=1, kind='stable')[:, :k]
    top = np.take_along_axis(logits, idx, axis=1)
    w = np.exp(top - top.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    out = np.zeros_like(u)
    for e in range(p['W1'].shape[0]):
        ge = (w * (idx == e)).sum(1)
        rows = ge > 0
        if rows.any():
            out[rows] += ge[rows, None] * _expert(u[rows], p, e)
    return out

def gpt_oss_block(x, p, n_heads, n_kv_heads, window, k):
    x = np.asarray(x, dtype=float)
    h = x + _attn(_rms(x, p['attn_norm']), p, n_heads, n_kv_heads, window)
    return h + _moe(_rms(h, p['mlp_norm']), p, k)
'''

HINTS = [
    'Slice the fused `qkv` by widths $Hd$, $H_{kv}d$, $H_{kv}d$, split into heads, apply RoPE to $Q$ and $K$, then repeat the KV heads.',
    'For the sinks, append one column holding `sinks[h]` to each head\'s score rows, softmax over $T+1$ columns, and keep the first $T$.',
    'Router weights come from a softmax over just the $k$ selected logits. In each expert, the gate and linear halves are interleaved columns, and the gate is only clamped from above.',
]

_r = np.random.default_rng(2025)
def _model(D, H, Hkv, d, E, F):
    n = lambda *s: _r.normal(size=s)
    return dict(attn_norm=1 + 0.1 * n(D), mlp_norm=1 + 0.1 * n(D), W_qkv=n(D, (H + 2 * Hkv) * d) / np.sqrt(D),
                b_qkv=0.1 * n((H + 2 * Hkv) * d), W_o=n(H * d, D) / np.sqrt(H * d), b_o=0.1 * n(D), sinks=n(H),
                rope_base=10000.0, W_router=n(D, E) / np.sqrt(D), b_router=0.1 * n(E), W1=n(E, D, 2 * F) * 2,
                b1=0.5 * n(E, 2 * F), W2=n(E, F, D) / np.sqrt(F), b2=0.1 * n(E, D))

_m1 = _model(4, 2, 1, 2, 3, 4)
_m2 = _model(8, 4, 2, 2, 4, 6)
_x2 = _r.normal(size=(6, 8))
TESTS = [
    (_r.normal(size=(3, 4)), _m1, 2, 1, None, 1),
    (_r.normal(size=(4, 4)), _m1, 2, 1, 2, 2),
    (_x2, _m2, 4, 2, None, 2),
    (_x2, _m2, 4, 2, 3, 2),
    (_x2 * 4, _m2, 4, 2, 1, 4),
    (_x2[:2], _m2, 4, 2, 2, 1),
]

EXPLANATION = r"""
## How the pieces fit
- **Alternating banded and dense layers** halve the long-context attention cost and KV memory compared with all-dense, while dense layers keep long-range access. gpt-oss uses a 128-token band.
- **Sinks** matter most in the banded layers: without them a head that has nothing useful in its window must still spread weight over it.
- **Clamped SwiGLU**: bounding the gate and linear branches keeps activations in a range that low-precision MoE kernels handle well; the $+1$ on the linear branch and the $1.702$ inside the sigmoid (the sigmoid approximation of GELU) are specific to this model.
- **Softmax over the chosen logits** (rather than over all experts) makes the $k$ weights sum to 1 regardless of how much mass the other experts would have had.

## Complexity
Dense layers: $O(T^2 H d)$; banded: $O(T \cdot \text{window} \cdot H d)$ with a proper kernel. Each token runs $k$ experts.

## Common bugs
- Splitting gate and linear as the first and second half of the columns instead of interleaved.
- Clamping the gate from below as well (it is only clamped above).
- Taking the softmax over all experts and then picking the top $k$ without renormalising.
- Forgetting the biases on the attention projections and router; gpt-oss uses them.

## Interview follow-ups
- Where would MXFP4 dequantization happen in this block, and what stays in bf16?
- How does the KV cache of a banded layer differ from a dense one in a paged-attention server?
"""
