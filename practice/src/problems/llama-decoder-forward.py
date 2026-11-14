import numpy as np

META = dict(title='Forward pass of a tiny LLaMA-style decoder', tags=['Transformers', 'Language Models'], diff='Hard',
            runtime='numpy', lessons=['air-fanout-v1:08.01', 'aieng:04.03'],
            related=['rotary-position-embedding-rope', 'norm-gated-linear-transform', 'parameter-matched-swiglu-block',
                     'causal-grouped-query-attention'])

STATEMENT = r"""
LLaMA kept the GPT recipe (decoder-only, pre-norm, causal) and swapped in three components that are now standard: **RMSNorm** instead of LayerNorm, **rotary position embeddings** instead of a position table, and a **SwiGLU** feed-forward network. Later versions added **grouped-query attention**. Put all four together.

Write `llama_logits(ids, p, n_heads, n_kv_heads, rope_base=10000.0)` returning `(T, V)` logits for one sequence.

## Parameters
- `p['emb']` `(V, D)`, `p['norm']` `(D,)`, `p['out']` `(D, V)` (not tied).
- `p['layers']`: list of dicts with `attn_norm`, `ffn_norm` `(D,)`, `Wq` `(D, H·d)`, `Wk`, `Wv` `(D, H_kv·d)`, `Wo` `(H·d, D)`, `W1`, `W3` `(D, F)`, `W2` `(F, D)`. No biases anywhere. $d = D/H$.

## Model
- $\mathrm{RMSNorm}(x) = w \odot x / \sqrt{\mathrm{mean}(x^2) + 10^{-6}}$ over the last axis.
- Each layer: $h = x + \mathrm{Attn}(\mathrm{RMSNorm}(x))$, then $x = h + \mathrm{FFN}(\mathrm{RMSNorm}(h))$ with $\mathrm{FFN}(u) = (\mathrm{SiLU}(uW_1) \odot uW_3)\,W_2$.
- **Attention**: $Q$ has $H$ heads, $K, V$ have $H_{kv}$ heads (contiguous column blocks of width $d$). Query head $h$ uses KV head $\lfloor h / (H/H_{kv}) \rfloor$. Causal mask, $1/\sqrt{d}$ scaling, heads concatenated before $W_o$.
- **RoPE** on $Q$ and $K$ per head, interleaved pairs: dims $(2i, 2i{+}1)$ at position $t$ are rotated by angle $t\,\theta_i$ with $\theta_i = \text{base}^{-2i/d}$:
  $(a, b) \mapsto (a\cos - b\sin,\ a\sin + b\cos)$. Positions start at 0.
- Logits $= \mathrm{RMSNorm}_{final}(x)\,W_{out}$.
"""

STARTER = '''import numpy as np

def llama_logits(ids, p, n_heads, n_kv_heads, rope_base=10000.0):
    """Return (T, V) logits."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _rms(x, w, eps=1e-6):
    return w * x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)

def _rope(x, base):
    """x: (heads, T, d) with interleaved pairs."""
    T, d = x.shape[1], x.shape[2]
    ang = np.arange(T)[:, None] * base ** (-np.arange(0, d, 2) / d)     # (T, d/2)
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out

def _attn(x, L, H, Hkv, base):
    T, D = x.shape
    d = L['Wq'].shape[1] // H
    q = _rope((x @ L['Wq']).reshape(T, H, d).transpose(1, 0, 2), base)
    k = _rope((x @ L['Wk']).reshape(T, Hkv, d).transpose(1, 0, 2), base)
    v = (x @ L['Wv']).reshape(T, Hkv, d).transpose(1, 0, 2)
    rep = H // Hkv
    k, v = np.repeat(k, rep, axis=0), np.repeat(v, rep, axis=0)        # head h -> kv head h // rep
    S = q @ k.transpose(0, 2, 1) / np.sqrt(d)
    S = np.where(np.tril(np.ones((T, T), dtype=bool)), S, -np.inf)
    A = np.exp(S - S.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    return (A @ v).transpose(1, 0, 2).reshape(T, H * d) @ L['Wo']

def llama_logits(ids, p, n_heads, n_kv_heads, rope_base=10000.0):
    x = np.asarray(p['emb'], dtype=float)[np.asarray(ids)]
    for L in p['layers']:
        h = x + _attn(_rms(x, L['attn_norm']), L, n_heads, n_kv_heads, rope_base)
        u = _rms(h, L['ffn_norm'])
        g = u @ L['W1']
        x = h + ((g / (1 + np.exp(-g))) * (u @ L['W3'])) @ L['W2']
    return _rms(x, p['norm']) @ p['out']
'''

HINTS = [
    'RMSNorm has no mean subtraction and no bias: divide by the root-mean-square and multiply by the weight.',
    'Apply RoPE after splitting into heads and before the dot product, to both queries and keys, never to values. The angle for pair $i$ at position $t$ is $t \\cdot \\text{base}^{-2i/d}$.',
    'For GQA, `np.repeat(k, H // H_kv, axis=0)` on a `(H_kv, T, d)` array gives query head $h$ the KV head $h // (H/H_{kv})$.',
]

_r = np.random.default_rng(2302)
def _model(V, D, F, H, Hkv, nl):
    d = D // H
    s = 1 / np.sqrt(D)
    layers = [dict(attn_norm=1 + 0.1 * _r.normal(size=D), ffn_norm=1 + 0.1 * _r.normal(size=D),
                   Wq=_r.normal(size=(D, H * d)) * s, Wk=_r.normal(size=(D, Hkv * d)) * s, Wv=_r.normal(size=(D, Hkv * d)) * s,
                   Wo=_r.normal(size=(H * d, D)) * s, W1=_r.normal(size=(D, F)) * s, W3=_r.normal(size=(D, F)) * s,
                   W2=_r.normal(size=(F, D)) / np.sqrt(F)) for _ in range(nl)]
    return {'emb': _r.normal(size=(V, D)), 'norm': 1 + 0.1 * _r.normal(size=D), 'out': _r.normal(size=(D, V)) * s,
            'layers': layers}

_m1 = _model(8, 4, 8, 2, 2, 1)
_m2 = _model(12, 8, 16, 4, 2, 2)
_m3 = _model(10, 8, 12, 4, 1, 1)
TESTS = [
    (np.array([1, 2, 3]), _m1, 2, 2),
    (np.array([5, 5, 5, 5]), _m1, 2, 2),
    (np.array([3, 7, 1, 0, 9, 11]), _m2, 4, 2),
    (np.array([3, 7, 1, 0, 9, 11]), _m2, 4, 2, 500.0),
    (np.array([2, 4, 6, 8, 1]), _m3, 4, 1),
    (np.array([9]), _m3, 4, 1),
]

EXPLANATION = r"""
## Why each swap
- **RMSNorm** drops the mean subtraction and bias of LayerNorm. It is cheaper and, empirically, re-centring was not what made LayerNorm useful; re-scaling was.
- **RoPE** encodes position by rotating query and key pairs. The dot product of a query at $m$ and a key at $n$ then depends only on $m - n$, so attention sees relative positions without a learned table, and there is nothing to add to the embeddings.
- **SwiGLU** gates one linear projection with the SiLU of another. It costs a third matrix, so the hidden width is usually shrunk to about $\tfrac{2}{3} \cdot 4D$ to keep parameters equal to a 4x GELU MLP.
- **GQA** shares each K/V head across a group of query heads, cutting the KV cache by $H/H_{kv}$ with little quality loss.

## Two RoPE layouts
Meta's reference code rotates adjacent pairs $(2i, 2i{+}1)$ (complex multiplication); the Hugging Face port rotates $(i, i + d/2)$ ("rotate half") and permutes the projection weights to match. Mixing the two silently gives wrong outputs, which is why the layout is part of the spec.

## Complexity
Per layer: $O(T^2 D)$ attention plus $O(3TDF)$ for SwiGLU. The KV cache per token per layer is $2 H_{kv} d$ numbers instead of $2D$.

## Common bugs
- Rotating values, or applying RoPE before the head split with the wrong per-head dimension.
- Repeating KV heads with `np.tile` (gives head $h \bmod H_{kv}$) instead of `np.repeat`.
- Forgetting the final norm before the output projection.

## Interview follow-ups
- Show that $\langle R_m q, R_n k \rangle$ depends only on $m - n$.
- What is the KV cache size of a 32-layer model with $H_{kv} = 8$, $d = 128$ at 8k tokens in bf16?
"""
