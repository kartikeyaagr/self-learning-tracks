import numpy as np

META = dict(title='Interleaved local and global attention layers (Gemma 3 style)', tags=['Transformers', 'Attention'],
            diff='Hard', runtime='numpy', lessons=['aieng:07.02', 'aieng:04.03'],
            related=['qk-norm-vs-logit-softcap', 'causal-grouped-query-attention', 'rotary-position-embedding-rope',
                     'norm-gated-linear-transform'])

STATEMENT = r"""
Gemma 3 keeps long contexts affordable by making most layers **local**: they attend only within a short sliding window, so their KV cache stays small. Every few layers a **global** layer attends to the whole prefix. Local and global layers also use different RoPE bases. Build a tiny stack with this pattern.

Write `gemma_logits(ids, p, n_heads, n_kv_heads, window, pattern)` returning `(T, V)` logits.

## Layer schedule
Layer $i$ (0-based) is **global** if $(i+1)$ is a multiple of `pattern`, otherwise **local**. With `pattern = 6` that is five local layers per global one.
- Local: query $t$ may attend to keys $s$ with $t - \text{window} < s \le t$; RoPE base $10^4$.
- Global: ordinary causal mask; RoPE base $10^6$.

## Everything else
- Norm: $\mathrm{RMSNorm}_w(x) = (1 + w) \odot x / \sqrt{\mathrm{mean}(x^2) + 10^{-6}}$ (zero-centred weights).
- Embedding: $x = \sqrt{D}\cdot \text{emb}[ids]$. Output: $\mathrm{RMSNorm}_{final}(x)\,\text{emb}^\top$ (tied).
- Layer (sandwich norms around both sublayers):
  $x \leftarrow x + \mathrm{N}_{post\_attn}(\mathrm{Attn}(\mathrm{N}_{pre\_attn}(x)))$, then $x \leftarrow x + \mathrm{N}_{post\_ffn}(\mathrm{FFN}(\mathrm{N}_{pre\_ffn}(x)))$.
- Attn: `Wq` `(D, H·d)`, `Wk`, `Wv` `(D, H_kv·d)`, `Wo` `(H·d, D)`, $d$ given by `Wq.shape[1] // H`. Split heads (contiguous blocks), apply **QK-norm** (RMSNorm with weights `q_norm`, `k_norm` of shape `(d,)`, zero-centred) to each head vector, then RoPE, then scores scaled by $1/\sqrt{d}$. Query head $h$ uses KV head $\lfloor h/(H/H_{kv})\rfloor$.
- RoPE, "rotate-half" layout: with $x = [x_1, x_2]$ split into halves of size $d/2$ and angles $t\,\theta_i$, $\theta_i = \text{base}^{-2i/d}$: $x \mapsto [x_1\cos - x_2\sin,\ x_2\cos + x_1\sin]$.
- FFN (GeGLU): $(\mathrm{GELU}(uW_{gate}) \odot uW_{up})\,W_{down}$, tanh-approximate GELU.

Layer dict keys: `pre_attn`, `post_attn`, `pre_ffn`, `post_ffn` `(D,)`, `q_norm`, `k_norm` `(d,)`, `Wq Wk Wv Wo Wgate Wup Wdown`. Top level: `emb` `(V, D)`, `final_norm` `(D,)`, `layers`.
"""

STARTER = '''import numpy as np

def gemma_logits(ids, p, n_heads, n_kv_heads, window, pattern):
    """Return (T, V) logits."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _rms(x, w, eps=1e-6):
    return (1.0 + w) * x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)

def _rope(x, base):
    T, d = x.shape[-2], x.shape[-1]
    ang = np.arange(T)[:, None] * base ** (-np.arange(0, d, 2) / d)
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)

def _gelu(u):
    return 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))

def _attn(x, L, H, Hkv, allowed, base):
    T = x.shape[0]
    d = L['Wq'].shape[1] // H
    q = (x @ L['Wq']).reshape(T, H, d).transpose(1, 0, 2)
    k = (x @ L['Wk']).reshape(T, Hkv, d).transpose(1, 0, 2)
    v = (x @ L['Wv']).reshape(T, Hkv, d).transpose(1, 0, 2)
    q, k = _rope(_rms(q, L['q_norm']), base), _rope(_rms(k, L['k_norm']), base)
    k, v = np.repeat(k, H // Hkv, axis=0), np.repeat(v, H // Hkv, axis=0)
    S = np.where(allowed, q @ k.transpose(0, 2, 1) / np.sqrt(d), -np.inf)
    A = np.exp(S - S.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    return (A @ v).transpose(1, 0, 2).reshape(T, H * d) @ L['Wo']

def gemma_logits(ids, p, n_heads, n_kv_heads, window, pattern):
    E = np.asarray(p['emb'], dtype=float)
    x = np.sqrt(E.shape[1]) * E[np.asarray(ids)]
    T = x.shape[0]
    t, s = np.arange(T)[:, None], np.arange(T)[None, :]
    causal = s <= t
    local = causal & (s > t - window)
    for i, L in enumerate(p['layers']):
        is_global = (i + 1) % pattern == 0
        allowed, base = (causal, 1e6) if is_global else (local, 1e4)
        x = x + _rms(_attn(_rms(x, L['pre_attn']), L, n_heads, n_kv_heads, allowed, base), L['post_attn'])
        u = _rms(x, L['pre_ffn'])
        x = x + _rms((_gelu(u @ L['Wgate']) * (u @ L['Wup'])) @ L['Wdown'], L['post_ffn'])
    return _rms(x, p['final_norm']) @ E.T
'''

HINTS = [
    'Build two boolean masks once: `causal = s <= t` and `local = causal & (s > t - window)`. Each layer picks one, plus its RoPE base, from its index.',
    'Order inside attention: project, split heads, QK-norm each head vector, rotate with RoPE, then the dot product.',
    'Sandwich norm means the sublayer output is normalised again before it is added to the residual stream; the residual itself is never normalised.',
]

_r = np.random.default_rng(2503)
def _model(V, D, H, Hkv, d, F, nl):
    s = 1 / np.sqrt(D)
    w = lambda n: 0.1 * _r.normal(size=n)
    layers = [dict(pre_attn=w(D), post_attn=w(D), pre_ffn=w(D), post_ffn=w(D), q_norm=w(d), k_norm=w(d),
                   Wq=_r.normal(size=(D, H * d)) * s, Wk=_r.normal(size=(D, Hkv * d)) * s, Wv=_r.normal(size=(D, Hkv * d)) * s,
                   Wo=_r.normal(size=(H * d, D)) / np.sqrt(H * d), Wgate=_r.normal(size=(D, F)) * s,
                   Wup=_r.normal(size=(D, F)) * s, Wdown=_r.normal(size=(F, D)) / np.sqrt(F)) for _ in range(nl)]
    return {'emb': _r.normal(size=(V, D)) * s, 'final_norm': w(D), 'layers': layers}

_m1 = _model(10, 4, 2, 1, 2, 8, 2)
_m2 = _model(12, 8, 4, 2, 4, 16, 3)
_ids = np.array([3, 1, 4, 1, 5, 9, 2, 6])
TESTS = [
    (np.array([1, 2, 3, 4]), _m1, 2, 1, 2, 2),
    (np.array([1, 2, 3, 4]), _m1, 2, 1, 4, 2),
    (_ids, _m2, 4, 2, 3, 3),
    (_ids, _m2, 4, 2, 3, 1),
    (_ids, _m2, 4, 2, 2, 6),
    (_ids[:5], _m2, 4, 2, 1, 2),
]

EXPLANATION = r"""
## Why mostly-local layers
A global layer's KV cache grows with the full context; a local layer's is capped at the window. With five local layers (window 1024) per global layer, a 128k-token context needs full-length KV only in one layer of six, which is where most of Gemma 3's memory saving comes from. Information still travels beyond the window: each local layer extends the receptive field by `window` tokens, and global layers connect everything directly.

## Different RoPE bases
Local layers never see distances beyond the window, so a short base (10k) gives them fine positional resolution. Global layers use a much larger base (1M) so their slowest frequencies don't wrap around within the long context.

## The rest of the block
- **QK-norm** replaced Gemma 2's attention soft-capping.
- **Sandwich norms** (pre- and post-sublayer RMSNorm) keep the residual stream's scale controlled in deep stacks.
- **$\sqrt{D}$ embedding scale** with tied output weights, as in the earlier Gemma models.

## Complexity
Local layers cost $O(T \cdot \text{window} \cdot d)$ per head (this dense version still computes and masks $T^2$ scores; a real kernel skips them). Global layers cost $O(T^2 d)$.

## Common bugs
- Off-by-one in the window: with `window = w` a query sees itself and the $w - 1$ previous tokens.
- Choosing global layers with `i % pattern == 0`, which makes layer 0 global instead of the last of each group.
- Applying QK-norm after RoPE; RMS normalisation is not rotation-invariant per pair weight, so the order matters with non-zero weights.

## Interview follow-ups
- How much KV memory does a 5:1 local/global pattern save at 32k tokens with window 1024?
- What breaks if you run a model trained with window 1024 at window 512?
"""
