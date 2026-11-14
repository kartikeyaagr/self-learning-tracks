import numpy as np

META = dict(title='Dense prefix layers, then MoE layers (GLM-4.5 style)', tags=['Transformers', 'Language Models'], diff='Hard',
            runtime='numpy', lessons=['aieng:06.02', 'aieng:04.03'],
            related=['aux-loss-free-moe-routing', 'rotary-query-and-key-embeddings', 'qk-norm-vs-logit-softcap',
                     'causal-grouped-query-attention'])

STATEMENT = r"""
GLM-4.5 is a deep, relatively narrow MoE decoder. Its first few layers use a dense feed-forward network and the rest use mixture-of-experts with a shared expert and sigmoid routing. Its attention is GQA with **QK-norm** and **partial RoPE**: only the first part of each head vector is rotated. Build a tiny stack and return next-token logits.

Write `glm_logits(ids, p, n_heads, n_kv_heads, k, rotary_frac)` returning `(T, V)`.

## Layers
`p['layers'][i]` is dense if it has key `'ffn'` and MoE if it has key `'moe'`. Every layer:
$h = x + \mathrm{Attn}(\mathrm{RMS}_{in}(x))$, $x = h + \mathrm{FF}(\mathrm{RMS}_{post}(h))$ with $\mathrm{RMS}_w(z) = w\odot z/\sqrt{\mathrm{mean}(z^2) + 10^{-5}}$.

## Attention
- `Wq` `(D, H·d)`, `Wk`, `Wv` `(D, H_kv·d)`, `Wo` `(H·d, D)`, no biases. Heads are contiguous column blocks.
- **QK-norm**: RMSNorm of every query and key head vector with weights `q_norm`, `k_norm` `(d,)` (same formula as above).
- **Partial RoPE**: the first $r = $ `int(d * rotary_frac)` dims of each normalised head vector are rotated (rotate-half layout within those $r$ dims, $\theta_i = 10000^{-2i/r}$, positions from 0); the remaining $d - r$ dims pass through unchanged. $r$ is even.
- GQA head mapping $h \to \lfloor h/(H/H_{kv})\rfloor$, causal mask, $1/\sqrt d$ scaling.

## Feed-forward
- SwiGLU MLP: $\mathrm{F}(u) = (\mathrm{SiLU}(uW_1)\odot uW_3)W_2$ for a dict with `W1`, `W3`, `W2`.
- Dense layer: $\mathrm{FF} = \mathrm{F}_{ffn}$.
- MoE layer (`moe` dict with `shared`, `experts`, `W_r` `(D, E)`, `bias` `(E,)`): $s = \sigma(uW_r)$, choose top `k` by $s + \text{bias}$, gates are the chosen $s$ normalised to sum 1, and $\mathrm{FF}(u) = \mathrm{F}_{shared}(u) + \sum_j g_j\,\mathrm{F}_{e_j}(u)$.

## Ends
$x_0 = \text{emb}[ids]$; logits $= \mathrm{RMS}_{final}(x)\,W_{out}$. Keys: `emb`, `final_norm`, `W_out`, `layers`.
"""

STARTER = '''import numpy as np

def glm_logits(ids, p, n_heads, n_kv_heads, k, rotary_frac):
    """Return (T, V) logits."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _rms(z, w, eps=1e-5):
    return w * z / np.sqrt((z * z).mean(-1, keepdims=True) + eps)

def _partial_rope(x, r):
    if r == 0:
        return x
    T = x.shape[-2]
    ang = np.arange(T)[:, None] * 10000.0 ** (-np.arange(0, r, 2) / r)
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[..., : r // 2], x[..., r // 2: r]
    return np.concatenate([a * c - b * s, b * c + a * s, x[..., r:]], axis=-1)

def _swiglu(u, e):
    g = u @ e['W1']
    return ((g / (1 + np.exp(-g))) * (u @ e['W3'])) @ e['W2']

def _attn(z, L, H, Hkv, frac):
    T = z.shape[0]
    d = L['Wq'].shape[1] // H
    r = int(d * frac)
    q = (z @ L['Wq']).reshape(T, H, d).transpose(1, 0, 2)
    k = (z @ L['Wk']).reshape(T, Hkv, d).transpose(1, 0, 2)
    v = (z @ L['Wv']).reshape(T, Hkv, d).transpose(1, 0, 2)
    q, k = _partial_rope(_rms(q, L['q_norm']), r), _partial_rope(_rms(k, L['k_norm']), r)
    k, v = np.repeat(k, H // Hkv, axis=0), np.repeat(v, H // Hkv, axis=0)
    S = np.where(np.tril(np.ones((T, T), dtype=bool)), q @ k.transpose(0, 2, 1) / np.sqrt(d), -np.inf)
    A = np.exp(S - S.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    return (A @ v).transpose(1, 0, 2).reshape(T, H * d) @ L['Wo']

def _moe(u, m, k):
    s = 1 / (1 + np.exp(-(u @ m['W_r'])))
    idx = np.argsort(-(s + m['bias']), axis=1, kind='stable')[:, :k]
    g = np.take_along_axis(s, idx, axis=1)
    g = g / g.sum(1, keepdims=True)
    out = _swiglu(u, m['shared'])
    for e, ex in enumerate(m['experts']):
        w = (g * (idx == e)).sum(1)
        rows = w > 0
        if rows.any():
            out[rows] += w[rows, None] * _swiglu(u[rows], ex)
    return out

def glm_logits(ids, p, n_heads, n_kv_heads, k, rotary_frac):
    x = np.asarray(p['emb'], dtype=float)[np.asarray(ids)]
    for L in p['layers']:
        h = x + _attn(_rms(x, L['in_norm']), L, n_heads, n_kv_heads, rotary_frac)
        u = _rms(h, L['post_norm'])
        x = h + (_swiglu(u, L['ffn']) if 'ffn' in L else _moe(u, L['moe'], k))
    return _rms(x, p['final_norm']) @ p['W_out']
'''

HINTS = [
    'Partial RoPE: rotate `x[..., :r]` with the usual rotate-half formula (frequencies computed from $r$, not $d$) and concatenate `x[..., r:]` back unchanged.',
    'Order inside attention: project, split heads, QK-norm, partial RoPE, then scores. The layer type only changes the feed-forward call.',
    'The MoE gates use the sigmoid scores without the bias; the bias only decides which experts are chosen.',
]

_r = np.random.default_rng(2507)
def _mlp(D, F):
    return dict(W1=_r.normal(size=(D, F)) / np.sqrt(D), W3=_r.normal(size=(D, F)) / np.sqrt(D), W2=_r.normal(size=(F, D)) / np.sqrt(F))
def _layer(D, H, Hkv, d, F, moe_E):
    L = dict(in_norm=1 + 0.1 * _r.normal(size=D), post_norm=1 + 0.1 * _r.normal(size=D), q_norm=1 + 0.1 * _r.normal(size=d),
             k_norm=1 + 0.1 * _r.normal(size=d), Wq=_r.normal(size=(D, H * d)) / np.sqrt(D),
             Wk=_r.normal(size=(D, Hkv * d)) / np.sqrt(D), Wv=_r.normal(size=(D, Hkv * d)) / np.sqrt(D),
             Wo=_r.normal(size=(H * d, D)) / np.sqrt(H * d))
    if moe_E:
        L['moe'] = dict(shared=_mlp(D, F), experts=[_mlp(D, F) for _ in range(moe_E)],
                        W_r=_r.normal(size=(D, moe_E)) / np.sqrt(D), bias=0.1 * _r.normal(size=moe_E))
    else:
        L['ffn'] = _mlp(D, 2 * F)
    return L
def _model(V, D, H, Hkv, d, F, n_dense, n_moe, E):
    return dict(emb=_r.normal(size=(V, D)), final_norm=1 + 0.1 * _r.normal(size=D), W_out=_r.normal(size=(D, V)) / np.sqrt(D),
                layers=[_layer(D, H, Hkv, d, F, 0) for _ in range(n_dense)] + [_layer(D, H, Hkv, d, F, E) for _ in range(n_moe)])

_m1 = _model(9, 4, 2, 1, 4, 4, 1, 1, 3)
_m2 = _model(13, 8, 4, 2, 4, 4, 1, 2, 4)
TESTS = [
    (np.array([1, 2, 3]), _m1, 2, 1, 1, 0.5),
    (np.array([1, 2, 3]), _m1, 2, 1, 2, 1.0),
    (np.array([4, 8, 1, 0, 6]), _m2, 4, 2, 2, 0.5),
    (np.array([4, 8, 1, 0, 6]), _m2, 4, 2, 2, 0.0),
    (np.array([12, 3, 3, 7, 9, 2, 5]), _m2, 4, 2, 3, 0.5),
    (np.array([11]), _m2, 4, 2, 1, 1.0),
]

EXPLANATION = r"""
## Design choices this stack reproduces
- **Dense prefix**: the first layers of a deep MoE model see poorly separated token representations, so routing there is noisy and load balancing is hard. GLM-4.5 (like DeepSeek-V3) keeps the first few layers dense.
- **Depth over width**: GLM-4.5 chose more layers and fewer, wider attention heads than comparable MoE models, reporting better reasoning results for the same active parameters.
- **Partial RoPE**: rotating only part of each head leaves some dimensions position-free, which can carry content matching independent of distance. Earlier GLM models also rotated only part of each head.
- **QK-norm** stabilises attention logits; **sigmoid routing with bias balancing** follows DeepSeek-V3.

## Complexity
Attention $O(T^2 H d)$ per layer; dense layers run one wide MLP, MoE layers run one shared plus $k$ routed experts per token.

## Common bugs
- Computing partial-RoPE frequencies with $d$ instead of $r$ in the exponent.
- Rotating the last $r$ dims instead of the first.
- Applying RoPE before QK-norm.

## Interview follow-ups
- In a 90-layer model with 3 dense layers, where do most of the total parameters sit, and where do most of the active ones?
- What would an MTP head (as in the DeepSeek-V3 build) add to this model at inference?
"""
