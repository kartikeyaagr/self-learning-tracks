import numpy as np

META = dict(title='Grouped-query cross-attention with key padding', tags=['Attention', 'Transformers'], diff='Hard', runtime='numpy',
            lessons=['aieng:05.02', 'ieng:02.02'],
            related=['scaled-dot-product-attention-2', 'scaled-dot-product-attention', 'transformer-decoder-block'])

STATEMENT = r"""
Grouped-query attention (GQA) keeps $H$ query heads but only $H_{kv}$ key/value heads; each group of $H / H_{kv}$ consecutive query heads shares one key/value head. It shrinks the KV cache by a factor $H / H_{kv}$. With $H_{kv} = H$ it is ordinary multi-head attention, with $H_{kv} = 1$ it is multi-query attention.

Write `gqa(Xq, Xkv, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, kv_mask=None)` for **cross**-attention (queries and keys come from different sequences).

## Shapes
- `Xq`: `(B, T_q, D)` queries' source; `Xkv`: `(B, T_k, D)` keys' and values' source.
- $H$ = `n_heads` divides $D$; head width $d_h = D / H$. $H_{kv}$ = `n_kv_heads` divides $H$.
- `Wq`: `(D, D)`, `Wk` and `Wv`: `(D, H_kv * d_h)`, `Wo`: `(D, D)`. No biases.
- `kv_mask`: optional bool `(B, T_k)`, `True` for real tokens, `False` for padding. `None` means all real.

## Computation
1. $Q = X_q W_q$, $K = X_{kv} W_k$, $V = X_{kv} W_v$. Query head $h$ is columns $h d_h : (h+1) d_h$ of $Q$; KV head $g$ is columns $g d_h : (g+1) d_h$ of $K$ and $V$.
2. Query head $h$ uses KV head $g = \lfloor h / (H / H_{kv}) \rfloor$.
3. Per head: scores $Q_h K_g^\top / \sqrt{d_h}$, padded keys set to $-\infty$, stable softmax over keys, times $V_g$.
4. If a sample has **no** real keys, its attention output is all zeros for every head (not `nan`).
5. Concatenate heads in order to `(B, T_q, D)` and multiply by $W_o$.

## Output
Float array `(B, T_q, D)`. No Python loops over batch, heads or positions.
"""

STARTER = '''import numpy as np

def gqa(Xq, Xkv, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, kv_mask=None):
    """Return (B, Tq, D) grouped-query cross-attention output."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gqa(Xq, Xkv, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, kv_mask=None):
    Xq, Xkv = np.asarray(Xq, dtype=float), np.asarray(Xkv, dtype=float)
    B, Tq, D = Xq.shape
    Tk = Xkv.shape[1]
    H, G = n_heads, n_kv_heads
    dh = D // H
    Q = (Xq @ Wq).reshape(B, Tq, H, dh).transpose(0, 2, 1, 3)       # (B, H, Tq, dh)
    K = (Xkv @ Wk).reshape(B, Tk, G, dh).transpose(0, 2, 1, 3)      # (B, G, Tk, dh)
    V = (Xkv @ Wv).reshape(B, Tk, G, dh).transpose(0, 2, 1, 3)
    K, V = np.repeat(K, H // G, axis=1), np.repeat(V, H // G, axis=1)   # head h -> kv head h // (H/G)
    S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)                     # (B, H, Tq, Tk)
    if kv_mask is not None:
        S = np.where(np.asarray(kv_mask)[:, None, None, :], S, -np.inf)
    m = S.max(axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)                             # all-padding rows
    w = np.exp(S - m)
    tot = w.sum(axis=-1, keepdims=True)
    A = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
    O = (A @ V).transpose(0, 2, 1, 3).reshape(B, Tq, D)
    return O @ Wo
'''

HINTS = [
    'Project once, then reshape to heads: `(B, T, H, d_h)` transposed to `(B, H, T, d_h)` for queries, and `(B, H_kv, T_k, d_h)` for keys and values.',
    '`np.repeat(K, H // H_kv, axis=1)` duplicates each KV head for its group of consecutive query heads (repeat, not tile: head order must be `g0, g0, g1, g1, ...`). From there it is ordinary batched attention.',
    'Mask with `np.where(kv_mask[:, None, None, :], S, -inf)`. Before exponentiating, replace a non-finite row max by 0, and divide only where the row sum is positive so all-padding samples give zeros.',
]

_r = np.random.default_rng(89)
def _w(D, H, G):
    dh = D // H
    s = 1 / np.sqrt(D)
    return _r.normal(size=(D, D)) * s, _r.normal(size=(D, G * dh)) * s, _r.normal(size=(D, G * dh)) * s, _r.normal(size=(D, D)) * s
_I = np.eye(2)
TESTS = [
    (np.array([[[1.0, 0.0]]]), np.array([[[1.0, 0.0], [0.0, 1.0]]]), _I, _I, _I, _I, 1, 1),
    (np.array([[[1.0, 0.0]]]), np.array([[[1.0, 0.0], [0.0, 1.0]]]), _I, _I[:, :1], _I[:, :1], _I, 2, 1, np.array([[True, False]])),
    (_r.normal(size=(2, 3, 8)), _r.normal(size=(2, 5, 8)), *_w(8, 4, 2), 4, 2),
    (_r.normal(size=(2, 3, 8)), _r.normal(size=(2, 5, 8)), *_w(8, 4, 1), 4, 1, np.array([[1, 1, 1, 0, 0], [1, 0, 1, 1, 0]], dtype=bool)),
    (_r.normal(size=(2, 4, 6)), _r.normal(size=(2, 4, 6)), *_w(6, 3, 3), 3, 3),
    (_r.normal(size=(2, 2, 8)), _r.normal(size=(2, 3, 8)), *_w(8, 2, 2), 2, 2, np.array([[False, False, False], [True, True, False]])),
    (_r.normal(size=(1, 3, 12)) * 4, _r.normal(size=(1, 6, 12)) * 4, *_w(12, 6, 2), 6, 2),
]

EXPLANATION = r"""
## Why share keys and values
At decode time each new token must read the cached $K$ and $V$ of every previous token for every layer: $2 \cdot L \cdot T \cdot H_{kv} d_h$ numbers. Memory bandwidth, not FLOPs, limits generation speed, so cutting $H_{kv}$ from 32 to 8 (Llama-2 70B) cuts the cache and the bytes read per token by 4x, with little quality loss. Queries are not cached, so keeping many query heads is cheap.

## The head mapping
Consecutive query heads share a KV head: with $H = 4$, $H_{kv} = 2$ the mapping is $0, 0, 1, 1$. `np.repeat` produces exactly that; `np.tile` would give $0, 1, 0, 1$ and silently pair the wrong heads. Production kernels avoid materialising the repeated tensor by reshaping queries to `(B, H_kv, H/H_kv, T, d_h)` instead.

## Cross-attention and padding
In an encoder-decoder model the keys come from the encoder, so the mask is over **key** positions: padded source tokens must receive zero weight. A sample whose source is entirely padding (test 6) has nothing to attend to; returning zeros keeps `nan` from spreading through the residual stream.

## Complexity
$O(B H T_q T_k d_h)$ for the scores, the same as MHA; the saving is in the KV projections and cache, $H_{kv}/H$ of MHA's.

## Interview follow-ups
- How large is the KV cache of a 32-layer model with $H_{kv} = 8$, $d_h = 128$, a 32k context and fp16 values?
- How would you convert a pretrained MHA checkpoint to GQA? (Mean-pool the K/V projection heads within each group, then fine-tune briefly.)
"""
