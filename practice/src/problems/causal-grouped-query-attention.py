import numpy as np

META = dict(title='Causal grouped-query attention with padding and a sliding window', tags=['Attention', 'Transformers'],
            diff='Hard', runtime='torch', lessons=['aieng:05.02', 'ieng:02.02'],
            related=['multi-head-attention', 'scaled-dot-product-attention-2', 'rotary-query-and-key-embeddings'])

STATEMENT = r"""
A decoder self-attention layer as used in modern LLMs: grouped-query heads, a causal mask, right-padded sequences of different lengths, and optionally a sliding window that limits how far back each token looks.

Write `causal_gqa(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, lengths, window=None)` in PyTorch.

## Shapes
- `x`: `float64` `(B, T, D)`. $H$ = `n_heads` divides $D$; head width $d_h = D / H$. $H_{kv}$ = `n_kv_heads` divides $H$.
- `Wq`: `(D, H d_h)`, `Wk`, `Wv`: `(D, H_kv d_h)`, `Wo`: `(H d_h, D)`. No biases. Projections are `x @ W`.
- `lengths`: `int64` `(B,)`, $1 \le \ell_b \le T$: sequence $b$ has real tokens at positions $0, \dots, \ell_b - 1$; the rest is padding.
- `window`: `None` or an int $w \ge 1$.

## Computation
1. $Q = xW_q$, $K = xW_k$, $V = xW_v$, split into heads: feature block $h$ (columns $h d_h, \dots, (h+1)d_h - 1$) is head $h$. Query head $h$ uses key/value head $\lfloor h / (H/H_{kv}) \rfloor$.
2. Scores $S_{b,h,i,j} = Q_{b,h,i} \cdot K_{b,g(h),j} / \sqrt{d_h}$.
3. Query $i$ may attend to key $j$ only if $j \le i$, $j < \ell_b$, and (if `window` is set) $i - j < w$. Disallowed scores are excluded from the softmax.
4. Softmax over $j$, weighted sum of $V$, concatenate heads in order, multiply by `Wo`.
5. Rows of padding queries ($i \ge \ell_b$) are set to **exactly zero** in the output.

## Output
`float64` tensor `(B, T, D)`. Use tensor ops only (no Python loops over batch, heads or positions).

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import math
import torch

def causal_gqa(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, lengths, window=None):
    """Return the (B, T, D) output of causal GQA self-attention."""
    B, T, D = x.shape
    dh = D // n_heads
    q = (x @ Wq).view(B, T, n_heads, dh).transpose(1, 2)
    k = (x @ Wk).view(B, T, n_kv_heads, dh).transpose(1, 2)
    v = (x @ Wv).view(B, T, n_kv_heads, dh).transpose(1, 2)
    k = k.repeat(1, n_heads // n_kv_heads, 1, 1)      # TODO: wrong head order, and no masks at all
    v = v.repeat(1, n_heads // n_kv_heads, 1, 1)
    p = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
    return (p @ v).transpose(1, 2).reshape(B, T, D) @ Wo
'''

SOLUTION = '''import math
import torch

def causal_gqa(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, lengths, window=None):
    B, T, D = x.shape
    dh, G = D // n_heads, n_heads // n_kv_heads
    q = (x @ Wq).view(B, T, n_kv_heads, G, dh).permute(0, 2, 3, 1, 4)   # (B, Hkv, G, T, dh)
    k = (x @ Wk).view(B, T, n_kv_heads, dh).transpose(1, 2)[:, :, None]  # (B, Hkv, 1, T, dh)
    v = (x @ Wv).view(B, T, n_kv_heads, dh).transpose(1, 2)[:, :, None]
    s = q @ k.transpose(-1, -2) / math.sqrt(dh)                           # (B, Hkv, G, T, T)

    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    allowed = j <= i                                                      # causal
    if window is not None:
        allowed = allowed & (i - j < window)                              # sliding window
    allowed = allowed[None] & (j[None] < lengths[:, None, None])          # (B, T, T): no padded keys
    s = s.masked_fill(~allowed[:, None, None], float('-inf'))
    # A real query always keeps j = i. A padded query (i >= length) can lose every key when the window
    # no longer reaches a real token: give its row finite dummy scores (no NaN) and zero it afterwards.
    dead = ~allowed.any(-1)                                               # (B, T)
    s = s.masked_fill(dead[:, None, None, :, None], 0.0)
    p = torch.softmax(s, dim=-1)
    out = (p @ v).permute(0, 3, 1, 2, 4).reshape(B, T, D) @ Wo            # heads back in order h = g_kv*G + g
    real = torch.arange(T)[None, :] < lengths[:, None]                    # (B, T) real query rows
    return out * real[..., None].to(out.dtype)
'''

HINTS = [
    'Reshape the projected queries to `(B, H_kv, G, T, d_h)` with $G = H/H_{kv}$: head $h = g_{kv} G + g$, so consecutive query heads share a key head. Keys get a singleton group axis `(B, H_kv, 1, T, d_h)` and broadcast.',
    'Build one boolean `(B, T, T)` mask: `j <= i`, `i - j < window`, `j < lengths[b]`. Fill disallowed scores with `-inf` before the softmax.',
    'A padded query row can end up with every key masked, and a softmax over all `-inf` is NaN. Replace such rows with any finite scores, then zero padded query rows in the final output (multiplying by 0 does not remove NaN).',
]

_r = np.random.default_rng(8)


def _case(B, T, D, H, Hkv, s=0.4):
    dh = D // H
    return (_r.normal(size=(B, T, D)), _r.normal(size=(D, H * dh)) * s, _r.normal(size=(D, Hkv * dh)) * s,
            _r.normal(size=(D, Hkv * dh)) * s, _r.normal(size=(H * dh, D)) * s)


_one = (np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]), np.eye(2), np.eye(2), np.eye(2), np.eye(2))
TESTS = [
    (*_one, 1, 1, np.array([3])),
    (*_one, 1, 1, np.array([2]), 1),
    (*_case(2, 5, 8, 4, 2), 4, 2, np.array([5, 3])),
    (*_case(2, 5, 8, 4, 2), 4, 2, np.array([5, 3]), 2),
    (*_case(1, 6, 12, 6, 1), 6, 1, np.array([6]), 3),
    (*_case(3, 4, 8, 2, 2), 2, 2, np.array([1, 4, 2])),
    (*_case(2, 9, 16, 8, 4), 8, 4, np.array([9, 6]), 4),
    (*_case(1, 7, 8, 4, 4), 4, 4, np.array([7]), 100),
]

EXPLANATION = r"""
## Why it works
GQA is ordinary multi-head attention where each key/value head is shared by a group of $G$ consecutive query heads. Viewing queries as `(B, H_kv, G, T, d_h)` lets one broadcasted matmul serve the whole group without materialising $G$ copies of K and V, which is exactly the memory the KV cache saves.

The three masks are independent constraints on the pair $(i, j)$ and combine with a logical AND. Causality and the window depend only on $(i, j)$; padding depends on $(b, j)$. Because padding is on the right, a real query $i < \ell_b$ always keeps at least the key $j = i$, so only padded query rows can have no valid keys.

## Complexity
$O(B H T^2 d_h)$ time and $O(B H T^2)$ memory for the scores. A sliding window makes the useful work $O(B H T w d_h)$, but only a kernel that skips masked tiles (FlashAttention with window support) realises that saving; this dense version still computes every score.

## Common bugs
- `repeat` instead of `repeat_interleave`: `repeat` makes query head $h$ use key head $h \bmod H_{kv}$, wrong whenever $1 < H_{kv} < H$ (test 3).
- Off-by-one windows: $i - j < w$ means a window of $w$ tokens including the current one. With $w = 1$ (test 2) each token attends only to itself, so the output is $V$ projected by `Wo`.
- NaN from fully masked rows leaking into the result, or masking padded **queries** but forgetting padded **keys** (real tokens would then attend to padding; here it can't happen under causality with right padding, but it does with left padding).

## Interview follow-ups
- With left padding (common for batched generation), which mask changes, and why do position ids then need adjusting for RoPE?
- How does a sliding window bound the KV cache, and how does a rolling buffer index it?
"""
