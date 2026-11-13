import numpy as np

META = dict(title='Multi-query attention with a shared KV cache', tags=['Attention', 'Inference'], diff='Medium',
            runtime='torch', lessons=['ieng:03.06', 'aieng:04.04'],
            related=['multi-head-attention', 'multi-head-attention-mha', 'kv-cache-memory-for-mha-mqa-gqa-and-mla'])

STATEMENT = r"""
Multi-query attention (MQA) keeps $H$ query heads but a **single** key head and a single value head shared by all of them. The KV cache shrinks by a factor of $H$, which is what lets a server hold many more concurrent sequences.

Write `mqa(x, Wq, Wk, Wv, Wo, n_heads, past_k=None, past_v=None)` in PyTorch: causal self-attention for $T$ new tokens that may extend an existing cache.

## Shapes (all float64)
- `x`: `(B, T, D)`, the new tokens. $H$ = `n_heads` divides $D$; $d_h = D/H$.
- `Wq`: `(D, D)`, `Wk`: `(D, d_h)`, `Wv`: `(D, d_h)`, `Wo`: `(D, D)`. Projections are $XW$ (no transpose, no bias).
- `past_k`, `past_v`: either both `None` (empty cache) or both `(B, P, d_h)`: keys and values of the $P$ earlier tokens.

## Computation
1. $Q = xW_q$ split into $H$ heads (head $h$ = features $h d_h : (h+1) d_h$). New $k = xW_k$, $v = xW_v$, each `(B, T, d_h)`.
2. The cache becomes `K = cat(past_k, k)` and `V = cat(past_v, v)` along time, length $L = P + T$.
3. New token $i$ sits at position $P + i$ and attends to cached positions $j \le P + i$. Scores are $Q_h K^\top / \sqrt{d_h}$; softmax over keys; output $\sum_j a_{ij} V_j$.
4. Concatenate heads in order to `(B, T, D)` and multiply by $W_o$.

Share the single K/V head across query heads by **broadcasting**, not by copying it $H$ times.

## Output
A tuple `(y, K, V)`: `y` is `(B, T, D)`; `K` and `V` are the updated caches, `(B, P + T, d_h)` each.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def mqa(x, Wq, Wk, Wv, Wo, n_heads, past_k=None, past_v=None):
    """Return (y (B,T,D), K cache (B,P+T,dh), V cache (B,P+T,dh))."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import torch

def mqa(x, Wq, Wk, Wv, Wo, n_heads, past_k=None, past_v=None):
    B, T, D = x.shape
    dh = D // n_heads
    q = (x @ Wq).view(B, T, n_heads, dh).transpose(1, 2)      # (B, H, T, dh)
    k, v = x @ Wk, x @ Wv                                     # (B, T, dh): one head, shared
    if past_k is not None:
        k, v = torch.cat([past_k, k], dim=1), torch.cat([past_v, v], dim=1)
    L = k.shape[1]
    P = L - T

    # unsqueeze(1) gives (B, 1, L, dh): matmul broadcasts it over the H query heads without a copy
    scores = q @ k.unsqueeze(1).transpose(-2, -1) / math.sqrt(dh)     # (B, H, T, L)
    future = torch.arange(L)[None, :] > (torch.arange(T)[:, None] + P)  # new token i sits at P + i
    attn = torch.softmax(scores.masked_fill(future, float('-inf')), dim=-1)
    o = (attn @ v.unsqueeze(1)).transpose(1, 2).reshape(B, T, D)
    return o @ Wo, k, v
'''

HINTS = [
    'Project queries into `(B, H, T, dh)` as usual, but keys and values stay `(B, T, dh)`: there is only one KV head.',
    'Concatenate the new k, v after the cached ones along dim 1. `k.unsqueeze(1)` has shape `(B, 1, L, dh)` and broadcasts against `(B, H, T, dh)` in a matmul.',
    'The causal mask is aligned to the end: new token `i` may see cached position `j` when `j <= P + i`. With `T = 1` nothing is hidden.',
]

_r = np.random.default_rng(21)


def _w(D, H, s=0.4):
    dh = D // H
    return _r.normal(size=(D, D)) * s, _r.normal(size=(D, dh)) * s, _r.normal(size=(D, dh)) * s, _r.normal(size=(D, D)) * s


TESTS = [
    (np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.eye(2), np.array([[1.0], [0.0]]), np.array([[0.0], [2.0]]), np.eye(2), 2),
    (np.array([[[1.0, 1.0]]]), np.eye(2), np.array([[1.0], [0.0]]), np.array([[0.0], [2.0]]), np.eye(2), 2,
     np.array([[[1.0], [0.0]]]), np.array([[[0.0], [2.0]]])),
    (_r.normal(size=(2, 5, 8)), *_w(8, 4), 4),
    (_r.normal(size=(1, 3, 6)), *_w(6, 3), 3, _r.normal(size=(1, 4, 2)), _r.normal(size=(1, 4, 2))),
    (_r.normal(size=(3, 1, 8)), *_w(8, 2), 2, _r.normal(size=(3, 6, 4)), _r.normal(size=(3, 6, 4))),
    (_r.normal(size=(2, 4, 4)), *_w(4, 1), 1),
    (_r.normal(size=(2, 2, 12)), *_w(12, 6), 6, _r.normal(size=(2, 1, 2)), _r.normal(size=(2, 1, 2))),
]

EXPLANATION = r"""
## Why it works
All $H$ query heads score against the same key matrix and mix the same value matrix; only the queries differ per head. Broadcasting `(B, 1, L, d_h)` against `(B, H, T, d_h)` expresses exactly that without materialising $H$ copies. Appending to the cache and aligning the causal mask to its end makes the same function serve prefill ($P = 0$, $T$ = prompt length) and decode ($T = 1$).

## Memory and speed
Per token per layer, MHA caches $2 H d_h$ numbers; MQA caches $2 d_h$. For $H = 32$ that is 32 times more sequences (or 32 times longer contexts) in the same memory, and decode attention, which is bound by reading the cache, reads 32 times fewer bytes. The cost is some quality, which is why most recent models settle on grouped-query attention with 4 to 8 KV heads.

## Common bugs
- `k.repeat(...)` or `expand(...).contiguous()` to H heads: correct output, but it throws away the bandwidth saving inside the kernel.
- A top-left causal mask (`tril(ones(T, L))`) when $P > 0$: the new tokens then fail to see most of the cache.
- Projecting keys to width $D$: the whole point is a key width of $d_h$.

## Interview follow-ups
- How do you turn an MHA checkpoint into an MQA or GQA one without training from scratch? (Mean-pool the key and value heads in each group, then briefly continue training.)
- Under tensor parallelism with 8 GPUs, what happens to MQA's single KV head? (It is replicated on every GPU, so the per-GPU saving shrinks.)
"""
