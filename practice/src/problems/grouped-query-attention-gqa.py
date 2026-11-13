import numpy as np

META = dict(title='Convert MHA weights to grouped-query attention', tags=['Attention', 'Inference'], diff='Medium', runtime='torch',
            lessons=['aieng:04.04', 'ieng:03.06'],
            related=['multi-head-attention', 'multi-query-attention-mqa', 'kv-cache-memory-for-mha-mqa-gqa-and-mla'])

STATEMENT = r"""
Grouped-query attention (GQA) keeps $H$ query heads but only $H_{kv}$ key/value heads: query heads come in groups of $r = H / H_{kv}$ consecutive heads, and each group shares one KV head. A common way to get a GQA model is to **convert** a multi-head checkpoint: average the key (and value) projections of the heads in each group, then fine-tune briefly. Implement the conversion and the GQA forward pass.

Write `gqa_from_mha(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads)` in PyTorch.

## Inputs (float64; projections are $XW$, no biases)
- `x`: `(B, T, D)`. $H$ = `n_heads` divides $D$, $d_h = D / H$. $H_{kv}$ = `n_kv_heads` divides $H$.
- `Wq`, `Wk`, `Wv`, `Wo`: `(D, D)`, the **MHA** weights. Head $h$ of `Wk` is columns $h d_h : (h+1) d_h$ (same for `Wq`, `Wv`).

## Conversion
KV head $g$ of the new `Wk_g` (shape `(D, H_kv d_h)`, columns $g d_h : (g+1) d_h$) is the **mean** of MHA key heads $g r, \dots, g r + r - 1$. Same for `Wv_g`. `Wq` and `Wo` are unchanged.

## Forward pass
1. $Q = xW_q$ split into $H$ heads; $K = xW_{k,g}$, $V = xW_{v,g}$ split into $H_{kv}$ heads.
2. Query head $h$ uses KV head $\lfloor h / r \rfloor$. Scores $Q_h K_g^\top / \sqrt{d_h}$, **causal** (position $i$ sees $j \le i$), softmax over keys, times $V_g$.
3. Concatenate the $H$ head outputs in order to `(B, T, D)` and multiply by $W_o$.

Share each KV head across its group by reshaping and broadcasting, not with `repeat_interleave`.

## Output
A tuple `(y, Wk_g, Wv_g)`: `y` is `(B, T, D)`; `Wk_g`, `Wv_g` are `(D, H_kv d_h)`.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def gqa_from_mha(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads):
    """Return (y (B,T,D), Wk_g (D, H_kv*dh), Wv_g (D, H_kv*dh))."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import torch

def gqa_from_mha(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads):
    B, T, D = x.shape
    H, G = n_heads, n_kv_heads
    dh, r = D // H, H // G
    # (D, H*dh) -> (D, G, r, dh): heads g*r .. g*r+r-1 sit next to each other, so mean over the r axis
    Wk_g = Wk.view(D, G, r, dh).mean(dim=2).reshape(D, G * dh)
    Wv_g = Wv.view(D, G, r, dh).mean(dim=2).reshape(D, G * dh)

    q = (x @ Wq).view(B, T, G, r, dh).permute(0, 2, 3, 1, 4)          # (B, G, r, T, dh)
    k = (x @ Wk_g).view(B, T, G, 1, dh).permute(0, 2, 3, 1, 4)        # (B, G, 1, T, dh): broadcasts over r
    v = (x @ Wv_g).view(B, T, G, 1, dh).permute(0, 2, 3, 1, 4)

    scores = q @ k.transpose(-2, -1) / math.sqrt(dh)                   # (B, G, r, T, T)
    future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
    attn = torch.softmax(scores.masked_fill(future, float('-inf')), dim=-1)
    o = attn @ v                                                       # (B, G, r, T, dh)
    # head index h = g*r + j, so (G, r) flattens back to H in the right order
    o = o.permute(0, 3, 1, 2, 4).reshape(B, T, D)
    return o @ Wo, Wk_g, Wv_g
'''

HINTS = [
    'View `Wk` as `(D, H_kv, r, d_h)`. Because the heads of a group are consecutive, averaging over axis 2 gives exactly the grouped projection.',
    'Reshape queries to `(B, H_kv, r, T, d_h)` and keys/values to `(B, H_kv, 1, T, d_h)`. A batched matmul then broadcasts each KV head over its `r` query heads.',
    'After attention, permute back to `(B, T, H_kv, r, d_h)` and flatten the last three axes: query head `g*r + j` lands at the right feature columns.',
]

_r = np.random.default_rng(41)


def _case(B, T, D, H, G, s=0.4):
    n = lambda *sh: _r.normal(size=sh) * s
    return (_r.normal(size=(B, T, D)), n(D, D), n(D, D), n(D, D), n(D, D), H, G)


TESTS = [
    (np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.eye(2), np.array([[1.0, 3.0], [0.0, 2.0]]), np.array([[2.0, 0.0], [0.0, 4.0]]), np.eye(2), 2, 1),
    (np.array([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]]), np.eye(4), np.eye(4), np.eye(4) * 2, np.eye(4), 2, 2),
    _case(2, 5, 8, 4, 2),
    _case(1, 6, 8, 4, 1),
    _case(2, 4, 12, 6, 3),
    _case(3, 3, 6, 3, 3),
    _case(1, 1, 8, 8, 2),
    _case(2, 7, 16, 8, 4),
]

EXPLANATION = r"""
## Why it works
Each KV head is a linear map, so averaging the weight columns of $r$ heads gives the key that is the average of their keys: $x\,\bar W = \frac1r \sum x W_h$. That is the best single head in a least-squares sense when you know nothing else, and it keeps the model close enough that a short fine-tune recovers most of the quality. In the forward pass, laying out query heads as `(H_kv, r)` makes "head $h$ uses KV head $\lfloor h/r \rfloor$" a pure broadcast.

## Cost
Per token per layer the cache holds $2 H_{kv} d_h$ numbers instead of $2 H d_h$, an $r$-fold saving in memory and in the bytes a decode step must read. Compute is unchanged: there are still $H$ score matrices.

## Common bugs
- Grouping heads $h, h + H_{kv}, h + 2H_{kv}, \dots$ (strided) instead of consecutive heads. `view(D, G, r, dh)` groups consecutive heads; `view(D, r, G, dh)` would group strided ones.
- `repeat_interleave` of K and V to $H$ heads: correct numbers, but it rebuilds the full MHA-sized tensors and throws away the bandwidth win.
- Scaling by $\sqrt{D}$ or $\sqrt{H_{kv} d_h}$ instead of $\sqrt{d_h}$.

## Interview follow-ups
- Under 8-way tensor parallelism with $H_{kv} = 4$, how are KV heads placed? (Each KV head is replicated on 2 GPUs, so the cache per GPU is larger than $1/8$ of the total.)
- What breaks if you mean-pool the **query** heads too? (You would lose head diversity in the scores, which is what MHA quality depends on; GQA only shares K and V.)
"""
