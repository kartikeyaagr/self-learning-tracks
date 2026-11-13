import numpy as np

META = dict(title="Multi-head attention from nn.MultiheadAttention's packed weights", tags=['Attention', 'Transformers'],
            diff='Medium', runtime='torch', lessons=['aieng:05.02', 'air-fanout-v1:06.02'],
            related=['scaled-dot-product-attention-2', 'scaled-dot-product-attention-3', 'multi-query-attention-mqa'])

STATEMENT = r"""
Checkpoints often store attention as one **packed** input projection, as `torch.nn.MultiheadAttention` does. Loading such weights into your own kernel (or the other way round) means knowing the exact layout. Re-implement the module's forward pass from its parameters.

Write `mha_packed(x, in_w, in_b, out_w, out_b, n_heads, key_padding_mask=None, causal=False)`.

## Parameters (PyTorch `nn.Linear` convention: $y = x W^\top + b$)
- `x`: `(B, T, D)` float64 (batch first).
- `in_w`: `(3D, D)`, `in_b`: `(3D,)`. Rows `0:D` are the query projection, `D:2D` the key projection, `2D:3D` the value projection (same split for the bias).
- `out_w`: `(D, D)`, `out_b`: `(D,)`.
- `n_heads` $= H$ divides $D$; head $h$ owns features $h\,d_h$ to $(h+1)\,d_h - 1$ with $d_h = D/H$.
- `key_padding_mask`: optional bool `(B, T)`. **`True` marks padding**, which no query may attend to (PyTorch's convention).
- `causal`: if true, query $i$ may only attend to keys $j \le i$.

Tests guarantee every query can see at least one key.

## Output
A tuple `(out, attn)`:
- `out`: `(B, T, D)`, the module output: per-head $\mathrm{softmax}(Q_h K_h^\top / \sqrt{d_h})\,V_h$, heads concatenated in order, then the output projection.
- `attn`: `(B, T, T)`, the attention weights **averaged over heads** (what the module returns with `need_weights=True`).

No Python loops over batch, heads or positions. You can check your function by loading the same tensors into `nn.MultiheadAttention(D, H, batch_first=True, dtype=torch.float64)`.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def mha_packed(x, in_w, in_b, out_w, out_b, n_heads, key_padding_mask=None, causal=False):
    """Return (out (B,T,D), head-averaged attention weights (B,T,T))."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import torch

def mha_packed(x, in_w, in_b, out_w, out_b, n_heads, key_padding_mask=None, causal=False):
    B, T, D = x.shape
    dh = D // n_heads
    qkv = x @ in_w.T + in_b                                   # (B, T, 3D): one GEMM for all three projections
    q, k, v = qkv.split(D, dim=-1)
    heads = lambda t: t.view(B, T, n_heads, dh).transpose(1, 2)   # (B, H, T, dh)
    q, k, v = heads(q), heads(k), heads(v)

    scores = q @ k.transpose(-2, -1) / math.sqrt(dh)          # (B, H, T, T)
    hide = torch.zeros(B, 1, T, T, dtype=torch.bool)
    if key_padding_mask is not None:
        hide = hide | key_padding_mask.bool()[:, None, None, :]   # same keys hidden for every query and head
    if causal:
        hide = hide | torch.ones(T, T, dtype=torch.bool).triu(1)
    attn = torch.softmax(scores.masked_fill(hide, float('-inf')), dim=-1)

    o = (attn @ v).transpose(1, 2).reshape(B, T, D)           # merge heads back in order
    return o @ out_w.T + out_b, attn.mean(dim=1)
'''

HINTS = [
    'One matmul `x @ in_w.T + in_b` gives `(B, T, 3D)`; `split(D, dim=-1)` yields q, k, v in that order.',
    'Reshape each to `(B, T, H, dh)` and transpose to `(B, H, T, dh)`. A padding mask `(B, T)` becomes `(B, 1, 1, T)` so it hides key columns for all heads and queries.',
    'Combine padding and causal masks with `|`, fill with `-inf`, softmax over the last axis. Return `attn.mean(dim=1)` for the head-averaged weights and apply `out_w.T`, not `out_w`.',
]

_r = np.random.default_rng(11)


def _case(B, T, D, s=0.5):
    return _r.normal(size=(B, T, D)), _r.normal(size=(3 * D, D)) * s, _r.normal(size=3 * D) * 0.1, \
        _r.normal(size=(D, D)) * s, _r.normal(size=D) * 0.1


_x, _iw, _ib, _ow, _ob = _case(1, 3, 4)
TESTS = [
    (np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.vstack([np.eye(2)] * 3), np.zeros(6), np.eye(2), np.zeros(2), 1),
    (np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.vstack([np.eye(2)] * 3), np.zeros(6), np.eye(2), np.zeros(2), 2,
     None, True),
    (*_case(2, 5, 8), 2),
    (*_case(2, 4, 6), 3, np.array([[False, False, True, True], [False, False, False, False]])),
    (*_case(1, 6, 8), 4, None, True),
    (*_case(3, 5, 4), 1, np.array([[False] * 5, [False, False, False, True, True], [False, True, True, True, True]]), True),
    (_x, _iw, _ib, _ow, _ob, 4),
    (*_case(2, 7, 12, 0.3), 6, np.array([[False] * 6 + [True], [False] * 7]), True),
]

EXPLANATION = r"""
## Why it works
`nn.MultiheadAttention` keeps the three projections stacked in one `(3D, D)` matrix so a single GEMM produces Q, K and V. After the split, it is ordinary multi-head attention: reshape to heads, scale by $1/\sqrt{d_h}$, mask, softmax over keys, mix values, merge heads, project. The returned weights are the per-head attention matrices averaged over heads.

## Masks
Padding is a property of the **keys**, so the `(B, T)` mask broadcasts as `(B, 1, 1, T)`. Causality is a property of positions, `(T, T)`. Padded *query* rows still produce outputs (they see real keys); the model ignores them later. A fully hidden row would give `nan` (softmax of all `-inf`), which is why left padding with causal masks needs care.

## Complexity
Projections cost $O(BTD^2)$; attention costs $O(BT^2D)$ and $O(BHT^2)$ memory for the scores. For short sequences the projections dominate; past $T \approx D$ attention does.

## Common bugs
- Multiplying by `in_w` instead of `in_w.T` (silently wrong when $D$ is square).
- Reshaping `(B, T, D)` straight to `(B, H, T, dh)` without the transpose: that scrambles tokens and heads.
- Inverting the padding convention: in PyTorch `True` means "ignore", in many other libraries `1` means "keep".

## Interview follow-ups
- Why does packing Q, K and V into one projection speed things up on a GPU? (One larger GEMM, one kernel launch, better utilisation.)
- How would you convert these weights into a grouped-query layout with fewer KV heads?
"""
