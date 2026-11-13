import numpy as np

META = dict(title='Attention scores from named-axis layouts', tags=['Attention', 'Transformers'], diff='Easy', runtime='torch',
            lessons=['air-fanout-v1:03.02', 'ieng:00.02'],
            related=['scaled-dot-product-attention', 'causal-grouped-query-attention', 'rotary-query-and-key-embeddings'])

STATEMENT = r"""
Different codebases store attention inputs in different axis orders: `(batch, seq, head, dim)` from a fused QKV projection, `(batch, head, seq, dim)` after a transpose, sometimes `(seq, batch, head, dim)` in older code. Hard-coding `transpose(1, 2)` everywhere is how silent shape bugs happen. Naming the axes and letting `einsum` line them up is safer.

Write `attention_scores(q, k, q_layout, k_layout)`.

## Inputs
- `q`, `k`: `float64` tensors with 4 axes each.
- `q_layout`, `k_layout`: strings naming the axes of `q` and `k`, in order, separated by single spaces. Each is a permutation of the four names `batch`, `head`, `seq`, `dim`, for example `"batch seq head dim"`.
- `q` and `k` share `batch`, `head` and `dim` sizes. Their `seq` sizes can differ: $T_q$ for queries and $T_k$ for keys.

## Output
A `float64` tensor of shape `(batch, head, T_q, T_k)`:

$$S_{b,h,i,j} = \frac{1}{\sqrt{d}} \sum_{c} q_{b,h,i,c}\; k_{b,h,j,c}$$

where $d$ is the size of `dim` (indices written in logical order, whatever the stored layout). Use tensor ops only, no Python loops over the data.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import math
import torch

def attention_scores(q, k, q_layout, k_layout):
    """q, k: 4-D tensors described by layout strings. Return (batch, head, T_q, T_k) scores."""
    return q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])   # TODO: only right for "batch head seq dim"
'''

SOLUTION = '''import math
import torch

def attention_scores(q, k, q_layout, k_layout):
    # One einsum letter per axis name. The key's sequence axis needs its own letter (j), so it is
    # not summed against the query's (i).
    q_letters = {'batch': 'b', 'head': 'h', 'seq': 'i', 'dim': 'd'}
    k_letters = {'batch': 'b', 'head': 'h', 'seq': 'j', 'dim': 'd'}
    q_spec = ''.join(q_letters[n] for n in q_layout.split())
    k_spec = ''.join(k_letters[n] for n in k_layout.split())
    d = q.shape[q_layout.split().index('dim')]
    return torch.einsum(f'{q_spec},{k_spec}->bhij', q, k) / math.sqrt(d)
'''

HINTS = [
    'Map each axis name to an einsum letter. The output spec is fixed: `bhij`.',
    'Queries and keys both have a `seq` axis, but they must get **different** letters (`i` and `j`). With the same letter, einsum would multiply position $i$ only with key position $i$.',
    'Read $d$ from the position of `dim` in `q_layout`, not from `q.shape[-1]`.',
]

_r = np.random.default_rng(5)
TESTS = [
    (np.arange(8.0).reshape(1, 1, 2, 4), np.ones((1, 1, 3, 4)), 'batch head seq dim', 'batch head seq dim'),
    (np.eye(2).reshape(1, 2, 1, 2), np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]]), 'batch seq head dim', 'batch seq head dim'),
    (_r.normal(size=(2, 5, 3, 4)), _r.normal(size=(2, 7, 3, 4)), 'batch seq head dim', 'batch seq head dim'),
    (_r.normal(size=(2, 3, 5, 4)), _r.normal(size=(2, 7, 3, 4)), 'batch head seq dim', 'batch seq head dim'),
    (_r.normal(size=(6, 2, 3, 8)), _r.normal(size=(4, 2, 3, 8)), 'seq batch head dim', 'seq batch head dim'),
    (_r.normal(size=(8, 3, 2, 5)), _r.normal(size=(2, 3, 8, 4)), 'dim head batch seq', 'batch head dim seq'),
    (_r.normal(size=(1, 4, 1, 16)), _r.normal(size=(1, 4, 9, 16)), 'batch head seq dim', 'batch head seq dim'),
    (_r.normal(size=(3, 2, 2, 2)) * 1e3, _r.normal(size=(3, 2, 2, 2)), 'head seq batch dim', 'head seq batch dim'),
]

EXPLANATION = r"""
## Why einsum fits
`einsum` describes the computation by naming axes: letters that appear in both inputs but not the output are summed (`d`), letters in the output are kept and placed in output order (`bhij`). The stored layout only changes the input specs, so one line handles every permutation with no `permute` calls. Libraries like `einops` (`rearrange(q, "b t h d -> b h t d")`) apply the same idea to reshapes.

## The two sequence axes
Attention compares every query position with every key position, so the query and key sequence axes are different axes of the result. Giving both the same letter turns the product into a diagonal: `einsum("bhid,bhid->bhi")` is a per-position dot product, not attention. Test 1 has $T_q = 2$ and $T_k = 3$ precisely so that bug fails with a shape error.

## Cost
$O(B H T_q T_k d)$ multiply-adds, the same as `q @ k.transpose(-1, -2)`. PyTorch's einsum permutes the operands and calls a batched matmul, so there is no speed penalty for the readable version. The real cost of attention scores is memory: the `(B, H, T_q, T_k)` tensor grows with $T^2$, which is what FlashAttention avoids storing.

## Common bugs
- Taking $d$ from `q.shape[-1]` when `dim` is not last (test 6).
- Using `view` to "reorder" axes. `view(B, H, T, D)` on a `(B, T, H, D)` tensor reinterprets memory and scrambles heads without any error.

## Interview follow-ups
- After `transpose`, a tensor is no longer contiguous. Which operations then need `.contiguous()`, and what does that cost?
- How would you extend the function to grouped-query attention, where `k` has fewer heads than `q`?
"""
