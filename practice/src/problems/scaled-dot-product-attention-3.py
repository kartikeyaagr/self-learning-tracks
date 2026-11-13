import numpy as np

META = dict(title='Batched attention for cached decode steps', tags=['Attention', 'Inference'], diff='Easy', runtime='torch',
            lessons=['ieng:14.04', 'aieng:05.01'],
            related=['scaled-dot-product-attention', 'flashattention-with-online-softmax', 'multi-query-attention-mqa'])

STATEMENT = r"""
Inside an inference engine, attention is called with **fewer queries than keys**: a decode step has one new query per sequence but a cache full of keys, and a chunk of a long prompt has a few hundred queries against everything processed so far. Batches are also padded, so some key slots are not real tokens.

Write `sdpa(q, k, v, key_mask=None, causal=True)` in PyTorch.

## Shapes (float64)
- `q`: `(B, H, T, d)`, `k`: `(B, H, S, d)`, `v`: `(B, H, S, d_v)`. Scale the scores by $1/\sqrt{d}$.
- `key_mask`: optional bool `(B, S)`; `True` marks a real key, `False` a padding slot that no query may attend to. `None` means every key is real.

## Causal alignment
When `causal` is true, $T \le S$ and the queries are the **last** $T$ positions of the sequence: query $i$ sits at position $S - T + i$ and may attend to keys $j \le S - T + i$. With $T = S$ this is the usual lower-triangular mask; with $T = 1$ the single query sees every key.

## Output
`(B, H, T, d_v)`. A query row with **no** allowed key (every candidate is padding) returns zeros, not `nan`. Use tensor ops only, no Python loops.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def sdpa(q, k, v, key_mask=None, causal=True):
    """q (B,H,T,d), k (B,H,S,d), v (B,H,S,dv). Return (B,H,T,dv)."""
    scores = q @ k.transpose(-2, -1)
    return torch.softmax(scores, dim=-1) @ v       # TODO: scale, masks, empty rows
'''

SOLUTION = '''import math
import torch

def sdpa(q, k, v, key_mask=None, causal=True):
    T, S = q.shape[-2], k.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])         # (B, H, T, S)

    allowed = torch.ones(T, S, dtype=torch.bool, device=q.device)
    if causal:                                                        # bottom-right aligned: query i is at S - T + i
        allowed = torch.arange(S, device=q.device)[None, :] <= torch.arange(T, device=q.device)[:, None] + (S - T)
    if key_mask is not None:
        allowed = allowed & key_mask.bool()[:, None, None, :]         # (B, 1, T, S) after broadcasting

    attn = torch.softmax(scores.masked_fill(~allowed, float('-inf')), dim=-1)
    return attn.nan_to_num(0.0) @ v                                   # an all -inf row softmaxes to nan: make it 0
'''

HINTS = [
    'Build a boolean `(T, S)` "allowed" matrix and turn disallowed scores into `-inf` with `masked_fill` before the softmax.',
    'The causal rule is `j <= i + (S - T)`: compare `arange(S)[None, :]` with `arange(T)[:, None] + (S - T)`. The key mask needs shape `(B, 1, 1, S)` to broadcast over heads and queries.',
    'If every entry of a row is `-inf`, softmax returns `nan` there. `nan_to_num(0.0)` on the weights turns those rows into zeros.',
]

_r = np.random.default_rng(31)
_n = lambda *s: _r.normal(size=s)
TESTS = [
    (np.array([[[[1.0, 0.0], [0.0, 1.0]]]]), np.array([[[[1.0, 0.0], [0.0, 1.0]]]]), np.array([[[[1.0], [3.0]]]])),
    (np.array([[[[1.0, 1.0]]]]), np.array([[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]]), np.array([[[[1.0], [2.0], [9.0]]]]),
     np.array([[True, True, False]])),
    (_n(2, 3, 4, 8), _n(2, 3, 4, 8), _n(2, 3, 4, 5)),
    (_n(2, 2, 1, 4), _n(2, 2, 6, 4), _n(2, 2, 6, 4), np.array([[True] * 6, [True] * 4 + [False] * 2])),
    (_n(1, 4, 3, 4), _n(1, 4, 7, 4), _n(1, 4, 7, 2)),
    (_n(2, 1, 3, 4), _n(2, 1, 5, 4), _n(2, 1, 5, 3), np.array([[False, True, False, True, True], [True, True, True, False, False]]), False),
    (_n(1, 2, 2, 4), _n(1, 2, 4, 4), _n(1, 2, 4, 3), np.array([[False, False, False, True]])),
    (_n(3, 2, 5, 6), _n(3, 2, 5, 6), _n(3, 2, 5, 6), np.array([[True] * 5, [True] * 3 + [False] * 2, [False] * 5]), True),
]

EXPLANATION = r"""
## Why it works
Both rules reduce to one boolean matrix of allowed (query, key) pairs. Masking with $-\infty$ before the softmax gives exactly zero weight to forbidden keys while the remaining weights still sum to 1. Aligning the causal mask to the **end** of the key axis is what lets one function serve full prefill ($T = S$), chunked prefill ($1 < T < S$) and decode ($T = 1$): the cache holds the past, the queries are the newest tokens.

## Cost
The score tensor is $B \cdot H \cdot T \cdot S$. For decode ($T = 1$) the work is tiny but every key and value in the cache is read once, so decode attention is limited by memory bandwidth, not FLOPs. For prefill ($T = S$) it is the quadratic $T^2$ term that dominates, which is what FlashAttention attacks.

## Common bugs
- A top-left mask (`tril(ones(T, S))`) when $T < S$: a decode query then sees only key 0.
- Masking padded **queries** instead of padded keys. Padded query rows produce garbage that is ignored later; padded keys corrupt every real row.
- Using a large negative number like `-1e9` instead of `-inf` in fp16: it overflows to `-inf` anyway in half precision, and in fp32 it leaves rows that are fully masked as a uniform average of padding instead of zeros.

## Interview follow-ups
- Why can decode attention skip the causal mask entirely? (With $T = 1$ the query is the newest token, so every cached key is in its past.)
- With left padding instead of right padding, what else must change besides the key mask? (Position ids for RoPE, which must count only real tokens.)
"""
