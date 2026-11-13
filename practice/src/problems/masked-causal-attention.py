import numpy as np

META = dict(title='Causal attention over packed documents', tags=['Attention', 'Language Models'], diff='Medium',
            runtime='torch', lessons=['aieng:05.01', 'ieng:02.02'],
            related=['scaled-dot-product-attention-2', 'scaled-dot-product-attention', 'multi-head-attention'])

STATEMENT = r"""
LLM pre-training packs several short documents into one fixed-length row to avoid padding. A plain causal mask would then let a document attend to the end of the previous one. Build the **document-aware causal mask** and use it.

Write `packed_causal_attention(q, k, v, doc_ids)`:

- `q`, `k`, `v`: float64 `(B, nh, T, dh)`, already split into heads.
- `doc_ids`: int64 `(B, T)`. Each token's document id; `-1` marks padding. Ids are only compared for equality within a row.

Query $i$ may attend to key $j$ (same row) iff all of:

1. $j \le i$ (causal),
2. `doc_ids[b, i] == doc_ids[b, j]` (same document),
3. `doc_ids[b, j] != -1` (not padding).

Scores are $q_i \cdot k_j / \sqrt{d_h}$, softmaxed over the allowed keys; output row $i$ is the weighted sum of `v`. Every real query can attend to at least itself. Padding queries output zeros.

Return a tuple `(out, pos)`:

- `out`: `(B, nh, T, dh)`.
- `pos`: int64 `(B, T)`, each token's position **within its document** (0 for a document's first token), the position ids you'd feed to RoPE. Padding gets `-1`.

Use tensor ops, no Python loops over batch, heads or positions.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def packed_causal_attention(q, k, v, doc_ids):
    """Return (out (B, nh, T, dh), pos (B, T) int64)."""
    T = q.shape[-2]
    allowed = torch.ones(T, T, dtype=torch.bool).tril()           # TODO: documents and padding
    s = (q @ k.transpose(-2, -1)) / q.shape[-1] ** 0.5
    s = s.masked_fill(~allowed, float("-inf"))
    pos = torch.arange(T).expand_as(doc_ids)                       # TODO: restart at each document
    return torch.softmax(s, dim=-1) @ v, pos
'''

SOLUTION = '''import torch

def packed_causal_attention(q, k, v, doc_ids):
    T = q.shape[-2]
    causal = torch.ones(T, T, dtype=torch.bool).tril()            # (T, T): j <= i
    same = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)          # (B, T, T): same document
    real_key = (doc_ids != -1).unsqueeze(-2)                       # (B, 1, T)
    allowed = causal & same & real_key                             # (B, T, T); padding queries end up all False

    s = (q @ k.transpose(-2, -1)) / q.shape[-1] ** 0.5             # (B, nh, T, T)
    s = s.masked_fill(~allowed.unsqueeze(1), float("-inf"))        # broadcast over heads
    w = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)        # padding rows: all -inf -> NaN -> 0
    out = w @ v

    pos = allowed.sum(-1) - 1                                      # earlier same-doc tokens = position; pads give -1
    return out, pos
'''

HINTS = [
    'Compare ids pairwise with broadcasting: `doc_ids[:, :, None] == doc_ids[:, None, :]` is a `(B, T, T)` "same document" mask. AND it with `tril` and with "key is not padding".',
    'The mask is per row, not per head: add a head axis with `unsqueeze(1)` before `masked_fill`. A padding query has no allowed key, so its softmax row is NaN; replace it with 0.',
    'A token\'s position in its document is the number of earlier tokens of the same document, which is its row of the allowed mask summed, minus one. For padding that sum is 0, giving -1.',
]

_r = np.random.default_rng(31)

def _case(B, nh, T, dh, ids):
    return (_r.normal(size=(B, nh, T, dh)), _r.normal(size=(B, nh, T, dh)), _r.normal(size=(B, nh, T, dh)),
            np.array(ids, dtype=np.int64))

TESTS = [
    (np.zeros((1, 1, 4, 1)), np.zeros((1, 1, 4, 1)), np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4, 1),
     np.array([[0, 0, 1, 1]])),
    (np.array([1.0, 0.0, 2.0]).reshape(1, 1, 3, 1), np.array([1.0, 1.0, 1.0]).reshape(1, 1, 3, 1),
     np.array([3.0, 5.0, 7.0]).reshape(1, 1, 3, 1), np.array([[4, 4, -1]])),
    _case(2, 2, 6, 3, [[0, 0, 0, 1, 1, 2], [5, 5, 5, 5, 5, 5]]),
    _case(1, 3, 5, 4, [[0, 1, 2, 3, 4]]),
    _case(2, 1, 6, 2, [[7, 7, 3, 3, -1, -1], [-1, -1, -1, -1, -1, -1]]),
    _case(1, 2, 7, 2, [[0, 0, 1, 1, 1, 0, 0]]),
    _case(3, 2, 8, 4, [[0, 0, 0, 0, 1, 1, 1, -1], [2, 3, 3, 3, 3, 4, 4, 4], [9] * 8]),
]

EXPLANATION = r"""
## One mask, three conditions
The allowed set is the intersection of three boolean masks that broadcast to `(B, T, T)`: the lower triangle (causal), pairwise id equality (same document) and a key-side padding mask. Packed documents are contiguous, so the result is block-diagonal, with a causal triangle in each block. Because the rule is stated with ids alone, the same code also handles a document id that reappears later in the row (test 6): the second run attends back to the first, which is why packers give every document a unique id.

## Position ids come from the same mask
A token's position inside its document is the count of earlier tokens of that document, which is the row sum of the allowed mask minus one. Restarting positions is what makes a packed row equivalent to separate unpadded sequences when you use RoPE or learned positions.

## Complexity
$O(B \cdot n_h \cdot T^2 \cdot d_h)$ time and $O(B\,T^2)$ for the mask. Dense masking wastes the off-block work; variable-length FlashAttention kernels take cumulative sequence lengths instead and skip those tiles entirely, so cost scales with $\sum_d L_d^2$ instead of $T^2$.

## Common bugs
- Only a causal mask: tokens leak across document boundaries. Loss still goes down, so nobody notices until evaluation.
- Masking only padding keys but leaving padding queries to softmax over nothing (NaN).
- Building the mask per head with a Python loop, or forgetting the head axis so it broadcasts against the wrong dimension.

## Interview follow-ups
- How does `flash_attn_varlen_func` represent the same packing with `cu_seqlens`, and what does it gain over a dense mask?
- Should the loss on the first token of each packed document (predicted from the previous document's last token) be masked out?
"""
