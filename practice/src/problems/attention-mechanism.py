import numpy as np

META = dict(title='Additive attention for an encoder-decoder', tags=['Attention', 'NLP'], diff='Medium', runtime='torch',
            lessons=['air-fanout-v1:06.01', 'aieng:04.02'],
            related=['scaled-dot-product-attention', 'sequence-to-sequence-forward-pass', 'masked-mean-pooling'])

STATEMENT = r"""
Before dot-product attention, encoder-decoder models scored each source position with a small feed-forward network: additive (Bahdanau-style) attention. Implement it for all decoder steps at once.

Write `additive_attention(query, keys, values, W_q, W_k, v, src_mask)`:

- `query`: float64 `(B, T, Dq)`, one query per decoder step.
- `keys`: float64 `(B, S, Dk)`, `values`: float64 `(B, S, Dv)`: encoder states.
- `W_q`: `(A, Dq)`, `W_k`: `(A, Dk)`, `v`: `(A,)`: the scoring network (no biases).
- `src_mask`: `(B, S)` of 0/1 or bool; 1 marks a real source token, 0 padding.

For each batch $b$, step $t$ and source position $s$:

$$e_{bts} = v^\top \tanh\left(W_q\, q_{bt} + W_k\, k_{bs}\right)$$

Padded positions get no weight. The attention weights are $\alpha_{bt\cdot} = \text{softmax}_s(e_{bt\cdot})$ over the real positions, and the context is $c_{bt} = \sum_s \alpha_{bts}\, \text{values}_{bs}$.

Return a tuple `(context, weights)` of shapes `(B, T, Dv)` and `(B, T, S)`. If a sequence has no real source token, its weights and context are all zeros (not NaN).

Use broadcasting, no Python loops over `B`, `T` or `S`. Project the keys once, not once per query.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def additive_attention(query, keys, values, W_q, W_k, v, src_mask):
    """Return (context (B, T, Dv), weights (B, T, S))."""
    q = query @ W_q.T                                   # (B, T, A)
    k = keys @ W_k.T                                    # (B, S, A)
    e = torch.tanh(q + k) @ v                           # TODO: shapes don't line up, and padding is ignored
    w = torch.softmax(e, dim=-1)
    return w @ values, w
'''

SOLUTION = '''import torch

def additive_attention(query, keys, values, W_q, W_k, v, src_mask):
    q = (query @ W_q.T).unsqueeze(2)                    # (B, T, 1, A)
    k = (keys @ W_k.T).unsqueeze(1)                     # (B, 1, S, A): projected once, shared by every query
    e = torch.tanh(q + k) @ v                           # (B, T, S)
    real = src_mask.bool().unsqueeze(1)                 # (B, 1, S)
    e = e.masked_fill(~real, float("-inf"))
    w = torch.softmax(e, dim=-1)
    w = torch.nan_to_num(w, nan=0.0)                    # all-padding rows: softmax of all -inf is NaN
    return w @ values, w                                # (B, T, S) @ (B, S, Dv) -> (B, T, Dv)
'''

HINTS = [
    'Project once: `q = query @ W_q.T` is `(B, T, A)` and `k = keys @ W_k.T` is `(B, S, A)`. To pair every query with every key, unsqueeze them to `(B, T, 1, A)` and `(B, 1, S, A)`.',
    '`torch.tanh(q + k) @ v` contracts the last axis and gives scores `(B, T, S)`. Set padded scores to `-inf` with `masked_fill` before the softmax over the last axis.',
    'A row that is entirely `-inf` comes out of softmax as NaN. Replace those with 0 (`torch.nan_to_num` or `torch.where`); the context `w @ values` is then 0 as well.',
]

_r = np.random.default_rng(23)

def _case(B, T, S, Dq, Dk, Dv, A, mask):
    return (_r.normal(size=(B, T, Dq)), _r.normal(size=(B, S, Dk)), _r.normal(size=(B, S, Dv)),
            _r.normal(size=(A, Dq)), _r.normal(size=(A, Dk)), _r.normal(size=A), np.array(mask))

TESTS = [
    (np.array([[[1.0]]]), np.array([[[1.0], [-1.0]]]), np.array([[[10.0], [20.0]]]), np.array([[1.0]]),
     np.array([[1.0]]), np.array([1.0]), np.array([[1, 1]])),
    (np.array([[[0.0], [2.0]]]), np.array([[[1.0], [0.0], [5.0]]]), np.array([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]]),
     np.array([[1.0], [-1.0]]), np.array([[1.0], [1.0]]), np.array([1.0, 1.0]), np.array([[1, 1, 0]])),
    _case(2, 3, 4, 3, 2, 2, 5, [[1, 1, 1, 1], [1, 1, 0, 0]]),
    _case(1, 1, 5, 2, 2, 3, 4, [[1, 0, 1, 0, 1]]),
    _case(2, 2, 3, 2, 3, 2, 3, [[0, 0, 0], [1, 1, 1]]),
    _case(3, 4, 6, 4, 4, 3, 8, [[True] * 6, [True, True, True, False, False, False], [True] + [False] * 5]),
    _case(1, 5, 2, 3, 3, 4, 2, [[1, 1]]),
]

EXPLANATION = r"""
## Scoring with a tiny MLP
Dot-product attention compares query and key with $q^\top k$, which needs them in the same space. Additive attention projects both into a shared $A$-dimensional space, adds them, applies $\tanh$ and reads out a scalar with $v$: a one-hidden-layer network that can score queries and keys of different sizes. It was the attention in the first neural machine translation models with alignment.

## Broadcasting instead of loops
The pairwise term $W_q q_{bt} + W_k k_{bs}$ is a `(B, T, S, A)` tensor built by adding `(B, T, 1, A)` and `(B, 1, S, A)`. Projecting before broadcasting matters: projecting each of the $T \cdot S$ pairs costs $O(BTS\,A(D_q + D_k))$, projecting once costs $O(B(T D_q + S D_k) A)$.

## Complexity
Time $O(B\,T\,S\,A)$ for the scores plus $O(B\,T\,S\,D_v)$ for the context. Memory $O(BTSA)$ for the hidden activations, which is the main drawback against dot-product attention, where scores are a single `(B, T, S)` matmul and never materialise $A$-dimensional pairs.

## Common bugs
- Masking after the softmax (weights no longer sum to 1) or multiplying scores by the mask (a score of 0 still gets weight).
- Letting NaN through for an all-padding sequence; it then poisons the whole batch's gradients.
- Swapping `unsqueeze` positions, which silently broadcasts to `(B, S, T, A)` when $T = S$.

## Interview follow-ups
- Why did dot-product attention replace additive attention in Transformers, and why does it need the $1/\sqrt{d}$ scale that additive attention doesn't?
- How would you add location-aware attention (feeding the previous step's weights into the score) for speech recognition, where alignment is monotonic?
"""
