import numpy as np

META = dict(title='Embedding gradient with repeated ids', tags=['Embeddings', 'Neural Networks'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:03.05', 'aieng:02.01'], related=['mutation-trap', 'boolean-masking'])

STATEMENT = r"""
An embedding layer is a lookup: `out = E[ids]`, where `E` has shape `(V, d)` (one row per vocabulary item) and `ids` holds integer token ids. In the backward pass, the gradient for row `v` of `E` is the **sum** of the upstream gradients at every position where token `v` appeared:

$$\frac{\partial L}{\partial E_v} = \sum_{p\,:\,\text{ids}_p = v} G_p$$

Write `embedding_grad(ids, grad_out, vocab_size, pad_id=-1)`.

## Input
- `ids`: integer array of any shape `S` (for example `(B, T)`). Values are in `[0, vocab_size)` or equal to `pad_id`.
- `grad_out`: float array of shape `S + (d,)`; `grad_out[p]` is the gradient flowing into the output vector at position `p`.
- Positions where `ids == pad_id` are padding: they contribute **nothing**.

## Output
A `float64` array of shape `(vocab_size, d)`. Rows of tokens that never appear are zero.

## The trap
The same token usually appears many times in a batch. `dE[ids] += grad_out` looks right but is wrong: with repeated indices, buffered fancy-index assignment keeps only one of the updates per row.
"""

STARTER = '''import numpy as np

def embedding_grad(ids, grad_out, vocab_size, pad_id=-1):
    """ids: int (*S), grad_out: (*S, d). Return dE of shape (vocab_size, d), ignoring pad_id."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def embedding_grad(ids, grad_out, vocab_size, pad_id=-1):
    ids, grad_out = np.asarray(ids), np.asarray(grad_out, dtype=float)
    keep = ids != pad_id                       # boolean mask over positions, shape S
    dE = np.zeros((vocab_size, grad_out.shape[-1]))
    # Unbuffered scatter-add: every occurrence of a repeated id adds its row.
    np.add.at(dE, ids[keep], grad_out[keep])
    return dE
'''

HINTS = [
    'Drop the padding first. A boolean mask `ids != pad_id` applied to both `ids` and `grad_out` gives a flat list of ids and an `(m, d)` matrix of matching gradient rows.',
    'Try `dE[ids] += g` on `ids = [2, 2]`. Row 2 is updated once, not twice: the right side is computed for all positions and then written, so the last write wins. You need an accumulating scatter.',
    '`np.add.at(dE, flat_ids, flat_grads)` adds without buffering. A matrix-product alternative: build a one-hot `(m, V)` matrix and compute `onehot.T @ flat_grads`.',
]

_r = np.random.default_rng(13)
TESTS = [
    (np.array([0, 2, 2]), np.array([[1.0, 0.0], [0.5, 0.5], [2.0, -1.0]]), 4),
    (np.array([[1, -1], [1, 3]]), np.array([[[1.0], [9.0]], [[2.0], [3.0]]]), 4),
    (np.array([-1, -1]), np.ones((2, 3)), 2),
    (np.array([5]), np.array([[1e8, -1e8]]), 6),
    (np.array([[0, 0, 0, 0]]), np.arange(8.0).reshape(1, 4, 2), 1),
    (np.array([3, 0, 3]), np.array([[1.0], [2.0], [4.0]]), 5, 0),
    (_r.integers(-1, 6, size=(4, 7)), _r.normal(size=(4, 7, 3)), 6),
    (_r.integers(0, 50, size=(3, 40)), _r.normal(size=(3, 40, 5)), 50),
]

EXPLANATION = r"""
## Why it works
The forward pass `out = E[ids]` copies row `ids_p` of `E` to position `p`, so each row of `E` influences every position where it was looked up. By the chain rule its gradient is the sum of those positions' gradients: a **scatter-add**, the transpose of a gather. Masking out `pad_id` removes positions that should not train anything.

`np.add.at` is the unbuffered version of `+=`. For each index it reads, adds and writes back before moving on, so duplicates accumulate. A plain `dE[idx] += vals` evaluates `dE[idx] + vals` for all positions at once and then assigns, so for a repeated row only the last write survives.

## Complexity
$O(m d)$ for $m$ non-pad positions. `np.add.at` is slower per element than vectorised ops; for big batches, `np.bincount` per column or a sparse one-hot matrix product is often faster.

## Common bugs
- `dE[ids] += grad_out`, which silently undercounts frequent tokens. Tests with distinct ids still pass, which is why this bug survives code review.
- Leaving the padding in with `pad_id = -1`: index `-1` wraps around and pushes padding gradients into the **last** vocabulary row.
- Flattening `grad_out` with the wrong shape. Use the boolean mask, or `reshape(-1, d)`, so positions and rows stay aligned.

## Interview follow-ups
- Why do frameworks offer sparse embedding gradients? (Only the rows that appeared are non-zero, so a dense `(V, d)` gradient for a 100k vocabulary wastes memory and optimizer work.)
- With weight tying, the same `E` is also the output projection. How does its gradient combine? (The two contributions add: the scatter-add from the lookup plus the dense matmul gradient from the logits.)
"""
