import numpy as np

META = dict(title='Scaled dot-product attention with a mask', tags=['Attention', 'Transformers'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:06.02', 'aieng:04.02', 'aieng:05.01'],
            related=['softmax-rows', 'sinusoidal-positional-encoding', 'layer-norm'])

STATEMENT = r"""
Attention lets each query mix the values whose keys it matches best:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

Write `attention(Q, K, V, mask=None)`.

## Inputs
- `Q`: `(n, d_k)` queries, `K`: `(m, d_k)` keys, `V`: `(m, d_v)` values.
- `mask`: optional boolean `(n, m)`. `True` means query $i$ **may** attend to key $j$. Masked positions get $-\infty$ before the softmax.

## Output
- `(n, d_v)` array.
- If a query row has **no** allowed keys, its output row is all zeros (not `nan`).
"""

STARTER = '''import numpy as np

def attention(Q, K, V, mask=None):
    """Q: (n, dk), K: (m, dk), V: (m, dv), mask: bool (n, m) or None. Return (n, dv)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def attention(Q, K, V, mask=None):
    Q, K, V = (np.asarray(x, dtype=float) for x in (Q, K, V))
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    row_max = scores.max(axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)     # fully masked row: avoid -inf - -inf
    w = np.exp(scores - row_max)                               # exp(-inf) = 0 for masked keys
    total = w.sum(axis=-1, keepdims=True)
    w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    return w @ V
'''

HINTS = [
    'Build the `(n, m)` score matrix with one product, `Q @ K.T`, then divide by `sqrt(d_k)`.',
    'Apply the mask with `np.where(mask, scores, -np.inf)`, then take a stable softmax per row. `exp(-inf)` is exactly 0, so masked keys get zero weight.',
    'A row that is entirely `-inf` has max `-inf`, and `-inf - -inf` is `nan`. Replace a non-finite row max with 0, and guard the division so all-zero rows stay zero.',
]

_r = np.random.default_rng(42)
n, m, dk, dv = 4, 5, 8, 3
causal = np.tril(np.ones((4, 4), dtype=bool))
TESTS = [
    (np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[10.0], [20.0]])),
    (_r.normal(size=(n, dk)), _r.normal(size=(m, dk)), _r.normal(size=(m, dv))),
    (_r.normal(size=(4, dk)), _r.normal(size=(4, dk)), _r.normal(size=(4, dv)), causal),
    (np.ones((2, 2)), np.ones((3, 2)), np.arange(6.0).reshape(3, 2), np.array([[True, False, True], [False, False, False]])),
    (np.zeros((3, 4)), _r.normal(size=(3, 4)), _r.normal(size=(3, 2))),
    (_r.normal(size=(2, 4)) * 50, _r.normal(size=(6, 4)) * 50, _r.normal(size=(6, 5))),
]

EXPLANATION = r"""
## Why divide by $\sqrt{d_k}$
If the entries of $q$ and $k$ are independent with unit variance, $q\cdot k$ has variance $d_k$. Without scaling, scores grow with the head size, softmax saturates toward one-hot, and its gradient vanishes. Dividing by $\sqrt{d_k}$ brings the variance back to 1.

## Masks
- **Causal** (decoder LMs): query $i$ sees keys $j \le i$. That's `np.tril`, test 3.
- **Padding**: hide padded keys so real tokens never attend to them.
Setting scores to $-\infty$ (not just a large negative number) guarantees exactly zero weight. You then have to handle the fully masked row yourself, and that is the bug interviewers probe.

## Complexity
$O(n m d_k)$ for scores and $O(n m d_v)$ for the mix. The $(n, m)$ matrix is the memory problem for long contexts, and FlashAttention exists to avoid ever materialising it.

## Interview follow-ups
- Extend to multi-head: reshape to `(h, n, d_k/h)`, attend per head, concatenate, project.
- Why does a KV cache make decoding $O(m)$ per new token instead of $O(m^2)$?
"""
