import numpy as np

META = dict(title='InfoNCE loss with in-batch negatives', tags=['Loss Functions', 'Embeddings'], diff='Hard', runtime='numpy',
            related=['symmetric-clip-contrastive-loss', 'triplet-loss', 'in-batch-softmax-logq', 'softmax-numerical-stability-and-log-sum-exp'])

STATEMENT = r"""
Write `info_nce(q, k, temperature, extra_negatives=None)`.

- `q`, `k`: `(n, d)` query and key embeddings. Row $i$ of `k` is the positive for row $i$ of `q`; the other rows of `k` are negatives (in-batch negatives).
- `extra_negatives`: optional `(m, d)` array of additional negative keys shared by all queries (for example, a memory queue).
- L2-normalise all embeddings first. Logits are cosine similarities divided by the temperature: $s_{ij} = \hat q_i \cdot \hat k_j / \tau$, for $j$ over the $n$ batch keys followed by the $m$ extra negatives.
- Loss: mean over $i$ of $-\log \dfrac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}$, computed with a stable log-sum-exp.

Return `(loss, accuracy)` where accuracy is the share of queries whose largest logit is their own positive (first index on ties).
"""

STARTER = '''import numpy as np

def info_nce(q, k, temperature, extra_negatives=None):
    """Return (mean_loss, top1_accuracy)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _norm(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def info_nce(q, k, temperature, extra_negatives=None):
    Q, K = _norm(q), _norm(k)
    if extra_negatives is not None and len(extra_negatives):
        K = np.vstack([K, _norm(extra_negatives)])
    S = Q @ K.T / temperature
    mx = S.max(axis=1, keepdims=True)
    lse = (mx + np.log(np.exp(S - mx).sum(axis=1, keepdims=True)))[:, 0]
    idx = np.arange(len(Q))
    loss = (lse - S[idx, idx]).mean()
    acc = (S.argmax(axis=1) == idx).mean()
    return float(loss), float(acc)
'''

HINTS = [
    'Normalise every row to unit length, then one matrix product gives all cosine similarities.',
    'The positive for query i sits in column i, so the loss is cross-entropy with targets 0, 1, ..., n-1.',
    'Subtract the row maximum before exponentiating; with small temperatures logits can be large.',
]

_r = np.random.default_rng(41)
_q = _r.normal(size=(6, 8))
TESTS = [
    (np.eye(3), np.eye(3), 1.0),
    (np.eye(3), np.eye(3), 0.05),
    (_q, _q + 0.1 * _r.normal(size=(6, 8)), 0.1),
    (_q, _r.normal(size=(6, 8)), 0.5),
    (_q, _q + 0.3 * _r.normal(size=(6, 8)), 0.07, _r.normal(size=(10, 8))),
    (np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 0.0], [1.0, 0.0]]), 1.0),
]

EXPLANATION = r"""
## A classification over the batch
Each query must pick its own key out of $n + m$ candidates, so InfoNCE is ordinary softmax cross-entropy whose classes are the other examples in the batch. That is why bigger batches (or a queue of extra negatives, as in MoCo) usually help: more negatives make the task harder and the bound on mutual information looser, $I \ge \log(n+m) - \mathcal{L}$.

## Temperature
Dividing by a small $\tau$ sharpens the softmax so that the hardest negatives dominate the gradient. Too small makes training unstable; CLIP learns $\tau$ as a parameter.

## Duplicates
If two batch items are identical (last test), one query's "negative" is actually a positive, which caps accuracy at chance for those rows. Real pipelines deduplicate or mask such pairs.

## Interview follow-ups
- Why is the loss symmetrised (query-to-key and key-to-query) in CLIP?
"""
