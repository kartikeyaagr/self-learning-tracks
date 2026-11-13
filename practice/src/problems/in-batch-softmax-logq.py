import numpy as np

META = dict(title='Two-tower in-batch softmax with logQ correction', tags=['Recommender Systems', 'Embeddings', 'Loss Functions'],
            diff='Hard', runtime='numpy', lessons=['sdsd:16.05', 'aieng:02.06'],
            related=['softmax-cross-entropy-gradient', 'clip-cosine-retrieval', 'uniform-negative-sampling'])

STATEMENT = r"""
A two-tower retrieval model embeds users (queries) and items separately and scores pairs with a dot product, so serving is a nearest-neighbour search. It is commonly trained with an **in-batch softmax**: each row $b$ of a batch is a positive pair $(u_b, v_b)$, and the other rows' items act as negatives.

Popular items appear as in-batch negatives far more often than rare ones, which biases the model against them. The **logQ correction** subtracts the log of each item's sampling probability $q$ from its logit. Two rows can also hold the **same** item; that item is then a positive for both, so the duplicate must not act as a negative.

For a batch of $B$ rows, with temperature $\tau$:

$$\ell_{bj} = \frac{u_b^\top v_j}{\tau} - \log q_j, \qquad \ell_{bj} = -\infty \text{ if } j \ne b \text{ and } \mathrm{id}_j = \mathrm{id}_b$$

$$L = -\frac{1}{B} \sum_{b=1}^{B} \log \frac{e^{\ell_{bb}}}{\sum_{j=1}^{B} e^{\ell_{bj}}}$$

Write `in_batch_softmax(U, V, item_ids, q, tau)` that returns `(loss, dU, dV)`: the loss as a Python float and its gradients with respect to `U` and `V`.

## Inputs
- `U`: `(B, d)` user-tower outputs. `V`: `(B, d)` item-tower outputs; row $b$ is the positive item for row $b$ of `U`.
- `item_ids`: `(B,)` int ids of the items in `V`. Rows with equal ids have equal `V` rows in real data, but don't rely on it: treat `V` row by row.
- `q`: `(B,)` sampling probabilities, $0 < q_j \le 1$, for the item in row $j$.
- `tau` $> 0$.

## Rules
- The correction is applied to every column, including the positive $j = b$.
- Use a numerically stable log-sum-exp (subtract the row max). Masked entries contribute exactly 0 probability and 0 gradient.
- Return arrays with the same shapes as `U` and `V`.
"""

STARTER = '''import numpy as np

def in_batch_softmax(U, V, item_ids, q, tau):
    """Return (loss, dU, dV)."""
    U, V = np.asarray(U, dtype=float), np.asarray(V, dtype=float)
    logits = U @ V.T / tau
    logits -= logits.max(1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    return float(-np.mean(np.diag(logp))), np.zeros_like(U), np.zeros_like(V)
'''

SOLUTION = '''import numpy as np

def in_batch_softmax(U, V, item_ids, q, tau):
    U, V = np.asarray(U, dtype=float), np.asarray(V, dtype=float)
    ids, B = np.asarray(item_ids), len(U)
    logits = U @ V.T / tau - np.log(np.asarray(q, dtype=float))[None, :]
    dup = (ids[:, None] == ids[None, :]) & ~np.eye(B, dtype=bool)   # same item in another row
    logits[dup] = -np.inf

    m = logits.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(1))
    loss = float(np.mean(lse - np.diag(logits)))

    P = np.exp(logits - lse[:, None])           # softmax rows; masked entries are exactly 0
    G = (P - np.eye(B)) / (B * tau)             # dL/d(u_b . v_j); log q is a constant shift
    return loss, G @ V, G.T @ U
'''

HINTS = [
    'Build the full $(B, B)$ logit matrix with `U @ V.T / tau - np.log(q)[None, :]`, then set duplicate off-diagonal entries to `-np.inf` with a boolean mask from `ids[:, None] == ids[None, :]`.',
    'Softmax cross-entropy with the targets on the diagonal has gradient $(P - I)/B$ with respect to the logits. The $-\\log q$ term doesn\'t depend on the embeddings, so it only changes $P$.',
    'Chain through $\\ell = UV^\\top/\\tau$: with $G = (P - I)/(B\\tau)$, the gradients are `G @ V` for `U` and `G.T @ U` for `V`.',
]

_r = np.random.default_rng(77)
_U = _r.normal(size=(8, 4)); _U /= np.linalg.norm(_U, axis=1, keepdims=True)
_ids = np.array([3, 1, 4, 1, 5, 9, 2, 6])
_table = _r.normal(size=(10, 4)); _table /= np.linalg.norm(_table, axis=1, keepdims=True)
_V = _table[_ids]
_q = np.array([0.05, 0.3, 0.02, 0.3, 0.1, 0.01, 0.15, 0.07])

TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]), np.array([0.5, 0.5]), 1.0),
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]), np.array([0.9, 0.1]), 1.0),
    (np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]), np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
     np.array([7, 7, 2]), np.array([0.4, 0.4, 0.2]), 0.5),
    (np.array([[2.0]]), np.array([[3.0]]), np.array([0]), np.array([1.0]), 1.0),
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0, 1]), np.array([1.0, 1.0]), 0.01),
    (np.ones((3, 2)), np.ones((3, 2)), np.array([5, 5, 5]), np.array([0.2, 0.2, 0.2]), 1.0),
    (_U, _V, _ids, _q, 0.1),
    (_U, _V, np.arange(8), np.full(8, 1 / 8), 0.05),
]

EXPLANATION = r"""
## Why it works
The in-batch softmax treats each row as a $B$-way classification: pick the true item among the batch's items. Negatives are the other rows' **positives**, so they are drawn from the item frequency distribution $q$, not uniformly. A sampled softmax is an unbiased estimate of the full softmax only when each logit is corrected by $-\log q_j$: without it, popular items are pushed down simply for being common (test 2 vs test 1: the same embeddings, but the rarer item gets a logit bonus of $\log 9$). Masking duplicates removes false negatives: in test 3 rows 0 and 1 share item 7, and in test 6 every row is the same item, so each row's softmax has a single live entry and the loss and gradients are exactly 0.

The gradient is the familiar softmax cross-entropy one, $(P - I)/B$ on the logits, chained through the bilinear score. Masked entries have $P = 0$ and no target, so they drop out automatically.

## Complexity
$O(B^2 d)$ for the score matrix and both gradients, $O(B^2)$ memory. Large batches matter: more in-batch negatives is the cheapest way to make retrieval training harder.

## Common bugs
- Adding $+\log q$ instead of subtracting, which amplifies the popularity bias.
- Masking the diagonal along with the duplicates, which removes the positive itself and makes the loss infinite.
- Masking with a large negative constant such as `-1e9` works in float64 but leaves tiny non-zero probabilities in float16/bfloat16 training; `-inf` (with a live diagonal in every row) keeps them exactly 0.
- Overflow with small $\tau$ (test 5 has $\tau = 0.01$, logits around $\pm 100$) when skipping the max subtraction.

## Interview follow-ups
- $q$ isn't known in a streaming setting. How do you estimate it? (Count-min style hashed counters of the gap between consecutive occurrences of each item; the gap's inverse estimates its frequency.)
- In-batch negatives never include items that are rarely positives. How do you fix that? (Mix in uniformly sampled negatives from a cache or a separate item stream, with their own $\log q$ correction.)
"""
