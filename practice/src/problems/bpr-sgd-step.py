import numpy as np

META = dict(title='BPR loss and one SGD step on a batch of triples', tags=['Recommender Systems', 'Loss Functions', 'Optimization'],
            diff='Medium', runtime='numpy', lessons=['sdsd:12.08'],
            related=['uniform-negative-sampling', 'log-loss-binary-cross-entropy', 'sigmoid-stable'])

STATEMENT = r"""
Bayesian Personalised Ranking learns from implicit feedback by asking only that a user score an item they interacted with ($i$) above one they didn't ($j$). With user factors $P$ and item factors $Q$, a triple $(u, i, j)$ has margin

$$x_{uij} = p_u^\top (q_i - q_j)$$

and the batch loss over $B$ triples is

$$L = \frac{1}{B} \sum_{b=1}^{B} \Big[ -\log \sigma(x_b) + \frac{\lambda}{2}\big(\lVert p_{u_b} \rVert^2 + \lVert q_{i_b} \rVert^2 + \lVert q_{j_b} \rVert^2\big) \Big]$$

Write `bpr_step(P, Q, triples, lr, reg)` that computes $L$ at the **current** parameters, then takes one gradient-descent step on $L$ and returns `(loss, P_new, Q_new)`.

## Inputs
- `P`: `(n_users, d)` float. `Q`: `(n_items, d)` float.
- `triples`: `(B, 3)` int array of `(u, i, j)` rows, $B \ge 1$, with $i \ne j$ in each row. Users and items may repeat across rows.
- `lr`: step size. `reg`: $\lambda \ge 0$.

## Rules
- One step on the **mean** loss: every parameter moves by `-lr * dL/dparam`, computed from the old `P` and `Q`. Rows that appear in several triples accumulate their gradient contributions; the regulariser counts once per appearance, as in the formula.
- Rows of `P` and `Q` not touched by any triple are unchanged.
- Compute $-\log\sigma(x)$ stably as `np.logaddexp(0, -x)`.
- Don't modify the inputs. Return `loss` as a Python float and new arrays.
"""

STARTER = '''import numpy as np

def bpr_step(P, Q, triples, lr, reg):
    """Return (loss, P_new, Q_new)."""
    P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
    u, i, j = np.asarray(triples).T
    x = np.sum(P[u] * (Q[i] - Q[j]), axis=1)
    loss = float(np.mean(np.logaddexp(0, -x)))
    return loss, P.copy(), Q.copy()
'''

SOLUTION = '''import numpy as np

def bpr_step(P, Q, triples, lr, reg):
    P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
    u, i, j = np.asarray(triples).T
    B = len(u)
    pu, qi, qj = P[u], Q[i], Q[j]
    x = np.sum(pu * (qi - qj), axis=1)
    sq = (pu ** 2).sum(1) + (qi ** 2).sum(1) + (qj ** 2).sum(1)
    loss = float(np.mean(np.logaddexp(0, -x) + 0.5 * reg * sq))

    g = (-1.0 / (1.0 + np.exp(x)))[:, None] / B   # dL/dx = -sigmoid(-x) / B
    dP, dQ = np.zeros_like(P), np.zeros_like(Q)
    np.add.at(dP, u, g * (qi - qj) + reg * pu / B)   # add.at accumulates repeated indices
    np.add.at(dQ, i, g * pu + reg * qi / B)
    np.add.at(dQ, j, -g * pu + reg * qj / B)
    return loss, P - lr * dP, Q - lr * dQ
'''

HINTS = [
    'Differentiate $-\\log\\sigma(x)$: the derivative is $-\\sigma(-x)$. The chain rule then gives $\\partial x/\\partial p_u = q_i - q_j$, $\\partial x/\\partial q_i = p_u$, $\\partial x/\\partial q_j = -p_u$.',
    'Fancy-index assignment `dP[u] += ...` silently drops repeated indices. Use `np.add.at(dP, u, ...)` so a user in two triples gets both contributions.',
    'Every term carries the $1/B$ from the mean, including the regulariser: `reg * P[u] / B` per appearance.',
]

_r = np.random.default_rng(31)
_P = _r.normal(0, 0.3, size=(6, 4))
_Q = _r.normal(0, 0.3, size=(10, 4))
_T = np.column_stack([_r.integers(0, 6, 20), _r.integers(0, 5, 20), _r.integers(5, 10, 20)])

TESTS = [
    (np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[0, 0, 1]]), 0.1, 0.0),
    (np.array([[0.5, -0.5], [1.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
     np.array([[0, 0, 1], [1, 2, 0]]), 0.5, 0.1),
    (np.array([[1.0, 2.0]]), np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]), np.array([[0, 0, 1], [0, 0, 2]]), 1.0, 0.0),
    (np.zeros((2, 3)), np.zeros((3, 3)), np.array([[1, 0, 2]]), 1.0, 0.5),
    (np.array([[30.0]]), np.array([[2.0], [-2.0]]), np.array([[0, 1, 0]]), 0.01, 0.0),
    (np.array([[1.0, 1.0], [2.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0]]), 0.2, 0.05),
    (_P, _Q, _T, 0.05, 0.01),
]

EXPLANATION = r"""
## Why it works
BPR maximises the log-likelihood that each observed item beats each sampled unobserved one, under a logistic model of the margin. The gradient of $-\log\sigma(x)$ is $-\sigma(-x)$: close to $-1$ when the pair is misordered, close to 0 when it is already well separated, so well-ranked pairs stop pushing. Each triple then pulls $p_u$ toward $q_i - q_j$, pulls $q_i$ toward $p_u$ and pushes $q_j$ away from it. Test 5 has a margin of $-120$: a naive `-np.log(1/(1+np.exp(-x)))` overflows there, while `logaddexp` returns 120 exactly.

## Complexity
$O(Bd)$ time and $O(Bd)$ extra memory for the gathered rows; the update touches only rows in the batch.

## Common bugs
- `dP[u] += ...` with repeated `u`: NumPy buffers fancy-index updates, so only the last contribution survives (tests 3 and 6 repeat rows).
- Updating $P$ first and then using the new $P$ to compute $Q$'s gradient. Both gradients must come from the old parameters.
- Forgetting the $1/B$ on the regulariser, which makes regularisation strength depend on batch size.

## Interview follow-ups
- BPR optimises a smoothed AUC. Why might it underperform on top-heavy metrics like NDCG@10, and what changes help? (Uniform negatives are mostly easy and contribute little gradient; adaptive/hard negative sampling or WARP-style rank-weighted losses focus on the top.)
- How does BPR differ from treating the problem as binary classification with log loss on (positive, negative) labels? (Pairwise vs pointwise: BPR only cares about the order within a user, not calibrated probabilities.)
"""
