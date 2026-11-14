import numpy as np

META = dict(title='Capstone: train a two-tower retriever with logQ correction', tags=['Recommender Systems', 'Training'], diff='Hard', runtime='numpy',
            related=['two-tower-scores', 'in-batch-softmax-logq', 'top-k-recommendations', 'recall-tail-coverage'])

STATEMENT = r"""
Train linear user and item towers with in-batch negatives for several steps, then evaluate full-catalogue retrieval.

Write `train_two_tower(Xu, Xi, batches, q, Wu, Wi, tau, lr, use_logq, seen, held_out, k, head_frac)`:

**Training.** `batches` is a list of batches; each batch is a list of `(user, item)` positive pairs. For each batch, in order:
1. $Y_u = X_u[\text{users}]\,W_u$ and $Y_i = X_i[\text{items}]\,W_i$ (linear towers), then $U$ and $V$ are their rows scaled to unit length.
2. Logits $Z_{ab} = U_a \cdot V_b / \tau$, minus $\log q_{\text{item}_b}$ for every column if `use_logq`. Row $a$'s target is column $a$.
3. Loss: mean row-wise cross-entropy. With $G = (\text{softmax}(Z) - I)/B$: $g_U = G V/\tau$ and $g_V = G^\top U/\tau$. Through the normalisation, $\partial Y = (g - z\,(z \cdot g))/\lVert y \rVert$ row by row. Then $\partial W_u = X_u[\text{users}]^\top \partial Y_u$, $\partial W_i = X_i[\text{items}]^\top \partial Y_i$. Take one SGD step on both.

Record each batch's loss **before** its step.

**Evaluation** (with the final weights). For every user, score all items by the cosine of their tower outputs, remove items in `seen[u]`, take the top `k` (highest score first, lower id on ties), and compute `recall`, `tail_recall` and `coverage` as in the retrieval-report step, using `q` as the popularity.

Return a dict with `losses` (list) and the three metrics.
"""

STARTER = '''import numpy as np

def train_two_tower(Xu, Xi, batches, q, Wu, Wi, tau, lr, use_logq, seen, held_out, k, head_frac):
    """In-batch softmax training of linear towers (optionally logQ-corrected), then top-k retrieval metrics."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _report(recs, held_out, pop, k, head_frac):
    pop = np.asarray(pop, float); n = len(pop)
    head = set(np.lexsort((np.arange(n), -pop))[:int(np.ceil(head_frac * n))].tolist())
    rec, tail, shown = [], [], set()
    for r, h in zip(recs, held_out):
        top = set(r[:k]); shown |= top
        if h:
            rec.append(len(top & set(h)) / len(set(h)))
        ht = set(h) - head
        if ht:
            tail.append(len(top & ht) / len(ht))
    return float(np.mean(rec)) if rec else 0.0, float(np.mean(tail)) if tail else 0.0, len(shown) / n

def train_two_tower(Xu, Xi, batches, q, Wu, Wi, tau, lr, use_logq, seen, held_out, k, head_frac):
    Xu, Xi, Wu, Wi = (np.asarray(a, float) for a in (Xu, Xi, Wu, Wi))
    logq = np.log(np.asarray(q, float))
    losses = []
    for batch in batches:
        us = np.array([p[0] for p in batch]); its = np.array([p[1] for p in batch])
        Au, Ai = Xu[us], Xi[its]
        Yu, Yi = Au @ Wu, Ai @ Wi
        ru, ri = np.linalg.norm(Yu, axis=1, keepdims=True), np.linalg.norm(Yi, axis=1, keepdims=True)
        U, V = Yu / ru, Yi / ri
        Z = U @ V.T / tau
        if use_logq:
            Z = Z - logq[its][None, :]
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        B = len(batch)
        losses.append(float(-np.log(P[np.arange(B), np.arange(B)]).mean()))
        G = (P - np.eye(B)) / B
        gU, gV = G @ V / tau, G.T @ U / tau
        dU = (gU - U * (U * gU).sum(1, keepdims=True)) / ru
        dV = (gV - V * (V * gV).sum(1, keepdims=True)) / ri
        Wu, Wi = Wu - lr * Au.T @ dU, Wi - lr * Ai.T @ dV
    Eu, Ei = Xu @ Wu, Xi @ Wi
    S = (Eu / np.linalg.norm(Eu, axis=1, keepdims=True)) @ (Ei / np.linalg.norm(Ei, axis=1, keepdims=True)).T
    recs = []
    for u in range(len(S)):
        s = S[u].copy()
        s[list(seen[u])] = -np.inf
        order = np.lexsort((np.arange(len(s)), -s))
        recs.append([int(i) for i in order[:k] if np.isfinite(s[i])])
    r, t, c = _report(recs, held_out, q, k, head_frac)
    return {"losses": losses, "recall": r, "tail_recall": t, "coverage": c}
'''

HINTS = [
    'Subtracting log q from a column lowers the logit of a popular item when it acts as a negative, so the model no longer has to push it down just for appearing in many batches.',
    'Exclude seen items by setting their scores to -inf before ranking.',
    'Popularity q should be the sampling probability of each item in the batches, here the training interaction frequency.',
]

def _world(seed, nu, ni, d, f, n_batches, B, n_seen, n_held):
    g = np.random.default_rng(seed)
    zu, zi = g.normal(size=(nu, d)), g.normal(size=(ni, d))
    pop_bias = np.linspace(2.0, -2.0, ni)
    Xu = zu @ g.normal(size=(d, f)) + 0.3 * g.normal(size=(nu, f))
    Xi = zi @ g.normal(size=(d, f)) + 0.3 * g.normal(size=(ni, f))
    logits = zu @ zi.T + pop_bias[None]
    P = np.exp(logits - logits.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    seen, held, train_pairs = [], [], []
    for u in range(nu):
        items = g.choice(ni, size=n_seen + n_held, replace=False, p=P[u])
        seen.append(items[:n_seen].tolist()); held.append(items[n_seen:].tolist())
        train_pairs += [(u, int(i)) for i in items[:n_seen]]
    counts = np.bincount([i for _, i in train_pairs], minlength=ni) + 1.0
    q = counts / counts.sum()
    batches = []
    for _ in range(n_batches):
        idx = g.choice(len(train_pairs), size=B, replace=False)
        batches.append([train_pairs[j] for j in idx])
    Wu, Wi = 0.1 * g.normal(size=(f, d)), 0.1 * g.normal(size=(f, d))
    return Xu, Xi, batches, q.tolist(), Wu, Wi, seen, held

def _test(seed, use_logq, tau, lr, k, n_batches=40):
    Xu, Xi, batches, q, Wu, Wi, seen, held = _world(seed, 30, 40, 4, 8, n_batches, 16, 5, 3)
    return (Xu, Xi, batches, q, Wu, Wi, tau, lr, use_logq, seen, held, k, 0.2)

TESTS = [
    _test(1, False, 0.1, 0.5, 5, n_batches=60),
    _test(1, True, 0.1, 0.5, 5, n_batches=60),
    _test(2, False, 0.2, 1.0, 5, n_batches=60),
    _test(2, True, 0.2, 1.0, 5, n_batches=60),
    _test(3, True, 0.2, 1.0, 10, n_batches=1),
]

EXPLANATION = r"""
## In-batch negatives and their bias
Using the other positives in a batch as negatives is free, but popular items appear in many batches and so are used as negatives far more often than rare ones. Training then pushes popular items down too hard. The logQ correction (Yi et al., 2019) subtracts the log of each item's sampling probability from its logit, which turns the sampled softmax into an estimate of the full softmax.

## Reading the tests
Random retrieval would reach a recall@5 of about 0.14 here (5 of roughly 35 unseen items), so every trained model is above chance, some only slightly. Compare the first two tests (same data and seed, with and without logQ), and the third and fourth. The correction changes which items get retrieved: watch tail recall and coverage as well as overall recall. On this small synthetic catalogue the effect is small and goes in different directions for the two seeds. On real catalogues with a steep popularity curve, the uncorrected model tends to under-rank popular items and over-recommend the tail; Yi et al. report that the correction improves recall on production data.

## Interview follow-ups
- In a streaming system the item frequencies change daily. How would you estimate q online?
"""
