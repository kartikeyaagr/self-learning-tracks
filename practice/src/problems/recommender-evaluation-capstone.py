import numpy as np

META = dict(title='Capstone: evaluate two recommenders honestly', tags=['Recommender Systems', 'Evaluation Metrics'], diff='Hard', runtime='numpy',
            related=['precision-recall-at-k', 'hit-rate-and-mrr', 'ndcg-at-k', 'catalog-coverage-and-novelty', 'time-split-with-gap'])

STATEMENT = r"""
Compare a popularity baseline with an item co-occurrence recommender under a time-based split.

Write `evaluate(interactions, n_items, cutoff, gap, k)`, where each interaction is `[user, item, t]`:

1. **Split.** Train on rows with `t < cutoff - gap`, test on rows with `t >= cutoff`; rows in between are dropped. Duplicate (user, item) pairs count once.
2. **Users.** Evaluate users with at least one training item whose relevant set (test items not among their training items) is non-empty.
3. **Models.** `pop[j]` = number of training users with item `j`.
   - `popularity`: score = `pop[j]`.
   - `cooc`: score $= \sum_{i \in \text{train}(u)} C_{ij}$, where $C_{ij}$ is the number of training users who have both $i$ and $j$ ($i \ne j$).
   Rank by score, then by `pop` (both descending), then by lower id. Exclude the user's training items. Keep the top `k`.
4. **Metrics @k**, averaged over evaluated users: `precision`, `recall`, `ndcg` (binary gains, ideal DCG over $\min(|R|, k)$ hits), `hit_rate`, `mrr` (reciprocal rank of the first hit in the top `k`, else 0), and `coverage` (distinct recommended items / `n_items`).

Return `{"n_users": ..., "popularity": {...}, "cooc": {...}}`.
"""

STARTER = '''import numpy as np

def evaluate(interactions, n_items, cutoff, gap, k):
    """Time split with a gap, popularity and co-occurrence recommenders, and six ranking metrics each."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _metrics(recs, rel, k, n_items):
    P, R, N, H, M, shown = [], [], [], [], [], set()
    for r, s in zip(recs, rel):
        hits = [1.0 if i in s else 0.0 for i in r[:k]]
        shown |= set(r[:k])
        P.append(sum(hits) / k); R.append(sum(hits) / len(s)); H.append(float(any(hits)))
        dcg = sum(h / np.log2(j + 2) for j, h in enumerate(hits))
        idcg = sum(1 / np.log2(j + 2) for j in range(min(len(s), k)))
        N.append(dcg / idcg)
        M.append(next((1.0 / (j + 1) for j, h in enumerate(hits) if h), 0.0))
    return {"precision": float(np.mean(P)), "recall": float(np.mean(R)), "ndcg": float(np.mean(N)),
            "hit_rate": float(np.mean(H)), "mrr": float(np.mean(M)), "coverage": len(shown) / n_items}

def evaluate(interactions, n_items, cutoff, gap, k):
    train, test = {}, {}
    for u, i, t in interactions:
        if t < cutoff - gap:
            train.setdefault(u, set()).add(i)
        elif t >= cutoff:
            test.setdefault(u, set()).add(i)
    X = np.zeros((max(list(train) + list(test) + [0]) + 1, n_items))
    for u, s in train.items():
        X[u, list(s)] = 1.0
    pop = X.sum(0)
    C = X.T @ X
    np.fill_diagonal(C, 0.0)
    users, rel = [], []
    for u in sorted(train):
        r = test.get(u, set()) - train[u]
        if r:
            users.append(u); rel.append(r)
    out = {"n_users": len(users)}
    ids = np.arange(n_items)
    for name in ("popularity", "cooc"):
        recs = []
        for u in users:
            sc = pop.copy() if name == "popularity" else X[u] @ C
            order = np.lexsort((ids, -pop, -sc))
            recs.append([int(j) for j in order if j not in train[u]][:k])
        out[name] = _metrics(recs, rel, k, n_items)
    return out
'''

HINTS = [
    'Build a user-by-item 0/1 matrix from the training rows; popularity is its column sum and co-occurrence is X^T X with the diagonal zeroed.',
    'The gap drops interactions just before the cutoff, so a model cannot exploit "what the user did a minute before the test period".',
    'Recall divides by the number of relevant items; precision divides by k.',
]

def _log(seed, n_users, n_items, n_groups, per_user, t_max):
    g = np.random.default_rng(seed)
    groups = np.arange(n_items) % n_groups
    pop = np.exp(-np.arange(n_items) / (n_items / 4))
    rows = []
    for u in range(n_users):
        gu = g.integers(n_groups)
        w = pop * np.where(groups == gu, 6.0, 1.0); w /= w.sum()
        items = g.choice(n_items, size=per_user, replace=False, p=w)
        ts = np.sort(g.uniform(0, t_max, per_user))
        rows += [[u, int(i), float(round(t, 3))] for i, t in zip(items, ts)]
    return rows

_L1 = _log(1, 60, 40, 4, 12, 100.0)
_L2 = _log(2, 120, 80, 5, 15, 100.0)

TESTS = [
    (_L1, 40, 70.0, 5.0, 5),
    (_L1, 40, 70.0, 0.0, 5),
    (_L2, 80, 75.0, 5.0, 10),
    (_L2, 80, 50.0, 10.0, 10),
    ([[0, 0, 1.0], [0, 1, 2.0], [1, 0, 1.0], [1, 2, 2.0], [0, 2, 9.0], [1, 1, 9.0], [2, 1, 1.0], [2, 3, 9.0]], 4, 5.0, 1.0, 2),
]

EXPLANATION = r"""
## What an honest evaluation needs
Every piece from the steps appears here: a split that tests on the future, a gap that removes leakage from just before the cutoff, metrics that answer different questions (did the list find anything, how high, how much of the relevant set), and coverage to see what the model recommends at all.

## Reading the comparison
Popularity gives everyone nearly the same list; it differs between users only because each user's training items are removed. The co-occurrence model personalises. In these synthetic logs users mostly stay within one item group, and co-occurrence wins on every metric, but on the smaller log only by a few points. Popularity is usually a stronger baseline than expected, and a new model has to beat it under the same protocol before any claim is made.

## The small hand-built test
Train is everything before $t = 4$, test everything from $t = 5$. Users 0 and 1 each get their test item ranked first by both models. User 2's test item (3) never appears in training, so neither model can rank it above items with evidence, and both miss it: a cold-start item. Both models score a hit rate of 2/3, and the tie is the point: on tiny or sparse data the extra model buys nothing.

## Interview follow-ups
- Offline metrics improved but an online A/B test shows no change. List three reasons this happens with recommenders.
"""
