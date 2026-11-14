import numpy as np

META = dict(title='Full ranking versus 100 sampled negatives', tags=['Recommender Systems', 'Evaluation Metrics'], diff='Medium', runtime='numpy',
            related=['hit-rate-and-mrr', 'ndcg-at-k'])

STATEMENT = r"""
Many papers rank each held-out item against 100 random negatives instead of the whole catalogue. Compare the two protocols.

Write `rank_metrics(scores, target, seen, negatives, k)`:

- `scores (U, I)`: model scores. `target[u]`: the held-out item. `seen[u]`: list of training items (never ranked). `negatives[u]`: the sampled negative items for user `u` (none is the target or seen).
- **Full rank**: 1 + the number of items, other than the target and the seen items, with a score strictly above the target's.
- **Sampled rank**: 1 + the number of `negatives[u]` with a score strictly above the target's.
- For each protocol: `hr` = mean of [rank ≤ k], `ndcg` = mean of $1/\log_2(\text{rank}+1)$ when rank ≤ k (else 0).

Return a dict with `full_rank`, `sampled_rank` (lists of ints), `full_hr`, `full_ndcg`, `sampled_hr`, `sampled_ndcg`.
"""

STARTER = '''import numpy as np

def rank_metrics(scores, target, seen, negatives, k):
    """Rank of each held-out item against all items and against sampled negatives, with HR@k and NDCG@k."""
    S = np.asarray(scores, float)
    full = [int((S[u] > S[u, t]).sum()) for u, t in enumerate(target)]
    samp = [int((S[u, negatives[u]] > S[u, t]).sum()) for u, t in enumerate(target)]
    hr = lambda r: float(np.mean([x <= k for x in r]))
    return {"full_rank": full, "sampled_rank": samp, "full_hr": hr(full), "full_ndcg": 0.0, "sampled_hr": hr(samp), "sampled_ndcg": 0.0}
'''

SOLUTION = '''import numpy as np

def rank_metrics(scores, target, seen, negatives, k):
    S = np.asarray(scores, float)
    full, samp = [], []
    for u, t in enumerate(target):
        s = S[u].copy()
        mask = np.ones(len(s), bool); mask[t] = False; mask[list(seen[u])] = False
        full.append(1 + int((s[mask] > s[t]).sum()))
        samp.append(1 + int((s[list(negatives[u])] > s[t]).sum()))
    hr = lambda r: float(np.mean([x <= k for x in r]))
    nd = lambda r: float(np.mean([1 / np.log2(x + 1) if x <= k else 0.0 for x in r]))
    return {"full_rank": full, "sampled_rank": samp, "full_hr": hr(full), "full_ndcg": nd(full),
            "sampled_hr": hr(samp), "sampled_ndcg": nd(samp)}
'''

HINTS = [
    'Ranks start at 1: the target is first when nothing scores above it.',
    'Exclude the seen items from the full ranking; they would never be recommended.',
    'With 100 negatives out of 10,000 items, a target at full rank 500 lands around sampled rank 6.',
]

def _case(seed, U, I, n_seen, n_neg, skill):
    g = np.random.default_rng(seed)
    S = g.normal(size=(U, I))
    target, seen, negs = [], [], []
    for u in range(U):
        perm = g.permutation(I)
        t = int(perm[0]); target.append(t)
        seen.append(perm[1:1 + n_seen].tolist())
        negs.append(g.choice(perm[1 + n_seen:], size=n_neg, replace=False).tolist())
        S[u, t] += skill
    return (S, target, seen, negs)

TESTS = [
    ([[3.0, 2.0, 1.0, 0.0]], [1], [[0]], [[2, 3]], 1),
    ([[0.0, 5.0, 1.0, 2.0]], [0], [[]], [[2]], 2),
    _case(1, 20, 1000, 5, 100, 2.0) + (10,),
    _case(2, 20, 1000, 5, 100, 0.0) + (10,),
    _case(3, 50, 2000, 10, 100, 3.0) + (10,),
]

EXPLANATION = r"""
## Why sampled metrics flatter
A target that beats 99% of items will usually beat all 100 sampled negatives, so HR@10 under sampling can be near 1 while the real HR@10 over the full catalogue is small. The third test shows the gap.

## Worse than flattering
Krichene and Rendle (2020) showed that sampled metrics can change the **ordering** of models: a model that is better under full ranking can look worse under sampling, because sampling mostly rewards getting the target out of the bottom of the list rather than into the top. Ranking against everything, or correcting the sampled metric, is the honest choice when the catalogue fits in memory.

## The fourth test
With no signal (`skill = 0`) the expected full HR@10 is about 10/994 ≈ 0.01, while the expected sampled HR@10 is about 10/101 ≈ 0.10: a random model looks ten times better under sampling. With only 20 users the test happens to see 0.0 and 0.05.

## Interview follow-ups
- If full ranking is too expensive, what would you do instead of uniform sampling?
"""
