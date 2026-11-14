import numpy as np

META = dict(title='Recall at k of an IVF index against brute force', tags=['Recommender Systems', 'Inference', 'Metrics & Evaluation'], diff='Medium', runtime='numpy')

STATEMENT = r"""
Approximate nearest-neighbour indexes trade recall for speed. Write `ivf_recall(base, queries, centroids, k, nprobe)` that simulates an inverted-file index and measures it.

1. Assign every base vector to its nearest centroid (squared Euclidean, ties to the lower index).
2. For each query, find its `nprobe` nearest centroids, gather the base vectors in those lists, and return the `k` closest of them (ties to the lower base index).
3. Exact neighbours: the `k` closest base vectors by brute force, same tie rule.
4. Recall@k for a query is |approx ∩ exact| / k.

Return `{"recall": mean recall, "scanned_frac": mean fraction of base vectors scanned per query}`.
"""

STARTER = '''import numpy as np

def ivf_recall(base, queries, centroids, k, nprobe):
    """Return {"recall", "scanned_frac"}."""
    return {"recall": 1.0, "scanned_frac": nprobe / len(centroids)}
'''

SOLUTION = '''import numpy as np

def _d2(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)

def _topk(d, k):
    return np.lexsort((np.arange(len(d)), d))[:k]

def ivf_recall(base, queries, centroids, k, nprobe):
    X, Q, C = (np.asarray(a, float) for a in (base, queries, centroids))
    assign = _d2(X, C).argmin(1)
    DQ, DC = _d2(Q, X), _d2(Q, C)
    rec, scan = [], []
    for q in range(len(Q)):
        lists = _topk(DC[q], nprobe)
        cand = np.flatnonzero(np.isin(assign, lists))
        approx = cand[_topk(DQ[q][cand], k)]
        exact = _topk(DQ[q], k)
        rec.append(len(set(approx.tolist()) & set(exact.tolist())) / k)
        scan.append(len(cand) / len(X))
    return {"recall": float(np.mean(rec)), "scanned_frac": float(np.mean(scan))}
'''

HINTS = [
    'np.lexsort((indices, distances)) sorts by distance with index as the tie-break.',
    'A probed list can be small, so fewer than k candidates is possible; recall then drops naturally.',
    'Scanned fraction is the cost proxy: recall is only meaningful next to it.',
]

def _case(seed, n, d, nc, nq, k, nprobe):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    C = X[r.choice(n, nc, replace=False)]
    return (X, r.normal(size=(nq, d)), C, k, nprobe)

TESTS = [_case(1, 300, 4, 10, 20, 5, 1), _case(1, 300, 4, 10, 20, 5, 3), _case(1, 300, 4, 10, 20, 5, 10),
         _case(2, 500, 8, 16, 10, 10, 2), ([[0.0], [1.0], [10.0], [11.0]], [[0.4]], [[0.0], [10.0]], 2, 1),
         ([[0.0], [1.0], [10.0], [11.0]], [[5.4]], [[0.0], [10.0]], 3, 1)]

EXPLANATION = r"""
## How IVF works
Cluster the base vectors (k-means in practice), keep one list per cluster, and at query time scan only the lists of the closest few centroids. Cost drops roughly by `nprobe / n_lists`; recall drops when true neighbours sit in clusters you did not probe, usually near cluster boundaries.

## Measuring it properly
Recall@k against brute force on a held-out query set is the standard check, and it must be reported with the cost (vectors scanned or latency). Sweep `nprobe` to get the recall-cost curve and choose the operating point the product needs.

## Why it matters for the system
Retrieval recall caps the quality of everything downstream: a reranker cannot rank an item the index never returned. A silent recall drop after re-indexing is a common cause of unexplained quality regressions.

## Interview follow-ups
- How do product quantisation and HNSW change this trade-off?
"""
