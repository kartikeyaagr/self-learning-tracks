META = dict(title='Mean average precision at k', tags=['Metrics & Evaluation', 'Recommender Systems'], diff='Easy', runtime='python',
            related=['precision-recall-at-k', 'ndcg-at-k'])

STATEMENT = r"""
Write `map_at_k(recommended, relevant, k)` where both arguments are lists with one entry per user.

For one user with ranked list `rec` and relevant set `rel`:

$$\mathrm{AP@}k = \frac{1}{\min(|rel|, k)} \sum_{j=1}^{k} P@j \cdot [\text{rec}_j \in rel]$$

where $P@j$ is the fraction of the first $j$ items that are relevant. Duplicate items in `rec` count only at their first occurrence. A user with an empty relevant set is **skipped**. Return the mean over the remaining users, or `0.0` if none remain.
"""

STARTER = '''def map_at_k(recommended, relevant, k):
    """Mean AP@k over users."""
    hits = [len(set(r[:k]) & set(t)) / k for r, t in zip(recommended, relevant)]
    return sum(hits) / len(hits)
'''

SOLUTION = '''def map_at_k(recommended, relevant, k):
    aps = []
    for rec, rel in zip(recommended, relevant):
        rel = set(rel)
        if not rel:
            continue
        seen, hits, s = set(), 0, 0.0
        for j, item in enumerate(rec[:k], start=1):
            if item in rel and item not in seen:
                hits += 1
                s += hits / j
            seen.add(item)
        aps.append(s / min(len(rel), k))
    return sum(aps) / len(aps) if aps else 0.0
'''

HINTS = [
    'Precision is added only at ranks where a relevant item appears.',
    'Normalise by min(|rel|, k) so a perfect list scores 1 even when there are more relevant items than slots.',
    'Track items already counted so a repeated item does not score twice.',
]

TESTS = [
    ([[1, 2, 3]], [[1, 3]], 3),
    ([[1, 2, 3], [4, 5, 6]], [[1, 3], [9]], 3),
    ([[1, 2, 3, 4]], [[1, 2, 3, 4, 5, 6]], 2),
    ([[1, 1, 2]], [[1, 2]], 3),
    ([[1, 2]], [[]], 2),
    ([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]], [[1], [1, 2, 3]], 5),
]

EXPLANATION = r"""
## What it rewards
AP@k is high when relevant items appear early. Two lists with the same hits at different ranks get different scores, unlike precision@k. Averaging over users gives MAP@k.

## The normaliser
Some definitions divide by the number of hits, which makes a list with a single hit at rank 1 score 1.0 regardless of how many relevant items were missed. Dividing by $\min(|rel|, k)$ penalises missing items that could have fit.

## MAP versus NDCG
MAP assumes binary relevance; NDCG handles graded relevance and uses a log discount instead of precision. Both are rank-aware, and teams usually report one of them alongside recall@k.

## Interview follow-ups
- Why skip users with no relevant items instead of scoring them 0?
"""
