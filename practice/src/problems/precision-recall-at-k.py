META = dict(title='Precision@k and recall@k', tags=['Metrics & Evaluation', 'Recommender Systems'], diff='Easy', runtime='python',
            lessons=['sdsd:09.09'], related=['cosine-similarity-matrix'])

STATEMENT = r"""
A retrieval system returns a ranked list. Two standard ways to score the top $k$:

$$\mathrm{precision@}k = \frac{|\text{relevant} \cap \text{top-}k|}{k}, \qquad \mathrm{recall@}k = \frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|}$$

Write `precision_recall_at_k(ranked, relevant, k)` that returns the tuple `(precision, recall)` as floats.

## Rules
- `ranked`: list of item ids, best first (no duplicates). `relevant`: list of the ids that are actually relevant.
- If `ranked` has fewer than `k` items, the missing slots count as misses: still divide precision by `k`.
- If `relevant` is empty, recall is `0.0`.
- `k >= 1`.
"""

STARTER = '''def precision_recall_at_k(ranked, relevant, k):
    """Return (precision@k, recall@k) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''def precision_recall_at_k(ranked, relevant, k):
    rel = set(relevant)
    hits = sum(1 for item in ranked[:k] if item in rel)
    return hits / k, (hits / len(rel) if rel else 0.0)
'''

HINTS = [
    'Turn `relevant` into a set so each membership check is O(1).',
    'Count hits only within `ranked[:k]`. Slicing past the end is safe in Python.',
    'Precision always divides by `k`, even when fewer than `k` items were returned. Recall divides by the number of relevant items.',
]

TESTS = [
    ([1, 2, 3, 4, 5], [2, 5, 9], 3),
    ([1, 2, 3, 4, 5], [2, 5, 9], 5),
    (['a', 'b'], ['a'], 5),
    ([7, 8, 9], [], 2),
    ([3, 1, 2], [1, 2, 3], 1),
    ([10, 20, 30, 40], [40, 30, 20, 10], 4),
    ([1, 2, 3], [4, 5], 3),
]

EXPLANATION = r"""
## Reading the two numbers
- **Precision@k** asks how much of what we showed was useful. It's the user-facing quality of a results page.
- **Recall@k** asks how much of what's useful we found. It's the main metric for the **candidate generation** stage of a two-stage recommender, where a later ranker can only reorder what retrieval found.

They trade off as $k$ grows: recall can only rise, and precision usually falls.

## Neither looks at order inside the top k
Swapping rank 1 and rank k changes neither metric. When order matters, use MRR (reciprocal rank of the first hit), MAP, or NDCG, which discounts hits by $1/\log_2(\text{rank}+1)$.

## Interview follow-ups
- Implement NDCG@k with graded relevance.
- Your offline recall@100 improved but online CTR didn't move. List three reasons. (Position bias, the ranker discards the new candidates, label leakage in the offline set.)
"""
