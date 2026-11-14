import numpy as np

META = dict(title='Recall@K, tail recall and catalogue coverage', tags=['Recommender Systems', 'Evaluation Metrics'], diff='Medium', runtime='numpy',
            related=['precision-recall-at-k', 'catalog-coverage-and-novelty'])

STATEMENT = r"""
Write `retrieval_report(recs, held_out, popularity, k, head_frac)`.

- `recs[u]` is user `u`'s ranked list of item ids; only the first `k` count. `held_out[u]` lists the items user `u` interacted with in the test period.
- `popularity[i]` is item `i`'s training interaction count. **Head** items are the top `ceil(head_frac * n_items)` by popularity (ties broken by lower id); all others are **tail**.
- `recall`: mean over users with at least one held-out item of $|\text{top-}k \cap H_u| / |H_u|$.
- `tail_recall`: the same, using only held-out **tail** items, over users with at least one of them (0.0 if none).
- `coverage`: distinct items appearing in any user's top-`k`, divided by the number of items.

Return a dict with these three floats.
"""

STARTER = '''import numpy as np

def retrieval_report(recs, held_out, popularity, k, head_frac):
    """Recall@k overall and on long-tail items, plus catalogue coverage."""
    hits = [len(set(r[:k]) & set(h)) / k for r, h in zip(recs, held_out) if h]
    return {"recall": float(np.mean(hits)), "tail_recall": float(np.mean(hits)), "coverage": 0.0}
'''

SOLUTION = '''import numpy as np

def retrieval_report(recs, held_out, popularity, k, head_frac):
    pop = np.asarray(popularity, float)
    n = len(pop)
    n_head = int(np.ceil(head_frac * n))
    order = np.lexsort((np.arange(n), -pop))
    head = set(order[:n_head].tolist())
    rec, tail = [], []
    shown = set()
    for r, h in zip(recs, held_out):
        top = set(r[:k]); shown |= top
        if h:
            rec.append(len(top & set(h)) / len(set(h)))
        ht = set(h) - head
        if ht:
            tail.append(len(top & ht) / len(ht))
    return {"recall": float(np.mean(rec)) if rec else 0.0, "tail_recall": float(np.mean(tail)) if tail else 0.0,
            "coverage": len(shown) / n}
'''

HINTS = [
    'Recall divides by the number of held-out items, not by k.',
    'np.lexsort((ids, -popularity)) sorts by popularity descending, then id ascending.',
    'A model can have high recall and near-zero tail recall by recommending only popular items.',
]

_pop = [100, 80, 60, 5, 4, 3, 2, 1, 0, 0]

TESTS = [
    ([[0, 1, 2]], [[0, 5]], _pop, 3, 0.3),
    ([[0, 1, 2], [0, 1, 2]], [[1], [7]], _pop, 3, 0.3),
    ([[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 3], [4], [9]], _pop, 2, 0.2),
    ([[5, 0, 1], [6, 2, 0]], [[], [6]], _pop, 1, 0.5),
    ([[1, 2, 3, 4]], [[2, 3, 4]], [3, 3, 3, 1, 1], 2, 0.4),
]

EXPLANATION = r"""
## Why one number is not enough
Recommending the most popular items is a strong baseline for recall, because popular items are, by definition, what many users interact with next. A retrieval model that only learned popularity would look good on recall and be useless for discovery. Tail recall and coverage expose that.

## The trade-off
Raising tail recall usually costs some overall recall. The logQ correction in the next step is one deliberate move along that trade-off: it stops in-batch training from over-penalising popular items as negatives, which changes which items the model ranks highly.

## Interview follow-ups
- How would you choose the head/tail cut for a catalogue with a very steep popularity curve?
"""
