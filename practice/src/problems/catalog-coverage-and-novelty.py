import math

META = dict(title='Catalogue coverage, popularity and novelty of recommendations',
            tags=['Recommender Systems', 'Metrics & Evaluation'], diff='Easy', runtime='python',
            lessons=['sdsd:16.05'], related=['hit-rate-and-mrr', 'mmr-diversity-rerank'])

STATEMENT = r"""
Accuracy metrics alone reward a recommender that shows everyone the same bestsellers. Three **beyond-accuracy** metrics expose that popularity bias:

- **Coverage**: the fraction of the catalogue that appears in at least one user's list.
- **Average popularity**: the mean training interaction count of the recommended items, averaged over every recommendation slot.
- **Novelty**: the mean self-information of the recommended items, $-\log_2 p(i)$, averaged over every slot, where

$$p(i) = \frac{c_i + 1}{C + N}$$

with $c_i$ the training count of item $i$ (0 if absent), $C = \sum_i c_i$ the total of all counts, and $N$ the catalogue size. The $+1$ smoothing keeps never-seen items finite.

Write `coverage_novelty(recs, counts, n_items)` that returns a dict with float values under the keys `'coverage'`, `'avg_popularity'` and `'novelty'`.

## Inputs
- `recs`: list of lists. `recs[u]` is user `u`'s recommended item ids, integers in `[0, n_items)`. Lists may have different lengths, and may be empty.
- `counts`: dict `item id -> training interaction count` (non-negative int). Items missing from it have count 0.
- `n_items`: catalogue size $N \ge 1$.

## Rules
- A "slot" is one entry of one list. An item recommended to 3 users counts 3 times in the averages but once for coverage.
- If there are no slots at all, return `0.0` for all three metrics.
"""

STARTER = '''import math

def coverage_novelty(recs, counts, n_items):
    """Return {'coverage': ..., 'avg_popularity': ..., 'novelty': ...}."""
    raise NotImplementedError
'''

SOLUTION = '''import math

def coverage_novelty(recs, counts, n_items):
    slots = [item for lst in recs for item in lst]
    if not slots:
        return {'coverage': 0.0, 'avg_popularity': 0.0, 'novelty': 0.0}
    total = sum(counts.values())
    p = lambda i: (counts.get(i, 0) + 1) / (total + n_items)     # add-one smoothed popularity
    return {
        'coverage': len(set(slots)) / n_items,
        'avg_popularity': sum(counts.get(i, 0) for i in slots) / len(slots),
        'novelty': sum(-math.log2(p(i)) for i in slots) / len(slots),
    }
'''

HINTS = [
    'Flatten `recs` into one list of slots first. Coverage uses the set of slots, the other two use every slot.',
    'Use `counts.get(i, 0)` everywhere: the training data may never have seen a recommended item.',
    'The smoothed denominator is the total count **plus** the catalogue size, so the probabilities of all $N$ items sum to 1.',
]

import random as _random
_g = _random.Random(23)
_counts = {i: int(1000 / (i + 1)) for i in range(60)}          # long-tail popularity
_recs = [_g.sample(range(80), 10) for _ in range(12)]
_pop_recs = [list(range(10)) for _ in range(12)]

TESTS = [
    ([[0, 1], [0, 2]], {0: 6, 1: 2, 2: 0}, 4),
    ([[3]], {0: 5}, 4),
    ([], {0: 1}, 10),
    ([[], []], {}, 3),
    ([[0, 1, 2, 3]], {}, 4),
    (_recs, _counts, 80),
    (_pop_recs, _counts, 80),
]

EXPLANATION = r"""
## Why it works
All three metrics look only at what was shown, not at whether it was clicked. Coverage counts distinct items, so it measures how much of the catalogue the system can surface at all. Average popularity and novelty average over slots, so they reflect what a typical impression looks like. Novelty is the information content of an impression: a bestseller with $p \approx 0.1$ carries about 3.3 bits, a tail item with $p \approx 10^{-5}$ about 16.6 bits. Tests 6 and 7 use the same counts: random lists score high novelty and coverage, "top-10 for everyone" scores the minimum on both.

## Complexity
$O(S + |\text{counts}|)$ for $S$ slots.

## Common bugs
- Computing coverage over slots instead of distinct items (it can exceed 1).
- Averaging per user first, then over users. That weights users with short lists more. This problem averages over slots.
- Forgetting smoothing, so an item with zero training count gives $-\log 0 = \infty$.

## Interview follow-ups
- Coverage went up after adding an exploration slot, but click-through fell. How would you decide whether to ship? (Measure long-term metrics: retention, catalogue health, creator-side supply; run a holdout long enough for feedback loops to matter.)
- How would you measure **intra-list** diversity, and how does it differ from coverage? (Mean pairwise dissimilarity within one list vs distinct items across all lists.)
"""
