import math

META = dict(title='NDCG@k with graded relevance', tags=['Recommender Systems', 'Metrics & Evaluation'], diff='Easy',
            runtime='python', lessons=['sdsd:09.09'], related=['precision-recall-at-k', 'hit-rate-and-mrr'])

STATEMENT = r"""
Normalised discounted cumulative gain rewards putting **highly** relevant items **near the top**. With graded gains $g_i \ge 0$:

$$\mathrm{DCG@}k = \sum_{i=1}^{\min(k, n)} \frac{2^{g_{(i)}} - 1}{\log_2(i + 1)}, \qquad \mathrm{NDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}$$

where $g_{(i)}$ is the gain of the item at rank $i$ of the list, and $\mathrm{IDCG@}k$ is the DCG@k of the best possible list: all items in `gains` sorted by gain descending.

Write `ndcg_at_k(ranked, gains, k)` that returns NDCG@k as a float.

## Inputs
- `ranked`: list of item ids, best first, no duplicates. May be shorter than `k`.
- `gains`: dict `item id -> gain` (non-negative int or float). Items not in `gains` have gain 0.
- `k >= 1`.

## Rules
- Use the exponential gain $2^g - 1$ and the $\log_2(i+1)$ discount with 1-based rank $i$.
- The ideal list is built from **all** of `gains`, not only from items that appear in `ranked`.
- If $\mathrm{IDCG@}k = 0$ (no positive gains), return `0.0`.
"""

STARTER = '''import math

def ndcg_at_k(ranked, gains, k):
    """Return NDCG@k as a float."""
    raise NotImplementedError
'''

SOLUTION = '''import math

def ndcg_at_k(ranked, gains, k):
    def dcg(gs):
        # enumerate from 0, so the discount uses log2(i + 2) for 1-based rank i + 1
        return sum((2.0 ** g - 1.0) / math.log2(i + 2) for i, g in enumerate(gs[:k]))

    ideal = dcg(sorted(gains.values(), reverse=True))
    if ideal == 0:
        return 0.0
    return dcg([gains.get(item, 0) for item in ranked]) / ideal
'''

HINTS = [
    'Write one helper `dcg(list_of_gains)` and call it twice: on the gains of `ranked`, and on all gains sorted descending.',
    'Rank 1 has discount $\\log_2 2 = 1$. With `enumerate` starting at 0 that is `log2(i + 2)`.',
    'Build the ideal list from `gains.values()`, not from `ranked`. A list that misses a relevant item must score below 1.',
]

import random as _random
_g = _random.Random(5)
_gains = {i: _g.choice([0, 0, 1, 2, 3]) for i in range(40)}
_ranked = _g.sample(range(50), 15)

TESTS = [
    ([3, 1, 2], {1: 3, 2: 2, 3: 1}, 3),
    (['a', 'b', 'c', 'd'], {'b': 1, 'x': 2}, 4),
    ([1, 2, 3], {}, 3),
    ([1, 2, 3], {1: 0, 2: 0}, 2),
    ([7, 8], {7: 1, 8: 1, 9: 1}, 5),
    ([5, 4, 3, 2, 1], {1: 2, 2: 2, 3: 1}, 1),
    ([2, 1], {1: 1.5, 2: 0.5}, 2),
    (_ranked, _gains, 10),
]

EXPLANATION = r"""
## Why it works
DCG sums each item's gain divided by a slowly growing discount, so a relevant item contributes most at rank 1 and less further down. Dividing by the DCG of the perfect ordering puts every user on a $[0, 1]$ scale, so users with many relevant items don't dominate the average. Using all of `gains` for the ideal list is what makes NDCG punish **missing** relevant items, not only misordered ones (test 2: the gain-2 item `x` never shows up).

## Complexity
$O(n + m \log m)$ for a list of length $n$ and $m$ judged items (the sort for the ideal list). With $k \ll m$, `heapq.nlargest(k, ...)` makes it $O(m \log k)$.

## Common bugs
- Off-by-one in the discount: $\log_2(i)$ with 1-based $i$ divides by zero at rank 1.
- Building IDCG by re-sorting `ranked` instead of the judged set.
- Mixing gain conventions: the linear gain $g$ and the exponential $2^g - 1$ agree only for binary relevance. State which one a paper uses before comparing numbers.

## Interview follow-ups
- Why is NDCG hard to optimise directly with gradient descent, and what do LambdaRank/LambdaMART do about it? (It is piecewise constant in scores; LambdaRank scales pairwise gradients by the $|\Delta \mathrm{NDCG}|$ of swapping each pair.)
- How do position bias and unclicked-but-unseen items distort NDCG computed from click logs?
"""
