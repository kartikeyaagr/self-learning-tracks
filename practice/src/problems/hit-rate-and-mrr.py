META = dict(title='Hit rate@k and MRR@k for leave-one-out evaluation', tags=['Recommender Systems', 'Metrics & Evaluation'],
            diff='Easy', runtime='python', lessons=['sdsd:09.09'], related=['precision-recall-at-k', 'ndcg-at-k'])

STATEMENT = r"""
A common offline protocol for recommenders hides **one** interaction per user (usually the most recent), asks the model for a top-$k$ list, and checks where the hidden item landed. Two numbers summarise the result over all evaluated users $U$:

$$\mathrm{HR@}k = \frac{1}{|U|}\sum_{u \in U} \mathbb{1}[\mathrm{rank}_u \le k], \qquad \mathrm{MRR@}k = \frac{1}{|U|}\sum_{u \in U} \frac{\mathbb{1}[\mathrm{rank}_u \le k]}{\mathrm{rank}_u}$$

where $\mathrm{rank}_u$ is the 1-based position of user $u$'s held-out item in their recommendation list.

Write `hit_rate_mrr(recs, held_out, k)` that returns the tuple `(hit_rate, mrr)` as floats.

## Inputs
- `recs`: dict `user -> list of item ids`, best first, no duplicates. Lists may be longer or shorter than `k`.
- `held_out`: dict `user -> item id`. The evaluated users are exactly the keys of `held_out`.
- `k >= 1`.

## Rules
- Only the first `k` entries of a list count. A held-out item at position `k + 1` or later is a miss.
- A user in `held_out` with no entry in `recs` is a miss (contributes 0 to both sums but still counts in $|U|$).
- Users in `recs` but not in `held_out` are ignored.
- If `held_out` is empty, return `(0.0, 0.0)`.
"""

STARTER = '''def hit_rate_mrr(recs, held_out, k):
    """Return (HR@k, MRR@k) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''def hit_rate_mrr(recs, held_out, k):
    if not held_out:
        return 0.0, 0.0
    hits = rr = 0.0
    for user, target in held_out.items():
        top = recs.get(user, [])[:k]             # a missing user simply has an empty list
        if target in top:
            hits += 1
            rr += 1.0 / (top.index(target) + 1)  # positions are 1-based
    n = len(held_out)
    return hits / n, rr / n
'''

HINTS = [
    'Loop over `held_out`, not over `recs`: the denominator is the number of evaluated users.',
    'Use `recs.get(user, [])[:k]` so a missing user and a short list need no special case.',
    'The reciprocal rank is `1 / (index + 1)` because list indices start at 0 and ranks start at 1.',
]

import random as _random
_g = _random.Random(11)
_items = list(range(30))
_recs, _held = {}, {}
for _u in range(25):
    _recs[f'u{_u}'] = _g.sample(_items, 8)
    _held[f'u{_u}'] = _g.choice(_items)

TESTS = [
    ({'a': [5, 3, 9], 'b': [1, 2, 3]}, {'a': 3, 'b': 7}, 3),
    ({'a': [4, 8, 1, 6], 'b': [6, 4], 'c': [2, 9, 6]}, {'a': 1, 'b': 6, 'c': 6}, 2),
    ({}, {}, 5),
    ({'a': [1, 2, 3]}, {'a': 1, 'z': 2}, 10),
    ({'a': [1, 2, 3, 4, 5]}, {'a': 5}, 4),
    ({'x': ['p', 'q'], 'y': ['q', 'p'], 'extra': ['p']}, {'x': 'q', 'y': 'q'}, 1),
    (_recs, _held, 5),
    (_recs, _held, 8),
]

EXPLANATION = r"""
## Why it works
With a single relevant item per user, recall@k collapses to a 0/1 hit, so hit rate is just mean recall. MRR adds position: a hit at rank 1 is worth 1, at rank 2 worth 1/2, and so on. Iterating over the held-out users makes the denominator right even when some users have no list at all, which happens in practice when a model can't score cold users.

## Complexity
$O(|U| \cdot k)$ time with list membership; $O(1)$ extra space.

## Common bugs
- Dividing by `len(recs)` instead of the number of evaluated users. Silently drops users the model failed on, inflating both metrics.
- Forgetting the cut at `k` for MRR, so deep hits leak in.
- Using 0-based ranks, which divides by zero on a rank-1 hit.

## Interview follow-ups
- Many papers rank the held-out item against 100 sampled negatives instead of the full catalogue. Why can that change which of two models looks better? (Sampled metrics are not rank-preserving estimates of the full-catalogue metric, especially at small $k$.)
- How would you compute a confidence interval for HR@10 when comparing two models on the same users? (Paired bootstrap over users.)
"""
