import numpy as np

META = dict(title='Feed re-rank from multi-task predictions with a creator cap', tags=['Recommender Systems'], diff='Medium', runtime='numpy',
            related=['mmr-diversity-rerank'])

STATEMENT = r"""
A feed ranker predicts several outcomes per post. Write `rerank(preds, weights, creator, integrity, k, max_per_creator, demote)`.

- `preds` is a dict from task name to an array of probabilities (for example `"like"`, `"comment"`, `"share"`, `"hide"`). `weights` is a dict from task name to a weight; negative weights penalise (for example hiding). Value $v_i = \sum_t w_t \, p_{t,i}$ over the tasks in `weights`.
- `integrity[i]` is the probability that post $i$ is borderline content. Multiply value by `(1 - demote * integrity[i])`.
- Fill `k` slots greedily by adjusted value descending (ties to the lower index), skipping a post if its creator already has `max_per_creator` posts in the feed. Never include a post with adjusted value `<= 0`.

Return `[[index, round(adjusted_value, 6)], ...]` in feed order.
"""

STARTER = '''import numpy as np

def rerank(preds, weights, creator, integrity, k, max_per_creator, demote):
    """Return the feed as [[index, value], ...]."""
    v = sum(w * np.asarray(preds[t]) for t, w in weights.items())
    return [[int(i), float(v[i])] for i in np.argsort(-v)[:k]]
'''

SOLUTION = '''import numpy as np

def rerank(preds, weights, creator, integrity, k, max_per_creator, demote):
    n = len(creator)
    v = np.zeros(n)
    for t, w in weights.items():
        v += w * np.asarray(preds[t], float)
    v *= 1 - demote * np.asarray(integrity, float)
    used, feed = {}, []
    for i in sorted(range(n), key=lambda i: (-v[i], i)):
        if len(feed) >= k or v[i] <= 0:
            break
        c = creator[i]
        if used.get(c, 0) < max_per_creator:
            used[c] = used.get(c, 0) + 1
            feed.append([i, round(float(v[i]), 6)])
    return feed
'''

HINTS = [
    'Compute the weighted sum only over tasks listed in weights; preds may hold extra heads.',
    'Apply the integrity demotion multiplicatively before sorting.',
    'Because items are sorted, the first non-positive value means every later one is non-positive too.',
]

def _case(seed, n, nc):
    r = np.random.default_rng(seed)
    preds = {"like": r.beta(2, 8, n), "comment": r.beta(1, 20, n), "share": r.beta(1, 30, n), "hide": r.beta(1, 40, n), "dwell": r.random(n)}
    return (preds, {"like": 1.0, "comment": 4.0, "share": 8.0, "hide": -20.0}, list(r.integers(0, nc, n)), r.beta(1, 10, n))

TESTS = [
    ({"like": [0.5, 0.4, 0.3], "hide": [0.0, 0.0, 0.1]}, {"like": 1.0, "hide": -10.0}, ["a", "a", "b"], [0.0, 0.0, 0.0], 3, 1, 0.5),
    _case(1, 40, 5) + (10, 2, 0.8),
    _case(2, 40, 3) + (10, 1, 0.8),
    _case(3, 100, 20) + (25, 3, 0.0),
    ({"like": [0.2, 0.9]}, {"like": 1.0}, ["a", "b"], [0.0, 1.0], 2, 2, 1.0),
]

EXPLANATION = r"""
## Why many heads
"Engagement" is not one thing. A model predicting likes, comments, shares and hides separately lets the product team choose the trade-off in the weights, and change it without retraining. Negative-outcome heads such as hides are what stop the feed from optimising for clickbait.

## Integrity as a soft demotion
Clear violations are removed upstream. Borderline content is demoted in proportion to its predicted probability, so the penalty grows smoothly instead of a hard cutoff that is easy to game.

## Diversity constraints
A creator cap is the simplest guard against one prolific account filling the feed. Greedy re-ranking with constraints is standard because it is fast and the effect of each rule is easy to explain.

## Interview follow-ups
- How would you set the weights, and how would you check they still reflect what users value a year later?
"""
