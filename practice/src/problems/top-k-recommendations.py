import numpy as np

META = dict(title='Top-k recommendations excluding seen items', tags=['Recommender Systems'], diff='Easy', runtime='numpy')

STATEMENT = r"""
Write `recommend(U, V, seen, k)`.

- `U` is `(n_users, d)` and `V` is `(n_items, d)`; scores are `U @ V.T`.
- `seen` is a list of lists: `seen[u]` holds item ids user `u` already interacted with. Never recommend those.
- For each user return up to `k` item ids, highest score first. Break ties by the smaller item id.

Return a list of lists of Python ints.
"""

STARTER = '''import numpy as np

def recommend(U, V, seen, k):
    """Top-k unseen items per user."""
    S = np.asarray(U) @ np.asarray(V).T
    return [list(np.argsort(-s)[:k]) for s in S]
'''

SOLUTION = '''import numpy as np

def recommend(U, V, seen, k):
    S = np.asarray(U, float) @ np.asarray(V, float).T
    out = []
    for u, s in enumerate(S):
        s = s.copy()
        s[list(seen[u])] = -np.inf
        order = np.lexsort((np.arange(len(s)), -s))
        out.append([int(i) for i in order if np.isfinite(s[i])][:k])
    return out
'''

HINTS = [
    'Mask seen items by setting their scores to -inf before sorting.',
    'np.lexsort sorts by the last key first, so pass (item ids, -score) to get score descending with id as the tie-break.',
    'Filter out masked items after sorting, so a user who has seen nearly everything gets fewer than k results.',
]

_r = np.random.default_rng(4)
TESTS = [
    ([[1.0, 0.0]], [[3.0, 0], [1.0, 0], [2.0, 0]], [[0]], 2),
    (_r.normal(size=(3, 4)), _r.normal(size=(10, 4)), [[], [1, 2, 3], list(range(9))], 3),
    ([[1.0]], [[1.0], [1.0], [1.0]], [[]], 2),
    ([[1.0]], [[1.0], [2.0]], [[0, 1]], 2),
    (_r.normal(size=(5, 8)), _r.normal(size=(50, 8)), [list(_r.choice(50, 10, replace=False)) for _ in range(5)], 5),
]

EXPLANATION = r"""
## Why filtering matters
A model trained on interactions scores items the user already has very highly, because that is exactly what it learned. Without the filter the top slots are wasted on repeats. Some domains (groceries, music) do want repeats, in which case the filter becomes a business rule rather than a default.

## At scale
With millions of items you do not score everything. An approximate nearest-neighbour index returns a few hundred candidates; ask for $k$ plus the number of seen items, or over-fetch and filter, so the filter does not leave you short.

## Deterministic ties
Stable tie-breaking makes offline evaluation reproducible and A/B comparisons fair: two runs of the same model should give the same list.

## Interview follow-ups
- How would you enforce diversity, for example at most two items per category?
"""
