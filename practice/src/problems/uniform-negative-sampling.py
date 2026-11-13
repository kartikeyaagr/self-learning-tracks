import numpy as np

META = dict(title='Uniform negative sampling with rejection', tags=['Recommender Systems', 'Probability'], diff='Easy',
            runtime='numpy', lessons=['sdsd:12.08'], related=['bpr-sgd-step', 'in-batch-softmax-logq'])

STATEMENT = r"""
Implicit-feedback data only records what users **did** interact with. Pairwise losses such as BPR need, for every positive $(u, i)$, some items $j$ the user did **not** interact with. The simplest recipe draws $j$ uniformly from the catalogue and rejects draws that are positives of the same user.

Write `sample_negatives(users, items, n_items, n_neg, seed)` that returns an `int64` array of shape `(P, n_neg)`.

## Inputs
- `users`, `items`: int arrays of shape `(P,)` with `P >= 1`. Row $p$ is the positive pair `(users[p], items[p])`. Pairs may repeat.
- `n_items`: catalogue size; item ids are `0 .. n_items - 1`.
- `n_neg >= 1`: negatives per positive.
- `seed`: int.

## Exact sampling procedure (the tests depend on it)
1. A user's **positive set** is every item paired with that user anywhere in the input.
2. Create one generator: `rng = np.random.default_rng(seed)`.
3. Visit rows $p = 0, 1, \dots, P-1$ in order, and within a row fill columns $0, 1, \dots, n_{neg}-1$ in order.
4. To fill one cell, repeatedly draw `int(rng.integers(n_items))` (one call per candidate) and keep the first draw that is **not** in `users[p]`'s positive set.

Negatives may repeat within a row and across rows. Every user is guaranteed to have at least one non-positive item.
"""

STARTER = '''import numpy as np

def sample_negatives(users, items, n_items, n_neg, seed):
    """Return (P, n_neg) int64 negatives, drawn as described in the statement."""
    rng = np.random.default_rng(seed)
    return rng.integers(n_items, size=(len(users), n_neg))
'''

SOLUTION = '''import numpy as np

def sample_negatives(users, items, n_items, n_neg, seed):
    positives = {}
    for u, i in zip(np.asarray(users).tolist(), np.asarray(items).tolist()):
        positives.setdefault(u, set()).add(i)
    rng = np.random.default_rng(seed)
    out = np.empty((len(users), n_neg), dtype=np.int64)
    for p, u in enumerate(np.asarray(users).tolist()):
        seen = positives[u]
        for c in range(n_neg):
            j = int(rng.integers(n_items))
            while j in seen:                     # rejection: redraw until it is a true negative
                j = int(rng.integers(n_items))
            out[p, c] = j
    return out
'''

HINTS = [
    'Build a dict `user -> set of positive items` in one pass before sampling anything. Checking only `items[p]` misses the user\'s other positives.',
    'Rejection sampling: draw, check membership in the set, and redraw in a `while` loop until the draw is not a positive.',
    'The tests replay one generator\'s stream, so draw exactly one scalar per candidate with `rng.integers(n_items)` and fill rows and columns in order.',
]

_r = np.random.default_rng(8)
_u = _r.integers(0, 20, size=40)
_i = _r.integers(0, 50, size=40)

TESTS = [
    (np.array([0, 0, 1]), np.array([1, 2, 0]), 5, 2, 0),
    (np.array([0, 1]), np.array([0, 1]), 3, 3, 42),
    (np.array([0]), np.array([0]), 2, 4, 7),
    (np.array([2, 2, 2]), np.array([0, 1, 2]), 4, 2, 1),
    (np.array([5, 5, 6]), np.array([0, 0, 9]), 10, 3, 0),
    (np.array([0, 1, 0, 1]), np.array([3, 3, 3, 4]), 5, 1, 99),
    (_u, _i, 50, 4, 2024),
]

EXPLANATION = r"""
## Why it works
Rejection sampling from the uniform distribution, conditioned on "not a positive", gives exactly the uniform distribution over the user's non-positive items. The expected number of draws per cell is $N / (N - |P_u|)$, which is close to 1 in real catalogues where users touch a tiny fraction of items. Test 3 is the extreme: one valid negative out of two items, so every cell must be item 1.

## Complexity
$O(P)$ to build the sets, then $O(P \cdot n_{neg} \cdot N/(N-|P_u|))$ expected draws.

## Common bugs
- Rejecting only the current positive `items[p]`, so other positives of the same user leak in as "negatives" (test 1 catches it).
- Vectorising with `rng.integers(n_items, size=...)` and then patching collisions: valid in practice, but a different random stream.
- Rejecting by mistake against **all** users' positives, which shrinks the pool for everyone.

## Interview follow-ups
- Uniform negatives are mostly easy. What are popularity-weighted and hard negatives, and what bias does each introduce? (Popularity sampling, e.g. $\propto c_i^{0.75}$, makes the model work on items users plausibly saw; hard negatives from the current model speed learning but risk sampling unlabelled positives.)
- An unobserved item may be a positive the user never saw. How do implicit-feedback losses account for that? (Lower confidence on unobserved entries, as in weighted ALS.)
"""
