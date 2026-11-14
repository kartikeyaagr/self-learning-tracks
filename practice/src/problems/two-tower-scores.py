import numpy as np

META = dict(title='Two-tower scores: every user against every item', tags=['Recommender Systems', 'Embeddings'], diff='Easy', runtime='numpy')

STATEMENT = r"""
A two-tower recommender encodes users and items separately and scores a pair by a dot product.

Write `two_tower_scores(Xu, Xi, Wu, bu, Wi, bi, normalize)`:

- User tower: $u = \mathrm{ReLU}(X_u W_u + b_u)$. Item tower: $v = \mathrm{ReLU}(X_i W_i + b_i)$.
- If `normalize`, scale every embedding to unit length (an all-zero row stays zero).
- Return the score matrix $S = u v^\top$ of shape `(n_users, n_items)`.
"""

STARTER = '''import numpy as np

def two_tower_scores(Xu, Xi, Wu, bu, Wi, bi, normalize):
    """Score all user-item pairs with separate user and item towers."""
    u = np.asarray(Xu) @ np.asarray(Wu)
    v = np.asarray(Xi) @ np.asarray(Wu)
    return u @ v.T
'''

SOLUTION = '''import numpy as np

def _unit(E):
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return np.where(n > 0, E / np.where(n > 0, n, 1.0), 0.0)

def two_tower_scores(Xu, Xi, Wu, bu, Wi, bi, normalize):
    u = np.maximum(np.asarray(Xu, float) @ np.asarray(Wu, float) + np.asarray(bu, float), 0.0)
    v = np.maximum(np.asarray(Xi, float) @ np.asarray(Wi, float) + np.asarray(bi, float), 0.0)
    if normalize:
        u, v = _unit(u), _unit(v)
    return u @ v.T
'''

HINTS = [
    'The two towers never see each other’s inputs; that is what lets you precompute every item embedding offline.',
    'With normalisation the score is a cosine similarity, which stops popular items winning just by having large embeddings.',
]

def _case(seed, nu, ni, fu, fi, d):
    g = np.random.default_rng(seed)
    return (g.normal(size=(nu, fu)), g.normal(size=(ni, fi)), g.normal(size=(fu, d)), 0.1 * g.normal(size=d), g.normal(size=(fi, d)), 0.1 * g.normal(size=d))

TESTS = [
    ([[1.0, 0.0]], [[1.0], [2.0]], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [[1.0, 1.0]], [0.0, 0.0], False),
    ([[1.0, 0.0]], [[1.0], [2.0]], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [[1.0, 1.0]], [0.0, 0.0], True),
    _case(1, 3, 5, 4, 6, 3) + (False,),
    _case(2, 4, 6, 5, 3, 8) + (True,),
    ([[-1.0]], [[1.0]], [[1.0]], [0.0], [[1.0]], [0.0], True),
]

EXPLANATION = r"""
## Why two towers
Scoring every user against every item with a joint model costs one network call per pair, which is impossible over millions of items. With separate towers, item embeddings are computed once and stored in a nearest-neighbour index; at request time only the user tower runs, and retrieval is a fast approximate search for the largest dot products.

## The price
The towers can only interact through a dot product, so they cannot model feature crosses such as "this user likes this brand only at this price". That is why two-tower models are used for retrieval (picking a few hundred candidates) and a heavier ranking model reorders those candidates.

## Interview follow-ups
- Why is a dot product preferred over cosine similarity in some production systems, despite the popularity effect?
"""
