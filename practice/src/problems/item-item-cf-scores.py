import numpy as np

META = dict(title='Item-item collaborative filtering with shrunk cosine', tags=['Recommender Systems', 'Linear Algebra'],
            diff='Medium', runtime='numpy', lessons=['sdsd:12.08', 'sdsd:16.05'],
            related=['cosine-similarity-matrix', 'user-knn-rating-prediction', 'top-n-per-group'])

STATEMENT = r"""
Item-based collaborative filtering says: recommend items that are often consumed **together with** what the user already has. With a binary interaction matrix $R \in \{0,1\}^{U \times I}$:

- co-occurrence counts $C = R^\top R$, so $C_{ij}$ is the number of users who have both $i$ and $j$, and $n_i = C_{ii}$ is the number of users who have $i$;
- a **shrunk cosine** similarity that distrusts pairs supported by few users,

$$S_{ij} = \frac{C_{ij}}{\sqrt{n_i n_j} + \beta} \quad (i \ne j), \qquad S_{ii} = 0;$$

- user scores $\text{score}_{u,j} = \sum_i R_{ui} S_{ij}$, i.e. `R @ S`.

Write `item_cf_scores(R, shrink)` that returns the float array of scores, shape `(U, I)`.

## Inputs
- `R`: `(U, I)` array of 0s and 1s (int or float dtype).
- `shrink`: $\beta \ge 0$.

## Rules
- Items the user already has ($R_{uj} = 1$) get score `-inf`, so they can never be recommended.
- If the denominator is 0 (only possible when $\beta = 0$ and an item has no users), that $S_{ij}$ is 0.
- Vectorise: no Python loops over users or items.
"""

STARTER = '''import numpy as np

def item_cf_scores(R, shrink):
    """R: (U, I) binary. Return (U, I) float scores with seen items set to -inf."""
    R = np.asarray(R, dtype=float)
    S = R.T @ R
    return R @ S
'''

SOLUTION = '''import numpy as np

def item_cf_scores(R, shrink):
    R = np.asarray(R, dtype=float)
    C = R.T @ R                                   # (I, I) co-occurrence counts
    n = np.diag(C)                                # users per item
    denom = np.sqrt(np.outer(n, n)) + shrink
    S = np.divide(C, denom, out=np.zeros_like(C), where=denom > 0)
    np.fill_diagonal(S, 0.0)                      # an item must not vote for itself
    scores = R @ S
    scores[R > 0] = -np.inf                       # never recommend what the user already has
    return scores
'''

HINTS = [
    '`R.T @ R` gives every co-occurrence count at once, and its diagonal is the per-item user count $n_i$.',
    'Build the denominator with `np.sqrt(np.outer(n, n)) + shrink` and divide with `np.divide(..., where=denom > 0)` to avoid 0/0.',
    'Zero the diagonal of $S$ before multiplying, then set `scores[R > 0] = -np.inf` afterwards.',
]

_r = np.random.default_rng(3)
_R = (_r.random((12, 9)) < 0.3).astype(int)

TESTS = [
    (np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]), 0.0),
    (np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]), 1.0),
    (np.array([[0, 0], [0, 0]]), 0.0),
    (np.array([[1, 1, 1]]), 2.0),
    (np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]]), 0.0),
    (np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]]), 0.5),
    (_R, 0.0),
    (_R, 3.0),
]

EXPLANATION = r"""
## Why it works
$C = R^\top R$ counts, for each item pair, the users who have both. Dividing by $\sqrt{n_i n_j}$ is exactly the cosine between the two binary item columns, which stops very popular items from looking similar to everything. The shrinkage term $\beta$ pulls similarities supported by few users toward 0: with $n_i = n_j = 1$ and one shared user the plain cosine is a "perfect" 1.0, while $\beta = 3$ drops it to 0.25. A user's score for $j$ is the sum of similarities from each item they own, so items related to **many** of the user's items rise to the top.

## Complexity
$O(U I^2)$ to build $C$ and the scores, $O(I^2)$ memory. Real systems keep only the top-$m$ neighbours per item in a sparse matrix and precompute it offline, because item-item similarities change slowly.

## Common bugs
- Leaving the diagonal in. Here it only inflates owned items, which are masked anyway, but variants that keep seen items (re-watch, re-buy) or normalise by row sums of $S$ get wrong scores.
- Masking seen items with 0 instead of `-inf`: a seen item can then outrank unseen items with negative or zero scores in variants with negative similarities.
- Dividing without a guard when $\beta = 0$ and an item has no users (test 3).

## Interview follow-ups
- Why did item-based CF beat user-based CF at large retailers? (Items are fewer and more stable than users, so the similarity table can be precomputed and cached; a new user's recommendations update instantly from their history.)
- How would you compute $R^\top R$ when $R$ is 100M users × 10M items and very sparse? (Sparse matrix products sharded by user, emitting item pairs per user with a cap on very active users.)
"""
