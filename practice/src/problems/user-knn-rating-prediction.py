import numpy as np

META = dict(title='User-kNN rating prediction with mean-centred similarity', tags=['Recommender Systems', 'Linear Algebra'],
            diff='Medium', runtime='numpy', lessons=['sdsd:12.08'], related=['item-item-cf-scores', 'knn-classifier'])

STATEMENT = r"""
User-based collaborative filtering predicts user $u$'s rating of item $i$ from the ratings of **similar users** who rated $i$, after removing each user's personal bias (some people rate everything 4-5, others 2-3).

Given a rating matrix $R$ and a 0/1 mask $M$ of which entries are observed:

1. $\mu_u$ = mean of user $u$'s observed ratings. Centred ratings $Z_{ui} = (R_{ui} - \mu_u) M_{ui}$.
2. Similarity between users, over their **co-rated** items $\Omega_{uv} = \{i : M_{ui} = M_{vi} = 1\}$:
$$s_{uv} = \frac{\sum_{i \in \Omega_{uv}} Z_{ui} Z_{vi}}{\sqrt{\sum_{i \in \Omega_{uv}} Z_{ui}^2}\,\sqrt{\sum_{i \in \Omega_{uv}} Z_{vi}^2}}$$
with $s_{uv} = 0$ if either root is 0.
3. To predict $(u, i)$, the neighbourhood $N$ is the `k` users $v \ne u$ with $M_{vi} = 1$ and $s_{uv} > 0$ that have the largest $s_{uv}$ (ties: smaller user index first). If fewer than `k` qualify, use all of them.
4. Prediction:
$$\hat r_{ui} = \mu_u + \frac{\sum_{v \in N} s_{uv}(R_{vi} - \mu_v)}{\sum_{v \in N} s_{uv}}$$
If $N$ is empty, $\hat r_{ui} = \mu_u$.

Write `user_knn_predict(R, M, queries, k)` returning a float array of shape `(Q,)`, one prediction per query.

## Inputs
- `R`: `(U, I)` float; values where `M == 0` are meaningless and must be ignored.
- `M`: `(U, I)` of 0/1. Every user has at least one observed rating.
- `queries`: `(Q, 2)` int array of `(user, item)` pairs. Whether $u$ already rated $i$ does not matter; predict anyway (u is never its own neighbour).
- `k >= 1`.
"""

STARTER = '''import numpy as np

def user_knn_predict(R, M, queries, k):
    """Return (Q,) predicted ratings."""
    R, M = np.asarray(R, dtype=float), np.asarray(M, dtype=float)
    mu = (R * M).sum(1) / M.sum(1)
    return np.array([mu[u] for u, i in queries])
'''

SOLUTION = '''import numpy as np

def user_knn_predict(R, M, queries, k):
    R, M = np.asarray(R, dtype=float), np.asarray(M, dtype=float)
    mu = (R * M).sum(1) / M.sum(1)
    Z = (R - mu[:, None]) * M                       # 0 wherever unobserved
    num = Z @ Z.T                                   # only co-rated items contribute
    A = (Z ** 2) @ M.T                              # A[u, v] = sum of Z_ui^2 over items v also rated
    den = np.sqrt(A * A.T)
    S = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    out = np.empty(len(queries))
    for q, (u, i) in enumerate(np.asarray(queries).tolist()):
        cand = [v for v in range(len(R)) if v != u and M[v, i] > 0 and S[u, v] > 0]
        cand.sort(key=lambda v: (-S[u, v], v))      # most similar first, ties by index
        nb = cand[:k]
        if not nb:
            out[q] = mu[u]
            continue
        w = S[u, nb]
        out[q] = mu[u] + w @ (R[nb, i] - mu[nb]) / w.sum()
    return out
'''

HINTS = [
    'Centre with each user\'s mean and multiply by the mask, so unobserved entries become exact zeros. Then `Z @ Z.T` sums only over co-rated items.',
    'The norms are restricted to co-rated items too: `(Z**2) @ M.T` gives, for each pair (u, v), the sum of $Z_{ui}^2$ over items $v$ also rated. Its transpose gives the other factor.',
    'For each query, filter candidates (rated the item, positive similarity, not u), sort by `(-s, index)`, keep `k`, and fall back to $\\mu_u$ when none remain.',
]

_r = np.random.default_rng(17)
_R = np.round(_r.uniform(1, 5, size=(10, 8)), 3)
_M = (_r.random((10, 8)) < 0.6).astype(int)
_M[:, 0] = 1
_Q = np.array([[u, i] for u in range(0, 10, 3) for i in range(8)])

TESTS = [
    (np.array([[5.0, 3.0, 0.0], [4.0, 2.0, 4.0], [1.0, 5.0, 2.0]]), np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]]),
     np.array([[0, 2]]), 2),
    (np.array([[4.0, 2.0, 3.0, 5.0], [5.0, 1.0, 2.0, 4.0], [3.0, 3.0, 4.0, 1.0], [5.0, 2.0, 1.0, 5.0]]),
     np.ones((4, 4), dtype=int), np.array([[0, 2], [2, 0]]), 1),
    (np.array([[3.0, 0.0], [0.0, 4.0]]), np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]), 3),
    (np.array([[2.0, 2.0, 2.0], [1.0, 5.0, 3.0]]), np.array([[1, 1, 0], [1, 1, 1]]), np.array([[0, 2]]), 1),
    (np.array([[5.0, 1.0, 0.0], [1.0, 5.0, 2.0], [5.0, 1.0, 5.0]]), np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]]),
     np.array([[0, 2]]), 5),
    (_R, _M, _Q, 1),
    (_R, _M, _Q, 3),
    (_R, _M, _Q, 10),
]

EXPLANATION = r"""
## Why it works
Centring by $\mu_u$ turns ratings into "liked more or less than usual", so a harsh rater and a generous rater with the same taste look alike. The similarity is a cosine of those deviations over the items both users rated, a Pearson-style correlation that is 1 for identical taste and negative for opposite taste. The prediction starts at the target user's own mean and adds a similarity-weighted average of the neighbours' **deviations** on the item, translated back to $u$'s scale. Test 4 shows the fallback: user 0 rates everything 2, so all deviations are 0, the similarity is 0, and the prediction is just $\mu_0 = 2$.

## Complexity
$O(U^2 I)$ for the similarity matrix, then $O(U \log U)$ per query for the neighbour sort (a partial sort with `np.argpartition` makes it $O(U)$).

## Common bugs
- Normalising by each user's **full** norm instead of the co-rated norm. Users who rated many other items then look less similar for no reason.
- Averaging neighbours' raw ratings instead of their deviations, which reintroduces rater bias.
- Keeping neighbours with negative similarity in a formula that divides by $\sum s$: the denominator can approach 0 and blow up.
- Forgetting to exclude $u$ itself when $u$ has rated $i$.

## Interview follow-ups
- Two users share one co-rated item and get $s = 1$. How do you stop such flukes dominating? (Significance weighting: multiply by $\min(|\Omega_{uv}|, \gamma)/\gamma$, or shrink like $|\Omega|/(|\Omega| + \beta)$.)
- Why do user-kNN systems struggle at scale and with fresh users? (Similarities change with every new rating and need $O(U^2)$ work; a new user has no co-ratings yet.)
"""
