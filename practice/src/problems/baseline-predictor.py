import numpy as np

META = dict(title='Baseline predictor with regularised biases', tags=['Recommender Systems', 'Regression'], diff='Medium', runtime='numpy',
            related=['matrix-factorization-sgd-step'])

STATEMENT = r"""
Fit the baseline $\hat r_{ui} = \mu + b_u + b_i$ the way Koren describes, with two shrunk averages.

Write `fit_baseline(ratings, n_users, n_items, lam_i, lam_u)` for `(u, i, r)` triples:

1. $\mu$ = mean of all ratings.
2. $b_i = \dfrac{\sum_{u \in R(i)} (r_{ui} - \mu)}{\lambda_i + |R(i)|}$.
3. $b_u = \dfrac{\sum_{i \in R(u)} (r_{ui} - \mu - b_i)}{\lambda_u + |R(u)|}$.

Users or items with no ratings get bias 0. Return `(mu, bu, bi)` with `bu` of length `n_users` and `bi` of length `n_items`.
"""

STARTER = '''import numpy as np

def fit_baseline(ratings, n_users, n_items, lam_i, lam_u):
    """Return (mu, bu, bi)."""
    mu = float(np.mean([r for _, _, r in ratings]))
    return mu, np.zeros(n_users), np.zeros(n_items)
'''

SOLUTION = '''import numpy as np

def fit_baseline(ratings, n_users, n_items, lam_i, lam_u):
    u = np.array([x[0] for x in ratings]); i = np.array([x[1] for x in ratings])
    r = np.array([x[2] for x in ratings], float)
    mu = float(r.mean())
    bi = np.bincount(i, r - mu, n_items) / (lam_i + np.bincount(i, minlength=n_items))
    bu = np.bincount(u, r - mu - bi[i], n_users) / (lam_u + np.bincount(u, minlength=n_users))
    return mu, bu, bi
'''

HINTS = [
    'Item biases come first; user biases are computed on residuals after removing mu and b_i.',
    'np.bincount(idx, weights, minlength) sums weights per index, which gives every numerator in one call.',
    'The lambda in the denominator pulls biases toward 0 for items with few ratings.',
]

def _case(seed, nu, ni, n, li, lu):
    g = np.random.default_rng(seed)
    return ([(int(g.integers(nu)), int(g.integers(ni)), float(g.integers(1, 6))) for _ in range(n)], nu, ni, li, lu)

TESTS = [([(0, 0, 5.0), (1, 0, 3.0), (1, 1, 1.0)], 2, 2, 0.0, 0.0),
         ([(0, 0, 5.0), (1, 0, 3.0), (1, 1, 1.0)], 2, 3, 25.0, 10.0),
         _case(1, 5, 6, 20, 10.0, 5.0), _case(2, 10, 4, 50, 0.0, 0.0), _case(3, 8, 8, 30, 25.0, 10.0)]

EXPLANATION = r"""
## Why a baseline
Before any factor model, the global mean plus user and item offsets already explains a large share of rating variance. It is the floor every recommender should beat, and it is the starting point for the bias terms in matrix factorisation.

## Why shrinkage
An item with one 5-star rating would otherwise get $b_i = 5 - \mu$. Dividing by $\lambda + n$ treats $\lambda$ as pseudo-ratings at the global mean, so sparse items stay near zero until evidence accumulates. Koren used $\lambda_i = 25$ and $\lambda_u = 10$ on Netflix data.

## Order matters
Computing $b_u$ on residuals after $b_i$ avoids double counting: a user who watches only popular films is not mistaken for a generous rater.

## Interview follow-ups
- How would you add a time-dependent item bias for films whose popularity drifts?
"""
