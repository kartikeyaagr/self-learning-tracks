import numpy as np

META = dict(title='Matrix factorisation: one SGD epoch with biases', tags=['Recommender Systems', 'Optimization'], diff='Medium', runtime='numpy',
            related=['bpr-sgd-step', 'baseline-predictor'])

STATEMENT = r"""
Explicit-feedback matrix factorisation predicts $\hat r_{ui} = \mu + b_u + b_i + p_u \cdot q_i$.

Write `mf_epoch(ratings, mu, bu, bi, P, Q, lr, reg)` that runs one pass of SGD over `ratings`, a list of `(u, i, r)` triples, **in the given order**. For each triple, with error $e = r - \hat r_{ui}$ computed from the current parameters:

- $b_u \mathrel{+}= \eta(e - \lambda b_u)$, $b_i \mathrel{+}= \eta(e - \lambda b_i)$
- $p_u \mathrel{+}= \eta(e\, q_i - \lambda p_u)$ and $q_i \mathrel{+}= \eta(e\, p_u - \lambda q_i)$, both using the values of $p_u$ and $q_i$ from **before** this update.

`mu` stays fixed. Do not modify the inputs. Return `(bu, bi, P, Q, rmse)` where `rmse` is the root mean squared error over `ratings` **after** the epoch.
"""

STARTER = '''import numpy as np

def mf_epoch(ratings, mu, bu, bi, P, Q, lr, reg):
    """Return updated (bu, bi, P, Q, rmse)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def mf_epoch(ratings, mu, bu, bi, P, Q, lr, reg):
    bu, bi = np.array(bu, float), np.array(bi, float)
    P, Q = np.array(P, float), np.array(Q, float)
    for u, i, r in ratings:
        e = r - (mu + bu[u] + bi[i] + P[u] @ Q[i])
        bu[u] += lr * (e - reg * bu[u])
        bi[i] += lr * (e - reg * bi[i])
        pu, qi = P[u].copy(), Q[i].copy()
        P[u] += lr * (e * qi - reg * pu)
        Q[i] += lr * (e * pu - reg * qi)
    err = [r - (mu + bu[u] + bi[i] + P[u] @ Q[i]) for u, i, r in ratings]
    return bu, bi, P, Q, float(np.sqrt(np.mean(np.square(err))))
'''

HINTS = [
    'Copy P[u] and Q[i] before updating either, or the second update sees the new value of the first.',
    "np.array(x, float) makes a copy, so the caller's arrays stay unchanged.",
]

def _case(seed, nu, ni, k, n):
    r = np.random.default_rng(seed)
    R = [(int(r.integers(nu)), int(r.integers(ni)), float(r.integers(1, 6))) for _ in range(n)]
    return (R, 3.5, np.zeros(nu), np.zeros(ni), 0.1 * r.normal(size=(nu, k)), 0.1 * r.normal(size=(ni, k)), 0.05, 0.02)

TESTS = [([(0, 0, 5.0)], 3.0, [0.0], [0.0], [[0.1, 0.2]], [[0.3, -0.1]], 0.1, 0.0),
         _case(1, 4, 5, 2, 12), _case(2, 10, 8, 3, 40), _case(3, 3, 3, 1, 6), _case(4, 6, 6, 4, 30)]

EXPLANATION = r"""
## Why biases first
Most of the variance in ratings is "this user rates high" and "this film is liked". The bias terms absorb that so the latent factors only model interaction effects. This was one of the larger single gains in the Netflix Prize models.

## The simultaneous update
The gradient of the squared error with respect to $p_u$ involves $q_i$ and vice versa. Updating $p_u$ first and then using the new value for $q_i$ is a different (and slightly wrong) step; with large learning rates it can oscillate.

## SGD versus ALS
SGD is simple and handles ratings one at a time. Alternating least squares fixes one side and solves the other exactly, which parallelises well and is the usual choice for implicit feedback where every missing cell is a weak negative.

## Interview follow-ups
- How would you handle a new user with no ratings?
"""
