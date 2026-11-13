import numpy as np

META = dict(title='AdaBoost with decision stumps', tags=['Classic ML', 'Classification'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:04.07'], related=['gradient-boosted-regressor', 'bagging-classifier'])

STATEMENT = r"""
Discrete AdaBoost builds a strong classifier $F(x) = \sum_t \alpha_t h_t(x)$ from weak ones by re-weighting the training set after every round, so the next learner focuses on the examples the ensemble still gets wrong.

Write `adaboost(X_train, y_train, X_test, n_rounds)` that returns `(scores, rounds_used)`, where `scores` is $F$ evaluated on `X_test`.

## Weak learner: a stump
$h(x) = s$ if $x_j > t$, else $-s$, with polarity $s \in \{+1, -1\}$.
- Candidate thresholds for feature $j$: midpoints $(v_k + v_{k+1})/2$ between consecutive sorted distinct values of feature $j$ in the **whole** training set (they don't change between rounds).
- Weighted error: $\varepsilon = \sum_i w_i\,[h(x_i) \ne y_i]$.
- Scan features ascending, then thresholds ascending, then $s = +1$ before $s = -1$. A candidate replaces the current best only if its error is lower by more than `1e-12`.

## Boosting (follow exactly)
1. Start with $w_i = 1/n$.
2. Each round: pick the best stump; clip its error to $\tilde\varepsilon = \min(\max(\varepsilon, 10^{-10}),\ 1 - 10^{-10})$; set $\alpha = \tfrac12\ln\frac{1-\tilde\varepsilon}{\tilde\varepsilon}$; update $w_i \leftarrow w_i\, e^{-\alpha y_i h(x_i)}$ and renormalise to sum to 1.
3. Add $\alpha\,h(x)$ to the test scores.
4. Stop after `n_rounds` rounds, or right after a round whose **unclipped** error was below $10^{-10}$ (that stump is still included).

## Input / output
- `X_train`: float `(n, d)` with at least one non-constant feature; `y_train`: int `(n,)` in $\{-1, +1\}$; `X_test`: float `(m, d)`; `n_rounds >= 1`.
- `scores`: float array `(m,)`; `rounds_used`: Python `int`.
"""

STARTER = '''import numpy as np

def adaboost(X_train, y_train, X_test, n_rounds):
    """Discrete AdaBoost with stumps. Return (F(X_test) scores (m,), rounds_used)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def adaboost(X_train, y_train, X_test, n_rounds):
    X, y, Xt = np.asarray(X_train, float), np.asarray(y_train, float), np.asarray(X_test, float)
    n, d = X.shape
    cands = []
    for j in range(d):
        v = np.unique(X[:, j])
        cands += [(j, t, np.where(X[:, j] > t, 1.0, -1.0)) for t in (v[:-1] + v[1:]) / 2]
    w = np.full(n, 1.0 / n)
    scores, used = np.zeros(len(Xt)), 0
    for _ in range(n_rounds):
        best = None
        for j, t, base in cands:
            for s in (1.0, -1.0):
                err = w[s * base != y].sum()
                if best is None or err < best[0] - 1e-12:
                    best = (err, j, t, s)
        err, j, t, s = best
        e = min(max(err, 1e-10), 1 - 1e-10)
        alpha = 0.5 * np.log((1 - e) / e)
        h = s * np.where(X[:, j] > t, 1.0, -1.0)
        w = w * np.exp(-alpha * y * h)                  # up-weight mistakes, down-weight hits
        w /= w.sum()
        scores += alpha * s * np.where(Xt[:, j] > t, 1.0, -1.0)
        used += 1
        if err < 1e-10:                                 # perfect stump: later rounds would repeat it
            break
    return scores, used
'''

HINTS = [
    'Precompute every `(feature, threshold)` pair and the ±1 prediction vector it gives on the training set once. Each round then only re-scores them under new weights.',
    'Flipping the polarity turns error $\\varepsilon$ into $1 - \\varepsilon$, so the best stump never has error above 0.5. Still scan $s=+1$ then $s=-1$ explicitly so ties resolve the same way as the reference.',
    'Per round: `alpha = 0.5 * log((1 - e) / e)` with the clipped error, `w *= exp(-alpha * y * h)`, `w /= w.sum()`, and `scores += alpha * h_test`. Break after adding a stump whose raw error is below `1e-10`.',
]

_r = np.random.default_rng(67)
_X = _r.uniform(-2, 2, size=(50, 2))
_y = np.where(_X[:, 0] ** 2 + _X[:, 1] ** 2 < 1.5, 1, -1)
_Xt = _r.uniform(-2, 2, size=(10, 2))
TESTS = [
    (np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([-1, -1, 1, 1]), np.array([[0.0], [2.4], [2.6]]), 5),
    (np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), np.array([1, -1, -1, 1, 1]), np.array([[1.0], [2.5], [4.5]]), 3),
    (np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([1, 1, -1, -1]), np.array([[0.0], [5.0]]), 4),
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([-1, 1, 1, -1]), np.array([[0.0, 0.0], [1.0, 0.0]]), 6),
    (np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]), np.array([1, -1, 1]), np.array([[2.0, 5.0]]), 1),
    (_X, _y, _Xt, 10),
    (_X, _y, _Xt, 30),
    (_X * 100.0, _y, _Xt * 100.0, 15),
]

EXPLANATION = r"""
## Why these formulas
AdaBoost is forward stagewise additive modelling with the exponential loss $\sum_i e^{-y_i F(x_i)}$. Given the current $F$, adding $\alpha h$ multiplies sample $i$'s loss by $e^{-\alpha y_i h(x_i)}$, so the current weights $w_i \propto e^{-y_i F(x_i)}$ are exactly each sample's share of the loss. Minimising over $h$ means minimising the weighted error. Minimising over $\alpha$ gives $\alpha = \tfrac12\ln\frac{1-\varepsilon}{\varepsilon}$.

## A useful invariant
After the update, the stump just chosen has weighted error exactly $1/2$ under the new weights. So the next round can't pick it again unless nothing better exists, which is what pushes the ensemble toward the hard examples.

## Training error bound
The training error is at most $\prod_t 2\sqrt{\varepsilon_t(1-\varepsilon_t)}$, which shrinks exponentially as long as every weak learner beats chance by a fixed margin.

## Complexity
With $K$ precomputed candidates, each round is $O(Kn)$ for the errors. Sorting by feature and using cumulative weight sums brings a round down to $O(dn)$.

## Common bugs
- Forgetting to renormalise, so the weights drift and the errors are no longer in $[0, 1]$.
- Using $\ln$ without clipping: a perfect stump (test 1) divides by zero.
- Recomputing thresholds from the weighted sample: the candidate set is fixed, only the weights change.

## Interview follow-ups
- Why is AdaBoost sensitive to label noise, and how does the exponential loss explain it? (Mislabelled points get exponentially growing weight.)
- How does gradient boosting generalise AdaBoost to any differentiable loss?
"""
