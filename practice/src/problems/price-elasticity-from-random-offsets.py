import numpy as np

META = dict(title='Price elasticity from randomised price offsets', tags=['Regression', 'Probability and Statistics'], diff='Medium', runtime='numpy')

STATEMENT = r"""
A marketplace randomly shifts prices by a small percentage per session to learn demand. Write `elasticity(base_price, offset, bought)`:

- The shown price is `base_price * (1 + offset)`. `bought` is 0 or 1.
- Group sessions by offset value. For each group compute the purchase rate.
- Fit $\ln(\text{rate}) = \alpha + \varepsilon \ln(\text{mean shown price in the group})$ by ordinary least squares, **weighting each group by its number of sessions**. Groups with zero purchases are dropped (log of zero).
- Return `(epsilon, alpha, n_groups_used)`. If fewer than 2 groups remain, return `(nan, nan, n)`.
"""

STARTER = '''import numpy as np

def elasticity(base_price, offset, bought):
    """Return (epsilon, alpha, n_groups_used)."""
    p = np.asarray(base_price) * (1 + np.asarray(offset))
    e, a = np.polyfit(np.log(p), np.asarray(bought, float), 1)
    return float(e), float(a), 1
'''

SOLUTION = '''import numpy as np

def elasticity(base_price, offset, bought):
    b = np.asarray(base_price, float); o = np.asarray(offset, float); y = np.asarray(bought, float)
    price = b * (1 + o)
    xs, ys, ws = [], [], []
    for v in np.unique(o):
        m = o == v
        rate = y[m].mean()
        if rate > 0:
            xs.append(np.log(price[m].mean())); ys.append(np.log(rate)); ws.append(m.sum())
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), n
    X = np.c_[np.ones(n), xs]
    W = np.asarray(ws, float)
    coef = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * np.asarray(ys)))
    return float(coef[1]), float(coef[0]), n
'''

HINTS = [
    'Weighted least squares: solve (X^T W X) beta = X^T W y with W the group sizes.',
    'In a log-log model the slope is the elasticity: a 1% price rise changes demand by epsilon percent.',
    'Aggregate per offset first; regressing 0/1 outcomes on log price directly is a different model.',
]

def _case(seed, n, eps, base_lo, base_hi, offs):
    r = np.random.default_rng(seed)
    base = r.uniform(base_lo, base_hi, n)
    o = r.choice(offs, n)
    p = base * (1 + o)
    rate = np.clip(0.2 * (p / base.mean()) ** eps, 0, 1)
    return (base, o, (r.random(n) < rate).astype(int))

TESTS = [
    _case(1, 4000, -1.5, 20, 20, [-0.1, -0.05, 0.0, 0.05, 0.1]),
    _case(2, 4000, -0.5, 10, 30, [-0.1, 0.0, 0.1]),
    _case(3, 3000, -3.0, 50, 50, [-0.2, -0.1, 0.0, 0.1, 0.2]),
    ([10, 10, 10, 10], [0.0, 0.0, 0.1, 0.1], [1, 0, 0, 0]),
    ([10, 10, 10], [0.0, 0.1, 0.2], [1, 0, 1]),
]

EXPLANATION = r"""
## Why randomise
Observed prices are not random: sellers raise prices when demand is high. Regressing sales on observed prices then mixes the demand curve with the pricing policy and often gives a wrong-signed elasticity. Small random offsets break that link, so the price variation is exogenous and the slope is causal.

## Why log-log
With $\ln q = \alpha + \varepsilon \ln p$, the slope is the elasticity directly. Demand with $|\varepsilon| > 1$ is elastic: raising prices loses revenue.

## Practical limits
Offsets must be small enough not to hurt users and large enough to measure. Groups with zero purchases bias the fit upward when dropped; with sparse data, a Poisson regression on counts is the safer estimator.

## Interview follow-ups
- How would you let elasticity vary by category without running a separate experiment for each?
"""
