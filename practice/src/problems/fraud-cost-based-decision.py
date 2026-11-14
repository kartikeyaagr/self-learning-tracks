import numpy as np

META = dict(title='Fraud decisions from expected cost', tags=['Classification', 'Metrics & Evaluation'], diff='Medium', runtime='numpy')

STATEMENT = r"""
A payments model outputs a calibrated fraud probability $p$ per transaction of amount $a$. Three actions are possible:

- `"approve"`: expected cost $p \cdot a$ (you lose the amount if it is fraud).
- `"review"`: cost `review_cost` plus $p \cdot a \cdot$ `miss_rate` (reviewers miss some fraud). Reviews use one slot each, and at most `capacity` are available.
- `"decline"`: expected cost $(1 - p) \cdot a \cdot$ `margin` (lost profit on a good customer).

Write `decide(p, amount, review_cost, miss_rate, margin, capacity)`:

1. For each transaction pick the cheaper of approve and decline (ties go to approve). Call that the fallback.
2. The saving from review is fallback cost minus review cost. Give review slots to the transactions with the largest **positive** savings (ties to the lower index), up to `capacity`.

Return `(actions, total_expected_cost)`.
"""

STARTER = '''import numpy as np

def decide(p, amount, review_cost, miss_rate, margin, capacity):
    """Return (list of actions, total expected cost)."""
    acts = ["decline" if q > 0.5 else "approve" for q in p]
    return acts, 0.0
'''

SOLUTION = '''import numpy as np

def decide(p, amount, review_cost, miss_rate, margin, capacity):
    p, a = np.asarray(p, float), np.asarray(amount, float)
    c_app, c_dec = p * a, (1 - p) * a * margin
    c_rev = review_cost + p * a * miss_rate
    fallback = np.minimum(c_app, c_dec)
    acts = np.where(c_dec < c_app, "decline", "approve").astype(object)
    cost = fallback.copy()
    saving = fallback - c_rev
    order = sorted(range(len(p)), key=lambda i: (-saving[i], i))
    for i in [i for i in order if saving[i] > 0][:capacity]:
        acts[i], cost[i] = "review", c_rev[i]
    return [str(x) for x in acts], float(cost.sum())
'''

HINTS = [
    'Approve beats decline exactly when p * a <= (1 - p) * a * margin, so the break-even probability is margin / (1 + margin).',
    'Review is a scarce resource: rank by how much cost it removes, not by fraud probability.',
    'A transaction with zero or negative saving should never use a slot.',
]

_r = np.random.default_rng(8)
_p = np.clip(_r.beta(0.5, 6, 60), 0, 1)
_a = np.round(_r.lognormal(4, 1, 60), 2)
TESTS = [
    ([0.01, 0.5, 0.9], [100, 100, 100], 5.0, 0.1, 0.3, 1),
    ([0.01, 0.5, 0.9], [100, 100, 100], 5.0, 0.1, 0.3, 0),
    (_p, _a, 3.0, 0.2, 0.25, 5),
    (_p, _a, 3.0, 0.2, 0.25, 60),
    ([0.2, 0.2], [1000, 10], 2.0, 0.0, 1.0, 1),
    ([0.0, 1.0], [50, 50], 1.0, 0.5, 0.2, 2),
]

EXPLANATION = r"""
## Why not a fixed threshold
Losing a fraudulent \$5,000 transfer is not the same as losing a \$5 coffee. Multiplying probability by amount turns the score into money, and the best action changes with the amount even at the same probability.

## Review as a budgeted resource
Human review is the most accurate action but has fixed daily capacity. Ranking by expected saving sends analysts to the cases where their judgement is worth the most: usually mid-probability, high-amount transactions, not the obvious fraud that would be declined anyway.

## What this depends on
Calibration. If the model's 0.1 really means 0.3, every expected cost is wrong. Recalibrate after each retrain and after any change in the fraud base rate.

## Interview follow-ups
- How do you estimate the cost of declining a good customer, which includes churn, not only lost margin?
"""
