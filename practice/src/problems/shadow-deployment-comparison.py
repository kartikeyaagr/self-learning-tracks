import numpy as np

META = dict(title='Shadow deployment comparison and promotion call', tags=['MLOps', 'Metrics & Evaluation'], diff='Medium', runtime='numpy',
            related=['canary-mann-whitney'])

STATEMENT = r"""
A candidate model runs in shadow: it scores the same requests as production, but its outputs are not served. Write `shadow_report(p_prod, p_shadow, labels, lat_prod, lat_shadow, threshold, min_gain, p99_budget)`.

- `labels` may contain `NaN` for requests whose outcome is not known yet.
- `flip_rate`: fraction of **all** requests where the two models fall on different sides of `threshold` (a score `>= threshold` is positive).
- `logloss_prod`, `logloss_shadow`: binary log loss on labelled requests, with probabilities clipped to `[1e-7, 1 - 1e-7]`.
- `p99_shadow`: `np.percentile(lat_shadow, 99)` (default linear interpolation).
- `decision`: `"promote"` if there are at least 100 labelled requests, `logloss_shadow <= logloss_prod - min_gain`, and `p99_shadow <= p99_budget`; `"hold"` if the first condition fails; otherwise `"reject"`.

Return a dict with those five keys plus `n_labelled`.
"""

STARTER = '''import numpy as np

def shadow_report(p_prod, p_shadow, labels, lat_prod, lat_shadow, threshold, min_gain, p99_budget):
    """Return the shadow comparison dict."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _ll(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def shadow_report(p_prod, p_shadow, labels, lat_prod, lat_shadow, threshold, min_gain, p99_budget):
    a, b = np.asarray(p_prod, float), np.asarray(p_shadow, float)
    y = np.asarray(labels, float)
    m = ~np.isnan(y)
    n = int(m.sum())
    flip = float(np.mean((a >= threshold) != (b >= threshold)))
    llp = _ll(a[m], y[m]) if n else float("nan")
    lls = _ll(b[m], y[m]) if n else float("nan")
    p99 = float(np.percentile(np.asarray(lat_shadow, float), 99))
    if n < 100:
        decision = "hold"
    elif lls <= llp - min_gain and p99 <= p99_budget:
        decision = "promote"
    else:
        decision = "reject"
    return {"flip_rate": flip, "logloss_prod": llp, "logloss_shadow": lls,
            "p99_shadow": p99, "n_labelled": n, "decision": decision}
'''

HINTS = [
    'Compute flip rate on every request, but log loss only where the label is known.',
    'Decide "hold" first: with too few labels neither promote nor reject is justified.',
    'Clip probabilities before the log so a confident wrong prediction does not give infinity.',
]

def _case(seed, n, frac_lab, better, lat_scale):
    r = np.random.default_rng(seed)
    y = (r.random(n) < 0.3).astype(float)
    base = np.clip(0.3 + 0.25 * (y - 0.3) + 0.15 * r.normal(size=n), 0.01, 0.99)
    shadow = np.clip(base + better * (y - base) + 0.02 * r.normal(size=n), 0.01, 0.99)
    lab = np.where(r.random(n) < frac_lab, y, np.nan)
    return (base, shadow, lab, r.gamma(2, 10, n), r.gamma(2, 10 * lat_scale, n), 0.5, 0.005, 80.0)

TESTS = [_case(1, 500, 0.8, 0.3, 1.0), _case(2, 500, 0.8, 0.3, 3.0), _case(3, 500, 0.1, 0.3, 1.0),
         _case(4, 500, 0.9, -0.2, 1.0), _case(5, 2000, 0.5, 0.0, 1.0), _case(6, 300, 1.0, 0.5, 0.5)]

EXPLANATION = r"""
## What shadow traffic tells you
Shadowing runs the candidate on real production inputs, including the messy ones offline data misses, without user risk. You get latency under real load, the rate of decision changes, and, once labels arrive, a quality comparison on the same requests.

## Why flip rate matters
Two models with the same log loss can disagree on many individual decisions. A high flip rate means users would notice the change, so even a better model may need a gradual rollout and checks on the flipped cases.

## Delayed labels
Many outcomes (fraud chargebacks, conversions) arrive days later. The "hold" state exists so a decision is not made on the small, biased subset of quick labels.

## Interview follow-ups
- Why can shadow evaluation not measure feedback effects, such as a recommender changing what users click?
"""
