import numpy as np

META = dict(title='Evaluate ETA predictions by segment', tags=['Regression', 'Metrics & Evaluation'], diff='Easy', runtime='numpy')

STATEMENT = r"""
Write `eta_report(pred, actual, segment, late_tol)` for trip time predictions in minutes.

For each segment (sorted by name) and for `"ALL"` last, report:

- `n`
- `mae`: mean absolute error
- `mape`: mean of $|pred - actual| / actual$ over trips with `actual > 0`, or `nan` if none
- `bias`: mean of `pred - actual` (negative means the app promises too early)
- `late_rate`: fraction of trips where `actual > pred + late_tol`, the trips that arrived later than promised beyond the tolerance
- `p90_abs`: `np.percentile(abs error, 90)`

Return a dict from segment name to a dict of these values.
"""

STARTER = '''import numpy as np

def eta_report(pred, actual, segment, late_tol):
    """Per-segment ETA error report."""
    e = np.abs(np.asarray(pred) - np.asarray(actual))
    return {"ALL": {"n": len(e), "mae": float(e.mean())}}
'''

SOLUTION = '''import numpy as np

def _stats(p, a, tol):
    d = p - a
    pos = a > 0
    return {"n": int(len(p)), "mae": float(np.abs(d).mean()),
            "mape": float((np.abs(d[pos]) / a[pos]).mean()) if pos.any() else float("nan"),
            "bias": float(d.mean()), "late_rate": float((a > p + tol).mean()),
            "p90_abs": float(np.percentile(np.abs(d), 90))}

def eta_report(pred, actual, segment, late_tol):
    p, a, g = np.asarray(pred, float), np.asarray(actual, float), np.asarray(segment)
    out = {str(s): _stats(p[g == s], a[g == s], late_tol) for s in sorted(set(g.tolist()))}
    out["ALL"] = _stats(p, a, late_tol)
    return out
'''

HINTS = [
    'Write one helper that computes all metrics for a pair of arrays, then call it per segment and once for everything.',
    'Late rate is one-sided: a trip that arrives early is not late, however large the error.',
]

_r = np.random.default_rng(15)
_a = _r.gamma(4, 5, 300)
_seg = _r.choice(["airport", "city", "suburb"], 300)
_p = _a * (1 + 0.1 * _r.normal(size=300)) - np.where(_seg == "airport", 4, 0)
TESTS = [
    ([10, 20, 30], [12, 18, 40], ["a", "b", "a"], 2.0),
    (_p, _a, _seg, 3.0),
    ([5, 5], [0, 5], ["x", "x"], 0.0),
    ([0.0], [0.0], ["z"], 1.0),
    (_p[:50], _a[:50], ["one"] * 50, 5.0),
]

EXPLANATION = r"""
## Why more than MAE
Users judge ETAs asymmetrically. Arriving five minutes early is pleasant; arriving five minutes late after a promise is a complaint. Bias and late rate measure that directly, and a model with lower MAE but a negative bias can be worse for the product.

## Why segment
An overall metric hides groups where the model fails, such as airport trips with long waits at pick-up or rush-hour city trips. Reporting by segment is how you find the slice that needs a feature or a separate correction.

## Tail errors
The 90th percentile of absolute error tracks the bad experiences that drive churn; MAE can improve while the tail gets worse.

## Interview follow-ups
- Should the model be trained on a loss that matches this asymmetry, such as a quantile loss?
"""
