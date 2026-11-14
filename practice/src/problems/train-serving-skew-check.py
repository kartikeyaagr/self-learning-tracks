import numpy as np

META = dict(title='Detect train-serving skew from logged features', tags=['MLOps', 'Feature Engineering'], diff='Medium', runtime='numpy',
            related=['population-stability-index', 'feature-store-lookup'])

STATEMENT = r"""
To catch skew, a service logs the features it actually served, and the same requests are recomputed by the offline training pipeline. Write `skew_report(offline, online, abs_tol, rel_tol, max_mismatch)`.

- `offline` and `online` are dicts from feature name to equal-length arrays, row-aligned by request. Only features present in both are compared; report the others under `"missing_online"` and `"missing_offline"` (sorted lists).
- A row **mismatches** for a feature if either value is `NaN` while the other is not, or both are finite and $|a - b| > \text{abs\_tol} + \text{rel\_tol} \cdot |a|$ where $a$ is the offline value. Two `NaN`s match.
- For each compared feature report `[mismatch_rate, max_abs_diff]`, where `max_abs_diff` is over rows where both values are finite (0.0 if none).
- `"flagged"` lists features with mismatch rate above `max_mismatch`, sorted by rate descending then name.

Return a dict with keys `"features"`, `"flagged"`, `"missing_online"`, `"missing_offline"`.
"""

STARTER = '''import numpy as np

def skew_report(offline, online, abs_tol, rel_tol, max_mismatch):
    """Compare offline and served features row by row."""
    feats = {}
    for f in offline:
        d = np.abs(np.asarray(offline[f]) - np.asarray(online[f]))
        feats[f] = [float((d > abs_tol).mean()), float(d.max())]
    return {"features": feats, "flagged": [], "missing_online": [], "missing_offline": []}
'''

SOLUTION = '''import numpy as np

def skew_report(offline, online, abs_tol, rel_tol, max_mismatch):
    common = sorted(set(offline) & set(online))
    feats = {}
    for f in common:
        a, b = np.asarray(offline[f], float), np.asarray(online[f], float)
        na, nb = np.isnan(a), np.isnan(b)
        both = ~na & ~nb
        diff = np.abs(a - b)
        bad = (na != nb) | (both & (diff > abs_tol + rel_tol * np.abs(a)))
        feats[f] = [float(bad.mean()) if len(a) else 0.0, float(diff[both].max()) if both.any() else 0.0]
    flagged = sorted((f for f in common if feats[f][0] > max_mismatch), key=lambda f: (-feats[f][0], f))
    return {"features": feats, "flagged": flagged,
            "missing_online": sorted(set(offline) - set(online)),
            "missing_offline": sorted(set(online) - set(offline))}
'''

HINTS = [
    'Handle NaN explicitly: NaN compared with anything is False, so a plain difference test silently treats one-sided NaNs as matches.',
    'Use a combined absolute and relative tolerance so large-valued features are not flagged for rounding noise.',
    'A feature missing on one side is itself a skew bug; report it rather than raising.',
]

_r = np.random.default_rng(31)
_n = 200
_off = {"age": _r.integers(18, 70, _n).astype(float), "spend_7d": _r.gamma(2, 50, _n), "ctr_30d": _r.beta(2, 50, _n),
        "country_idx": _r.integers(0, 5, _n).astype(float)}
_on = {"age": _off["age"].copy(), "spend_7d": _off["spend_7d"] * (1 + 1e-7 * _r.normal(size=_n)),
       "ctr_30d": np.where(_r.random(_n) < 0.1, np.nan, _off["ctr_30d"]), "device": np.ones(_n)}
_on["age"][:30] += 1
TESTS = [
    (_off, _on, 1e-6, 1e-4, 0.05),
    (_off, _on, 1e-6, 1e-9, 0.05),
    (_off, _on, 2.0, 0.0, 0.01),
    ({"x": [1.0, np.nan, 3.0]}, {"x": [1.0, np.nan, np.nan]}, 0.0, 0.0, 0.0),
    ({"x": [100.0, 200.0]}, {"x": [100.5, 201.0]}, 0.0, 0.004, 0.4),
    ({}, {"y": [1.0]}, 0.0, 0.0, 0.0),
]

EXPLANATION = r"""
## What skew is
The model was trained on features computed by a batch pipeline and is served features computed by a different, online code path. When the two disagree (a different time window, a timezone, a default for missing values, a stale join) the model sees inputs it never saw in training. Offline metrics stay good while online metrics fall.

## Why compare row by row
Distribution checks such as PSI catch drift in the population. Skew can leave the distribution almost unchanged while every individual value is off by a day's worth of data. Logging served features and recomputing them offline for the same requests is the direct test.

## Typical findings
Off-by-one windows (age above), missing values filled differently online (the NaN case), and features that exist only on one side because a schema change was deployed to one path.

## Interview follow-ups
- How does a feature store that serves training and inference from one definition remove most of this class of bug?
"""
