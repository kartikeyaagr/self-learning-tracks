import pandas as pd

META = dict(title='Training/serving skew report', tags=['MLOps', 'Data Processing', 'Feature Engineering'], diff='Medium', runtime='pandas',
            lessons=['air-fanout-v1:11.12', 'air-fanout-v1:11.01'], related=['record-schema-validation', 'point-in-time-feature-join'])

STATEMENT = r"""
The online service computes features with one code path; the offline training pipeline recomputes them with another. When the two disagree, the model is trained on numbers it never sees in production. To catch this, the service logs the features it actually used, and you compare them with what the training pipeline produced for the same requests.

Write `skew_report(train_log, serve_log, features, atol=1e-6)` that returns one row per feature.

## Inputs
- `train_log` and `serve_log`: DataFrames with an int column `request_id` (unique within each frame) and every column in `features`. Other columns may exist; ignore them.
- `features`: list of column names. A feature is **numeric** if its column has a numeric dtype in `train_log` (`pd.api.types.is_numeric_dtype`); otherwise it holds strings or `None`. Each feature has the same kind in both frames.

## Comparison
Compare only requests present in both logs (inner join on `request_id`). For each feature and each joined pair `(t, s)`:
- numeric: both NaN is a match; exactly one NaN is a mismatch; otherwise a mismatch if $|t - s| > \text{atol}$.
- string: both missing (`None`/NaN) is a match; exactly one missing is a mismatch; otherwise a mismatch if `t != s`.

## Output
A DataFrame with columns, in this order:

| column | dtype | value |
|---|---|---|
| `feature` | str | the feature name |
| `compared` | int | number of joined requests |
| `mismatches` | int | number of mismatching pairs |
| `rate` | float | `mismatches / compared`, or `0.0` if `compared == 0` |
| `max_abs_diff` | float | numeric features: largest $\lvert t - s\rvert$ over pairs where both are non-NaN; `NaN` if there are none, and always `NaN` for string features |

Rows sorted by `rate` descending, ties by `feature` ascending. The index is ignored.
"""

STARTER = '''import pandas as pd

def skew_report(train_log, serve_log, features, atol=1e-6):
    """Return feature, compared, mismatches, rate, max_abs_diff (one row per feature)."""
    j = train_log.merge(serve_log, on='request_id', suffixes=('_t', '_s'))
    rows = []
    for f in features:
        bad = int((j[f + '_t'] != j[f + '_s']).sum())        # NaN != NaN, and no tolerance
        rows.append([f, len(j), bad, bad / len(j), float('nan')])
    return pd.DataFrame(rows, columns=['feature', 'compared', 'mismatches', 'rate', 'max_abs_diff'])
'''

SOLUTION = '''import numpy as np
import pandas as pd

def skew_report(train_log, serve_log, features, atol=1e-6):
    t = train_log[['request_id'] + features]
    s = serve_log[['request_id'] + features]
    j = t.merge(s, on='request_id', suffixes=('_t', '_s'))   # inner join: only requests in both logs
    n = len(j)
    rows = []
    for f in features:
        a, b = j[f + '_t'], j[f + '_s']
        both_missing, one_missing = a.isna() & b.isna(), a.isna() ^ b.isna()
        if pd.api.types.is_numeric_dtype(train_log[f]):
            diff = (a.astype(float) - b.astype(float)).abs()  # NaN wherever either side is NaN
            bad = one_missing | (diff > atol)
            max_diff = float(diff.max()) if diff.notna().any() else float('nan')
        else:
            bad = one_missing | (~both_missing & ~one_missing & (a != b))
            max_diff = float('nan')
        m = int(bad.sum())
        rows.append([f, n, m, m / n if n else 0.0, max_diff])
    out = pd.DataFrame(rows, columns=['feature', 'compared', 'mismatches', 'rate', 'max_abs_diff'])
    out = out.astype({'compared': 'int64', 'mismatches': 'int64', 'rate': 'float64', 'max_abs_diff': 'float64'})
    return out.sort_values(['rate', 'feature'], ascending=[False, True]).reset_index(drop=True)
'''

HINTS = [
    'Merge the two logs on `request_id` with `suffixes=("_t", "_s")`. The default inner join keeps only requests logged on both sides, and the suffixes give you `f + "_t"` and `f + "_s"` for each feature.',
    '`a != b` is `True` when both sides are NaN, and it ignores the tolerance. Build the masks explicitly: `both_missing = a.isna() & b.isna()`, `one_missing = a.isna() ^ b.isna()`, and for numeric features `diff = (a - b).abs()`, which is NaN wherever either side is missing.',
    'A numeric mismatch is `one_missing | (diff > atol)` (comparisons with NaN are `False`). Collect one list per feature, build the DataFrame, then `sort_values(["rate", "feature"], ascending=[False, True])`.',
]

def _logs(train_rows, serve_rows, cols):
    return pd.DataFrame(train_rows, columns=cols), pd.DataFrame(serve_rows, columns=cols)

_c = ['request_id', 'age', 'country', 'ctr_7d']
_t1, _s1 = _logs([[1, 30.0, 'US', 0.25], [2, 41.0, 'DE', 0.5], [3, 22.0, 'IN', 0.125]],
                 [[1, 30.0, 'US', 0.25], [2, 41.0, 'de', 0.5], [3, 22.0, 'IN', 0.375]], _c)
_t2, _s2 = _logs([[10, 1.0, None, 0.5], [11, None, 'US', 0.5], [12, 3.0, 'US', None], [13, 4.0, 'IN', 0.5]],
                 [[10, 1.0, None, 0.5], [11, None, None, 0.5], [12, 3.0000001, 'US', 0.75], [99, 0.0, 'US', 0.0]], _c)

_rng = __import__('numpy').random.default_rng(5)
_ids = _rng.permutation(200) + 1000
_x = _rng.integers(0, 1000, 200) / 8.0
_tr = pd.DataFrame({'request_id': _ids, 'x': _x, 'bucket': [f'b{k % 4}' for k in range(200)], 'model': 'v3'})
_sv = pd.DataFrame({'request_id': _ids[::-1], 'x': _x[::-1] + (_rng.integers(0, 10, 200) == 0) * 0.5,
                    'bucket': [f'b{k % 4}' for k in range(200)][::-1], 'model': 'v3'})

TESTS = [
    (_t1, _s1, ['age', 'country', 'ctr_7d']),
    (_t2, _s2, ['age', 'country', 'ctr_7d']),
    (_t2, _s2, ['age', 'country', 'ctr_7d'], 1e-9),
    (_t1, _s1.assign(request_id=[7, 8, 9]), ['ctr_7d', 'age']),
    (_t1, _s1, ['ctr_7d']),
    (pd.DataFrame({'request_id': [1, 2], 'x': [float('nan'), float('nan')]}),
     pd.DataFrame({'request_id': [2, 1], 'x': [float('nan'), float('nan')]}), ['x']),
    (_tr, _sv, ['x', 'bucket']),
]

EXPLANATION = r"""
## Why it works
Joining on `request_id` lines up the two computations of the *same* feature for the *same* request, so any disagreement is skew in the feature code or data, not a difference between populations (that's drift, a different problem). Requests logged on only one side can't be compared and are left out of `compared` (test 4).

The three masks make the missing-value rules explicit. `diff = |t - s|` is NaN whenever either side is NaN, and NaN comparisons are always `False`, so `one_missing | (diff > atol)` counts exactly the right pairs. Sorting by rate puts the worst feature first, which is what an on-call engineer reads.

## Complexity
One hash join, $O(n_t + n_s)$, then $O(n \cdot F)$ vectorised comparisons for $F$ features.

## Common bugs
- `a != b` on floats: NaN against NaN counts as a mismatch (test 6) and tiny float differences from a different library version count too.
- Comparing rows by position instead of joining on `request_id`: logs arrive in different orders (test 7).
- An outer join, which turns every request missing from one log into a "mismatch" on every feature.
- A string comparison that treats `None` vs `None` as a mismatch, or misses case differences such as `'DE'` vs `'de'` (test 1), which is a classic skew source.

## Interview follow-ups
- The serving log is sampled at 1%. How does that change the confidence you have in a mismatch rate of 0.3%? (Treat it as a proportion estimate with a confidence interval; you may need days of logs.)
- What design removes this class of bug instead of detecting it? (One feature definition shared by both paths, for example a feature store that serves training data and online lookups from the same transformation code, or logging served features and training on the logs.)
"""
