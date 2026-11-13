import numpy as np
import pandas as pd

META = dict(title='Windowed accuracy monitor with delayed labels', tags=['MLOps', 'Time Series', 'Metrics & Evaluation'], diff='Hard', runtime='pandas',
            lessons=['sdsd:17.04', 'ieng:24.01'], related=['consecutive-breach-alerts', 'expected-calibration-error'])

STATEMENT = r"""
Ground truth arrives late: a fraud label after a chargeback, a churn label a month later. A production accuracy dashboard must only use labels that had **actually arrived** by the time it's computed, report how much of the window is labelled, and not alert on a handful of labels.

Write `accuracy_monitor(events, check_times, window, min_labels, floor)` that returns one row per (check time, model).

## Inputs
- `events`: DataFrame with columns `model` (str), `pred_ts` (int, when the prediction was made), `label_ts` (float: when its label arrived, `NaN` if it hasn't and never will) and `correct` (int 0/1; meaningless while unlabelled).
- `check_times`: list of ints, possibly unsorted, possibly with repeats. Keep their order.
- `window` (int > 0), `min_labels` (int >= 1), `floor` (float).

## For a check time $T$ and a model
- **window events**: that model's events with $T - \text{window} < $ `pred_ts` $\le T$.
- **labelled**: window events with `label_ts <= T`.

| column | dtype | value |
|---|---|---|
| `check_ts` | int | $T$ |
| `model` | str | the model |
| `n_preds` | int | number of window events |
| `n_labeled` | int | number of labelled window events |
| `coverage` | float | `n_labeled / n_preds`, `NaN` if `n_preds == 0` |
| `accuracy` | float | mean of `correct` over labelled events, `NaN` if `n_labeled == 0` |
| `alert` | bool | `n_labeled >= min_labels` and `accuracy < floor` |

## Output
Columns in the order above. One row for **every** model appearing anywhere in `events` at every check time, even with no events in the window. Rows ordered by position in `check_times`, then by `model` ascending. The index is ignored.
"""

STARTER = '''import pandas as pd

def accuracy_monitor(events, check_times, window, min_labels, floor):
    """One row per (check time, model): check_ts, model, n_preds, n_labeled, coverage, accuracy, alert."""
    rows = []
    for T in check_times:
        w = events[(events['pred_ts'] > T - window) & (events['pred_ts'] <= T)]
        for m, g in w.groupby('model'):                       # labels from the future, missing models
            acc = g['correct'].mean()
            rows.append([T, m, len(g), len(g), 1.0, acc, acc < floor])
    return pd.DataFrame(rows, columns=['check_ts', 'model', 'n_preds', 'n_labeled', 'coverage', 'accuracy', 'alert'])
'''

SOLUTION = '''import numpy as np
import pandas as pd

COLS = ['check_ts', 'model', 'n_preds', 'n_labeled', 'coverage', 'accuracy', 'alert']

def accuracy_monitor(events, check_times, window, min_labels, floor):
    models = sorted(events['model'].unique())
    parts = []
    for T in check_times:
        w = events[(events['pred_ts'] > T - window) & (events['pred_ts'] <= T)]
        labeled = w['label_ts'] <= T                            # NaN compares False: never labelled
        g = (w.assign(lab=labeled.astype(int), hit=(labeled & (w['correct'] == 1)).astype(int))
              .groupby('model')[['lab', 'hit']].agg(['size', 'sum']))
        # reindex: models with no events in this window still get a row (zeros)
        n_preds = g[('lab', 'size')].reindex(models, fill_value=0).to_numpy()
        n_lab = g[('lab', 'sum')].reindex(models, fill_value=0).to_numpy()
        hits = g[('hit', 'sum')].reindex(models, fill_value=0).to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            coverage = np.where(n_preds > 0, n_lab / n_preds, np.nan)
            accuracy = np.where(n_lab > 0, hits / n_lab, np.nan)
        parts.append(pd.DataFrame({
            'check_ts': T, 'model': models, 'n_preds': n_preds, 'n_labeled': n_lab,
            'coverage': coverage, 'accuracy': accuracy,
            'alert': (n_lab >= min_labels) & (np.nan_to_num(accuracy, nan=np.inf) < floor),
        }))
    if not parts:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(parts, ignore_index=True)[COLS]
    return out.astype({'check_ts': 'int64', 'n_preds': 'int64', 'n_labeled': 'int64',
                       'coverage': 'float64', 'accuracy': 'float64', 'alert': 'bool'})
'''

HINTS = [
    'Handle one check time at a time. The window is `(T - window, T]` on `pred_ts`; inside it, an event is labelled only if `label_ts <= T`. Comparisons with NaN are `False`, so never-labelled events drop out by themselves.',
    "`groupby('model')` only produces models that have events in the window. Compute the model list once from all of `events` (sorted) and `reindex(models, fill_value=0)` each count, so every model gets a row at every check time.",
    'Count `n_labeled` and correct-and-labelled events, then derive coverage and accuracy with `np.where` to get NaN on zero denominators. The alert needs both conditions; make sure a NaN accuracy never alerts.',
]

def _ev(rows):
    return pd.DataFrame(rows, columns=['model', 'pred_ts', 'label_ts', 'correct'])

_nan = float('nan')
_rng = np.random.default_rng(8)
_k = 300
_big = _ev([[str(m), int(t), float(t + d) if d < 90 else _nan, int(c)]
            for m, t, d, c in zip(_rng.choice(['champion', 'challenger'], _k), _rng.integers(0, 1000, _k),
                                  _rng.integers(0, 120, _k), _rng.uniform(0, 1, _k) < 0.8)])
_big.loc[(_big['model'] == 'challenger') & (_big['pred_ts'] > 600), 'correct'] = 0   # challenger breaks at t=600

TESTS = [
    (_ev([['m1', 1, 2.0, 1], ['m1', 3, 4.0, 0], ['m1', 5, 20.0, 1]]), [5, 25], 10, 1, 0.6),
    (_ev([['a', 10, 11.0, 1], ['b', 10, _nan, 0], ['b', 12, 13.0, 0]]), [12, 9], 5, 1, 0.5),
    (_ev([['a', 0, 0.0, 0], ['a', 1, 1.0, 0]]), [1, 1, 100], 2, 3, 0.9),
    (_ev([['a', 5, 5.0, 1], ['a', 10, 10.0, 0]]), [10], 5, 1, 0.5),
    (_ev([['z', 1, 1.0, 1], ['y', 2, 2.0, 0], ['x', 3, 3.0, 1]]), [], 10, 1, 0.5),
    (_ev([['a', 1, _nan, 1], ['a', 2, _nan, 0]]), [2], 10, 1, 0.99),
    (_big, [300, 600, 700, 800, 1000], 200, 20, 0.6),
    (_big, [1000, 750], 100, 5, 0.75),
]

EXPLANATION = r"""
## Why it works
At time $T$ the dashboard could only have seen labels with `label_ts <= T`, so filtering on that reproduces what a live system would have shown. Without it, a backfilled dashboard looks better or worse than reality and an alert rule tuned on it fires at the wrong times. Coverage makes the delay visible: an accuracy of 0.95 on 3% coverage means little, especially because early labels are often a biased sample (fast chargebacks are a particular kind of fraud).

Reindexing every count to the full model list turns "no data" into an explicit row of zeros and NaNs instead of a silently missing row, which is how monitoring systems distinguish "healthy" from "not reporting". The alert needs `min_labels` so a single wrong label doesn't page anyone.

## Complexity
$O(|C| \cdot n)$ for $|C|$ check times over $n$ events with the per-check filter. For many check times, sort by `pred_ts` and use `searchsorted` to find each window, or bucket events into fixed intervals and pre-aggregate.

## Common bugs
- Using `correct` for events whose label hasn't arrived yet (the starter): the dashboard "knows" the future.
- Dropping models without events in a window, so a model that stopped serving traffic looks fine (test 2).
- Window boundaries: `pred_ts == T - window` is outside, `pred_ts == T` is inside (test 4).
- An alert that fires on `NaN < floor` evaluated as truthy somewhere, or on coverage too low to mean anything.

## Interview follow-ups
- Labels take two weeks. What can you monitor in the meantime that correlates with accuracy? (Prediction score distribution, input drift such as PSI, confidence, proxy labels like early user feedback.)
- How would you correct the bias that early-arriving labels are not representative? (Compare against the delay distribution from history, report accuracy per label-delay bucket, or reweight by the probability of a label having arrived.)
"""
