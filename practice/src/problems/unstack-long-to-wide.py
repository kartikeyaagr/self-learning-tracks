import numpy as np
import pandas as pd

META = dict(title='Training-log metrics from long to wide', tags=['Data Processing', 'Training', 'MLOps'], diff='Hard',
            runtime='pandas', lessons=['air-fanout-v1:11.06'], related=['melt-wide-to-long', 'pivot-tables', 'drop-duplicates'])

STATEMENT = r"""
An experiment tracker stores metrics in long form, one row per logged value. `df` has columns `run` (str), `step` (int), `metric` (str) and `value` (float). Training loss is logged every step, evaluation metrics only now and then, and a retry can log the same value twice.

Write `metrics_wide(df)` that builds the table you would plot:

1. If a `(run, step, metric)` triple appears more than once, keep the value from the row that appears **last** in `df`.
2. Make one row per `(run, step)` pair that appears in `df`, with columns `run`, `step`, then one column per metric name, metric columns sorted alphabetically.
3. A metric not logged at a step takes its most recent earlier value **from the same run** (forward fill along steps). Before a run first logs a metric, the cell is NaN.
4. Sort rows by `run` ascending, then `step` ascending.

## Example
```
  run  step  metric    value
0  r1     1  loss       2.0            run  step  acc  loss
1  r1     2  loss       1.5     ->     r1     1  NaN   2.0
2  r1     2  acc        0.6            r1     2  0.6   1.5
3  r1     3  loss       1.2            r1     3  0.6   1.2
```
`df` has at least one row; no metric is named `run` or `step`.
"""

STARTER = '''import pandas as pd

def metrics_wide(df):
    """Long (run, step, metric, value) -> wide: run, step, metrics A-Z; last duplicate wins; ffill within run."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def metrics_wide(df):
    # unstack needs unique (run, step, metric) keys, so resolve duplicates first.
    last = df.drop_duplicates(['run', 'step', 'metric'], keep='last')
    # Move the innermost index level (metric) into the columns. unstack sorts rows and columns.
    wide = last.set_index(['run', 'step', 'metric'])['value'].unstack('metric')
    # Forward fill along steps, restarting at each run so values never leak between runs.
    wide = wide.groupby(level='run').ffill()
    wide.columns.name = None
    return wide.reset_index()
'''

HINTS = [
    'With a three-level index `(run, step, metric)`, `unstack("metric")` turns the metric level into columns. It fails if a key appears twice, so `drop_duplicates(..., keep="last")` first.',
    '`unstack` returns rows sorted by `(run, step)` and metric columns sorted by name, with NaN where nothing was logged.',
    'Plain `ffill()` would carry the last value of run `r1` into the first rows of run `r2`. Use `groupby(level="run").ffill()`, then clear `columns.name` and `reset_index()`.',
]

_r = np.random.default_rng(808)
_rows = []
for _run in ['b', 'a', 'c']:
    for _s in range(1, 13):
        _rows.append([_run, _s, 'loss', round(float(3 / _s + _r.normal(0, 0.05)), 4)])
        if _s % 4 == 0:
            _rows.append([_run, _s, 'val_acc', round(float(0.5 + 0.03 * _s + _r.normal(0, 0.01)), 4)])
        if _r.random() < 0.15:
            _rows.append([_run, _s, 'loss', round(float(3 / _s), 4)])     # a retried log line
_rows = [_rows[i] for i in _r.permutation(len(_rows))]


def _df(rows):
    return pd.DataFrame(rows, columns=['run', 'step', 'metric', 'value']).astype({'value': float})


TESTS = [
    (_df([['r1', 1, 'loss', 2.0], ['r1', 2, 'loss', 1.5], ['r1', 2, 'acc', 0.6], ['r1', 3, 'loss', 1.2]]),),
    (_df([['r2', 1, 'acc', 0.1], ['r1', 5, 'acc', 0.9], ['r2', 2, 'loss', 3.0], ['r1', 1, 'loss', 4.0]]),),
    (_df([['r', 10, 'loss', 1.0], ['r', 10, 'loss', 0.5], ['r', 20, 'loss', 0.25], ['r', 10, 'loss', 0.75]]),),
    (_df([['only', 0, 'm', -1e6]]),),
    (_df([['a', 2, 'z', 1.0], ['a', 1, 'y', 2.0], ['a', 3, 'x', 3.0], ['b', 1, 'x', 4.0]]),),
    (_df(_rows),),
]

EXPLANATION = r"""
## Why it works
Long-to-wide is "move one index level into the columns". Put all key columns in a MultiIndex, then `unstack` the level you want as columns; every remaining `(run, step)` pair becomes a row, and missing combinations become NaN. Two properties make the rest simple: `unstack` requires unique keys (hence the explicit "last write wins" step before it), and it sorts both the row keys and the new columns.

Forward filling is a per-run operation. `groupby(level="run").ffill()` fills down within each run's block of rows, so a run that hasn't logged `acc` yet keeps NaN rather than inheriting the previous run's last value (test 2).

## Complexity
$O(N)$ to deduplicate and reshape, $O(N \log N)$ for the sort inside `unstack`, and $O(R \cdot M)$ for the dense $R \times M$ result.

## Common bugs
- `df.pivot(index=["run", "step"], columns="metric", values="value")` raises "Index contains duplicate entries" on retried logs (test 3).
- `pivot_table(..., aggfunc="mean")` hides the duplicates by averaging them, which is a different rule from "last wins".
- `keep="first"` instead of `keep="last"` (test 3 expects 0.75 at step 10).
- Ungrouped `ffill()` leaks values across runs.
- Leaving `columns.name = "metric"`: harmless for the values, but `reset_index()` output then prints a stray axis label and some serialisers keep it.

## Interview follow-ups
- Steps differ between runs (one logs every 10 steps, another every 7). How would you put both on a common step grid before comparing them? (`reindex` each run on the union of steps, then `ffill` or interpolate.)
- The tracker holds 50M rows. Why is keeping the long format in storage and pivoting only the slice you plot the usual design?
"""
