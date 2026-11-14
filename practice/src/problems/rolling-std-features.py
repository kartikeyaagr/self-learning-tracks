import pandas as pd

META = dict(title='Rolling mean and standard deviation without leakage', tags=['Time Series', 'Feature Engineering'], diff='Medium', runtime='pandas',
            related=['lag-features', 'exponential-moving-average', 'single-pass-mean-and-variance'])

STATEMENT = r"""
`df` has columns `id`, `t`, `value`. Write `rolling_features(df, window, min_periods)` that returns a copy sorted by `id`, `t` with a fresh index and two new columns:

- `roll_mean`: mean of the previous `window` values for the same id, **excluding the current row**;
- `roll_std`: sample standard deviation (ddof=1) of the same values.

If fewer than `min_periods` previous values exist, the result is NaN. Columns: `id, t, value, roll_mean, roll_std`.
"""

STARTER = '''import pandas as pd

def rolling_features(df, window, min_periods):
    """Return df sorted with leakage-free rolling mean and std."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def rolling_features(df, window, min_periods):
    out = df.sort_values(["id", "t"]).reset_index(drop=True)[["id", "t", "value"]].copy()
    prev = out.groupby("id")["value"].shift(1)
    grp = prev.groupby(out["id"])
    out["roll_mean"] = grp.transform(lambda s: s.rolling(window, min_periods=min_periods).mean()).astype(float)
    out["roll_std"] = grp.transform(lambda s: s.rolling(window, min_periods=min_periods).std()).astype(float)
    return out
'''

HINTS = [
    'Shift by one within each id first; then a normal rolling window over the shifted series never includes the current value.',
    'Group the shifted series by id again before rolling, so windows do not span two entities.',
    'pandas `rolling(...).std()` uses ddof=1 by default, and it needs at least two values to be finite.',
]

def _df(rows):
    return pd.DataFrame(rows, columns=['id', 't', 'value'])

TESTS = [
    (_df([['a', i, float(v)] for i, v in enumerate([1, 2, 3, 4, 5])]), 3, 2),
    (_df([['a', i, float(v)] for i, v in enumerate([5, 5, 5, 9])]), 2, 1),
    (_df([['b', 2, 4.0], ['a', 1, 1.0], ['b', 1, 2.0], ['a', 2, 3.0], ['a', 3, 6.0], ['b', 3, 8.0]]), 2, 2),
    (_df([['x', 1, 10.0]]), 3, 1),
    (_df([['s', i, float(v)] for i, v in enumerate([2, 4, 4, 4, 5, 5, 7, 9])]), 4, 3),
]

EXPLANATION = r"""
## The leakage trap
`rolling(window).mean()` on the raw series includes the current value. If the current value is the target (or strongly tied to it), the feature contains the answer and offline metrics look far better than production. Shifting by one first makes the feature use only the past.

## Why a rolling standard deviation
Volatility features help with demand, fraud and anomaly detection: a transaction of 500 means something different for an account whose amounts usually vary by 10 than for one that varies by 1,000.

## Edge behaviour
The standard deviation of a single value with ddof=1 is undefined (NaN), even when `min_periods` is 1.

## Interview follow-ups
- How would you compute the same feature incrementally in a stream? (Keep a ring buffer and running sums, or use Welford's algorithm.)
"""
