import numpy as np
import pandas as pd

META = dict(title='Impute sensor gaps by station', tags=['Data Processing', 'Time Series'], diff='Medium',
            runtime='pandas', lessons=['air-fanout-v1:11.12'], related=['inspect-dataframe-shape', 'replace-values', 'groupby-basics'])

STATEMENT = r"""
Weather stations report `temp` and `humidity`, but readings drop out. `df` has columns `station` (str), `temp` (float) and `humidity` (float), one row per report in time order; missing readings are NaN.

Write `impute_readings(df)` that applies these rules:

1. **Drop** the rows where `temp` and `humidity` are **both** missing.
2. Add a column `temp_imputed` (bool) that is True where `temp` was missing in a kept row.
3. Fill a missing `temp` with the **mean of that station's observed temps**. If the station has no observed temp at all, use the mean of every observed temp in `df`. If there is none anywhere, leave it NaN.
4. Fill a missing `humidity` with the **most recent earlier** humidity reported by the **same station** (forward fill within the station, in row order). If the station has no earlier humidity, leave it NaN.

Return a DataFrame with columns `station, temp, humidity, temp_imputed`, holding the kept rows in their original order. Don't modify `df`.

## Example
```
  station  temp  humidity
0       A  10.0      50.0
1       A   NaN       NaN
2       B  20.0       NaN
3       A   NaN      55.0
```
gives

```
  station  temp  humidity  temp_imputed
0       A  10.0      50.0         False
2       B  20.0       NaN         False
3       A  10.0      55.0          True
```
Row 1 is dropped. Row 3's temp is station A's observed mean; B's humidity stays NaN because B has no earlier reading. (The index is not checked; only columns and row order are.)
"""

STARTER = '''import numpy as np
import pandas as pd

def impute_readings(df):
    """Drop all-missing rows, flag and fill temp (station mean, then global mean), ffill humidity per station."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
import pandas as pd

def impute_readings(df):
    out = df.dropna(subset=['temp', 'humidity'], how='all').copy()
    out['temp_imputed'] = out['temp'].isna()
    # Compute both means from observed values only, before any filling.
    station_mean = out.groupby('station')['temp'].transform('mean')
    global_mean = out['temp'].mean()
    out['temp'] = out['temp'].fillna(station_mean).fillna(global_mean)
    # groupby().ffill() fills forward inside each station and never across stations.
    out['humidity'] = out.groupby('station')['humidity'].ffill()
    return out[['station', 'temp', 'humidity', 'temp_imputed']]
'''

HINTS = [
    '`dropna(subset=[...], how="all")` drops a row only when every listed column is missing. Record the `temp` mask before you fill anything.',
    '`groupby("station")["temp"].transform("mean")` gives every row its station\'s mean (NaNs are skipped), aligned with the original rows, so it plugs straight into `fillna`.',
    'Chain `.fillna(station_mean).fillna(global_mean)`, with the global mean computed before filling. For humidity, `groupby("station")["humidity"].ffill()` stays within each station.',
]

_r = np.random.default_rng(77)
_n = 30
_raw = pd.DataFrame({'station': _r.choice(['north', 'south', 'east'], _n),
                     'temp': _r.normal(15, 5, _n).round(1), 'humidity': _r.uniform(30, 90, _n).round(1)})
_raw = _raw.mask(pd.DataFrame({'station': False, 'temp': _r.random(_n) < 0.3, 'humidity': _r.random(_n) < 0.3}))


def _df(rows):
    return pd.DataFrame(rows, columns=['station', 'temp', 'humidity']).astype({'temp': float, 'humidity': float})


N = np.nan
TESTS = [
    (_df([['A', 10.0, 50.0], ['A', N, N], ['B', 20.0, N], ['A', N, 55.0]]),),
    (_df([['A', 1.0, 40.0], ['A', N, N], ['A', 3.0, N], ['A', N, 41.0], ['A', N, N]]),),
    (_df([['A', 10.0, 50.0], ['B', N, 60.0], ['B', N, N], ['B', N, 61.0]]),),
    (_df([['X', N, 1.0], ['Y', N, N], ['X', N, N]]),),
    (_df([['A', N, N]]),),
    (_df([['A', -40.0, 10.0], ['B', 45.0, N], ['A', N, N], ['B', N, 20.0], ['A', N, 15.0], ['B', 44.0, N]]),),
    (_df([['A', 10.0, 1.0], ['A', N, 2.0], ['A', N, 3.0], ['B', 30.0, 4.0], ['C', N, 5.0]]),),
    (_raw,),
]

EXPLANATION = r"""
## Why it works
Each rule maps onto one pandas primitive:

| rule | tool |
|---|---|
| drop rows missing everything | `dropna(subset=..., how="all")` |
| per-group fill value, aligned to rows | `groupby(...).transform("mean")` |
| fallback value | a second `fillna` |
| carry the last observation forward within a group | `groupby(...).ffill()` |

`transform` is the key idea: unlike `agg`, it returns one value **per original row** (the group's mean, repeated), with the original index, so `fillna` can use it directly. Means skip NaN by default, so they are computed from observed values only, as long as you compute them **before** filling.

## Complexity
$O(N)$ for the group means and the forward fill (grouping uses hashing); one pass for each rule.

## Common bugs
- `dropna()` with no arguments drops any row with **one** missing value, losing half the data.
- Flagging `temp_imputed` after filling: everything is False by then.
- Computing the global mean after the station fill, so imputed values feed back into it. In test 7, station C must get $(10 + 30)/2 = 20$, not $(10+10+10+30)/4 = 15$.
- `df["humidity"].ffill()` without `groupby` leaks a reading from station A into station B (tests 1 and 6).
- `bfill`, or `interpolate`, uses **future** readings. That is a data leak if the cleaned table feeds a forecasting model.

## Interview follow-ups
- Readings are hourly and a station was offline for a week. Should forward fill be unlimited? (Use `ffill(limit=k)` and leave long gaps missing, possibly with a flag.)
- When is mean imputation harmful for a downstream model, and what does adding the `*_imputed` indicator column buy you?
"""
