import pandas as pd

META = dict(title='Lag and difference features per entity', tags=['Time Series', 'Feature Engineering'], diff='Easy', runtime='pandas',
            related=['rolling-std-features', 'point-in-time-feature-join', 'time-split-with-gap'])

STATEMENT = r"""
`df` has columns `id`, `t` (an integer time step) and `value`, possibly unsorted. Write `add_lags(df, lags)` that returns a new DataFrame sorted by `id` then `t`, with a fresh 0..n-1 index, and for every `k` in `lags`:

- `lag_k`: the `value` from `k` rows earlier **for the same id** (in time order), or NaN if there is none;
- `diff_k`: `value - lag_k`.

Columns: `id, t, value`, then `lag_k, diff_k` for each `k` in the order given.
"""

STARTER = '''import pandas as pd

def add_lags(df, lags):
    """Return a sorted copy of df with lag_k and diff_k columns."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def add_lags(df, lags):
    out = df.sort_values(["id", "t"]).reset_index(drop=True)[["id", "t", "value"]].copy()
    g = out.groupby("id")["value"]
    for k in lags:
        out[f"lag_{k}"] = g.shift(k).astype(float)
        out[f"diff_{k}"] = (out["value"] - out[f"lag_{k}"]).astype(float)
    return out
'''

HINTS = [
    'Sort by id and time first; a lag only makes sense in time order.',
    '`df.groupby("id")["value"].shift(k)` shifts within each id, so values never leak from one entity to the next.',
    'The difference is just value minus the lag column; NaN propagates automatically.',
]

def _df(rows):
    return pd.DataFrame(rows, columns=['id', 't', 'value'])

TESTS = [
    (_df([['a', 1, 10.0], ['a', 2, 12.0], ['a', 3, 15.0]]), [1]),
    (_df([['a', 2, 5.0], ['b', 1, 1.0], ['a', 1, 3.0], ['b', 2, 4.0]]), [1]),
    (_df([['x', 1, 1.0], ['x', 2, 2.0], ['x', 3, 4.0], ['x', 4, 8.0]]), [1, 2]),
    (_df([['u', 5, 7.0]]), [1, 3]),
    (_df([['p', 3, 3.0], ['q', 1, 10.0], ['p', 1, 1.0], ['q', 2, 20.0], ['p', 2, 2.0], ['q', 3, 30.0]]), [2]),
]

EXPLANATION = r"""
## Why per entity
Lag features describe an entity's own recent history: yesterday's sales for this store, the previous reading of this sensor. Shifting the whole column without grouping would hand store B the last value of store A.

## Rows versus time
`shift(k)` looks back $k$ rows, which equals $k$ time steps only if there are no gaps. With irregular timestamps, reindex each entity onto a complete time grid first, or join on `t - k` explicitly.

## Leakage
A lag of 0 is the target itself. Make sure every lag you use would be available at prediction time, including any delay before data arrives.

## Interview follow-ups
- How would you compute these features in a streaming system?
"""
