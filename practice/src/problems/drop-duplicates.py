import numpy as np
import pandas as pd

META = dict(title='Keep the latest event per user', tags=['Data Processing'], diff='Medium', runtime='pandas',
            lessons=['air-fanout-v1:10.03'], related=['top-n-per-group', 'replace-values', 'concatenate-dataframes'])

STATEMENT = r"""
An event log was assembled from several sources, so the same fact shows up more than once and user ids are typed inconsistently (`"Ann"`, `" ann"`, `"ANN "`).

`df` has columns `user` (str), `event` (str), `ts` (int, a timestamp) and `value` (float). Write `latest_events(df)` that:

1. normalises `user` by stripping leading/trailing whitespace and lower-casing it;
2. keeps **one row per `(user, event)` pair**: the row with the largest `ts`. If several rows tie on that largest `ts`, keep the one that appears **last** in `df`;
3. returns the surviving rows with columns `user, event, ts, value` (with `user` normalised), in the order they appear in `df`.

`event` is compared exactly as given. Don't modify `df`.

## Example
```
   user  event  ts  value
0   Ann  click   5    1.0
1  ann   click   9    2.0       ->    user  event  ts  value
2   bob  view    3    0.5             ann   click   9    2.0
3  BOB   view    3    0.7             bob   view    3    0.7
```
"""

STARTER = '''import pandas as pd

def latest_events(df):
    """Normalise user, keep the latest row per (user, event) (ties: last in df), original order."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def latest_events(df):
    out = df.assign(user=df['user'].str.strip().str.lower())
    # A stable sort by ts keeps equal timestamps in their original order, so keep="last"
    # picks the largest ts and, among ties, the row that came last in df.
    out = out.sort_values('ts', kind='stable').drop_duplicates(['user', 'event'], keep='last')
    return out.sort_index()[['user', 'event', 'ts', 'value']]
'''

HINTS = [
    'Normalise first: `df["user"].str.strip().str.lower()`. Otherwise `"Ann"` and `" ann"` count as different users.',
    '`drop_duplicates(subset=[...], keep="last")` keeps the last row of each key **in the current row order**. Arrange the rows so that the one you want comes last.',
    'Sort by `ts` with `kind="stable"` (ties keep their original order), drop duplicates on `["user", "event"]` with `keep="last"`, then `sort_index()` to restore the original order.',
]

_r = np.random.default_rng(64)
_n = 40
_log = pd.DataFrame({'user': _r.choice(['ann', ' Ann', 'BOB', 'bob ', 'cy'], _n), 'event': _r.choice(['click', 'view', 'buy'], _n),
                     'ts': _r.integers(0, 8, _n), 'value': _r.normal(size=_n).round(3)})


def _df(rows):
    return pd.DataFrame(rows, columns=['user', 'event', 'ts', 'value']).astype({'value': float})


TESTS = [
    (_df([['Ann', 'click', 5, 1.0], ['ann ', 'click', 9, 2.0], ['bob', 'view', 3, 0.5], ['BOB', 'view', 3, 0.7]]),),
    (_df([['u', 'a', 1, 0.0], ['u', 'b', 1, 0.0], ['u', 'a', 0, 9.0]]),),
    (_df([['solo', 'x', 42, 3.14]]),),
    (_df([['x', 'e', 7, 1.0], ['X', 'e', 7, 2.0], [' x ', 'e', 7, 3.0]]),),
    (_df([['a', 'Buy', 1, 1.0], ['a', 'buy', 1, 2.0], ['b', 'buy', -5, 3.0], ['b', 'buy', -2, 4.0]]),),
    (_df([['p', 'e', 3, 1.0], ['q', 'e', 1, 2.0], ['p', 'e', 1, 3.0], ['q', 'e', 4, 4.0], ['p', 'e', 3, 5.0]]),),
    (_log,),
]

EXPLANATION = r"""
## Why it works
`drop_duplicates(keep="last")` is order-dependent by design: for each key it keeps whichever row comes last **in the frame as it is now**. So the job is to put the right row last. A stable sort by `ts` does that: within a key, larger timestamps move later, and rows with equal timestamps keep their original relative order, so among ties the one that appeared last in `df` stays last. `sort_index()` then undoes the sort, because the labels still hold the original positions.

Normalising first matters because duplicates are only duplicates under the right equality. Raw strings make `"Ann"` and `"ann "` different keys.

## Complexity
$O(N \log N)$ for the sort; `drop_duplicates` hashes the key columns in $O(N)$.

## Common bugs
- `drop_duplicates()` without `subset` only removes rows identical in **every** column, which these rarely are.
- `keep="last"` without sorting keeps the last row in file order, not the latest timestamp (test 2).
- An unstable sort makes the tie rule nondeterministic (tests 4 and 6).
- `groupby(["user", "event"])["ts"].idxmax()` returns the **first** row among ties, the opposite of the rule here.
- Normalising `event` too: the statement compares it exactly, so `"Buy"` and `"buy"` stay separate (test 5).

## Interview follow-ups
- Do it with `groupby(...).tail(1)` after the same sort, and explain why it gives the same answer.
- The log has 5B rows across files. How would you deduplicate without one global sort? (Hash-partition by key, then keep the max `(ts, file_order)` per partition.)
"""
