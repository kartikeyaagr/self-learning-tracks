import numpy as np
import pandas as pd

META = dict(title='Store league table within each region', tags=['Data Processing'], diff='Hard', runtime='pandas',
            lessons=['sdsd:10.06'], related=['groupby-basics', 'top-n-per-group', 'pivot-tables'])

STATEMENT = r"""
A retail chain logs revenue per store and month: `df` has columns `region` (str), `store` (str), `month` (int, 1-12) and `revenue` (float, $\ge 0$). A store belongs to one region; a `(store, month)` pair may appear in several rows (several tills), which are summed.

Write `store_report(df)` that returns one row per `(region, store)` with these columns, in this order:

| column | value |
|---|---|
| `region`, `store` | the keys |
| `total` | the store's total revenue |
| `share` | `total` divided by its region's total revenue |
| `rank` | 1 for the region's highest `total`, 2 for the next, ...; ties go to the alphabetically first `store` |
| `best_month` | the month with the store's largest **monthly** revenue (rows of the same month summed); ties go to the **earliest** month |

Sort the rows by `region` ascending, then `rank` ascending.

## Input
`df` has at least one row, and every region's total revenue is positive. Revenues in the tests are multiples of 0.5, so sums are exact and ties are real ties.
"""

STARTER = '''import pandas as pd

def store_report(df):
    """Per (region, store): total, share of region, rank in region, best_month. Sorted by region, rank."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def store_report(df):
    # One value per (region, store, month): a Series with a 3-level MultiIndex, sorted by its keys.
    monthly = df.groupby(['region', 'store', 'month'])['revenue'].sum()
    by_store = monthly.groupby(level=['region', 'store'])

    out = by_store.sum().to_frame('total')
    # transform broadcasts each region's sum back onto its stores.
    out['share'] = out['total'] / out.groupby(level='region')['total'].transform('sum')
    # idxmax returns the full (region, store, month) key of the first maximum; months are sorted,
    # so "first" means earliest.
    out['best_month'] = by_store.idxmax().map(lambda key: key[2])

    out = out.reset_index().sort_values(['region', 'total', 'store'], ascending=[True, False, True])
    out['rank'] = out.groupby('region').cumcount() + 1
    return out[['region', 'store', 'total', 'share', 'rank', 'best_month']].reset_index(drop=True)
'''

HINTS = [
    'Group by all three keys `["region", "store", "month"]` and sum first. The result is a Series with a three-level index, and every later step is a regrouping of it by `level=`.',
    'Regroup by `level=["region", "store"]` for `total` and `idxmax` (which returns the whole index tuple of the best month). Region totals come from grouping the totals by `level="region"` with `transform("sum")`, so they line up with each store.',
    'For `rank`, sort by `region` asc, `total` desc, `store` asc and number the rows within each region with `groupby("region").cumcount() + 1`.',
]

_r = np.random.default_rng(2024)
_stores = {'north': ['n1', 'n2', 'n3'], 'south': ['s1', 's2'], 'west': ['w1', 'w2', 'w3', 'w4']}
_rows = [[reg, st, int(_r.integers(1, 13)), float(_r.integers(0, 40)) * 0.5]
         for reg, sts in _stores.items() for st in sts for _ in range(int(_r.integers(2, 8)))]


def _df(rows):
    return pd.DataFrame(rows, columns=['region', 'store', 'month', 'revenue']).astype({'revenue': float})


TESTS = [
    (_df([['EU', 'paris', 1, 30.0], ['EU', 'paris', 2, 10.0], ['EU', 'rome', 1, 60.0], ['US', 'nyc', 3, 5.0]]),),
    (_df([['R', 'b', 5, 4.0], ['R', 'b', 5, 4.0], ['R', 'b', 6, 7.5], ['R', 'a', 2, 15.5]]),),
    (_df([['solo', 'only', 12, 0.5]]),),
    (_df([['R', 'x', 3, 2.0], ['R', 'x', 1, 2.0], ['R', 'y', 7, 4.0], ['R', 'z', 1, 0.0], ['R', 'z', 2, 0.0], ['R', 'z', 1, 1.0]]),),
    (_df([['B', 'k', 1, 1e6], ['A', 'k2', 1, 2.5], ['B', 'j', 1, 1e6], ['A', 'k1', 4, 2.5]]),),
    (_df(_rows),),
]

EXPLANATION = r"""
## Why it works
Everything here is a reduction at a different level of the same hierarchy region > store > month:

| quantity | grouping level |
|---|---|
| monthly revenue | region, store, month |
| `total`, `best_month` | region, store |
| region total for `share` | region |
| `rank` | ordering within region |

Summing once at the finest level gives a Series with a MultiIndex, and `groupby(level=...)` regroups it without touching the raw rows again. `transform("sum")` is the piece that makes `share` easy: it returns the region total **repeated for every store** of that region, aligned on the same index, so the division is element-wise.

`idxmax` returns the index label of the maximum, which for a MultiIndex is the whole `(region, store, month)` tuple; picking element 2 gives the month. Because `groupby` sorts keys, months within a store are ascending, and `idxmax` takes the first maximum, so ties go to the earliest month.

## Complexity
$O(N)$ for the first grouping, then $O(G \log G)$ to sort the $G$ store rows.

## Common bugs
- Taking `best_month` from the raw rows instead of monthly sums: in test 2, store `b` has rows 4.0 and 4.0 in May (8.0 total), which beats June's single 7.5.
- `rank(method="dense")` or `method="average"` on `total` gives tied stores the same rank; the statement wants distinct ranks broken by name (tests 4 and 5).
- Computing `share` against the chain's grand total instead of the region's.
- Calling `.reset_index()` too early and then trying to group "by level".
- `groupby(..., sort=False)` would break the "earliest month wins" tie rule for `idxmax`.

## Interview follow-ups
- Add each store's month-over-month growth for its best month. Which months need a `reindex` to handle months with no revenue?
- How would you compute the same report in one SQL query? (A `GROUP BY` subquery for monthly sums, `SUM(...) OVER (PARTITION BY region)` for share, and `ROW_NUMBER()` for both rank and best month.)
"""
