import numpy as np
import pandas as pd

META = dict(title='Quarterly revenue pivot with totals', tags=['Data Processing'], diff='Medium', runtime='pandas',
            lessons=['sdsd:10.07'], related=['groupby-basics', 'cross-tabulation', 'unstack-long-to-wide'])

STATEMENT = r"""
Finance wants a region-by-quarter revenue grid with totals on both edges, the classic spreadsheet pivot.

`df` is a sales log with columns `region` (str), `quarter` (str such as `"Q1"`) and `revenue` (float); a region and quarter pair can appear in many rows. Write `revenue_pivot(df)` that returns a DataFrame with:

- columns: `region`, then one column per quarter that appears in `df`, sorted ascending as strings, then `total`;
- one row per region, sorted ascending, followed by a **last row** whose `region` is `"total"`;
- each cell is the **sum** of `revenue` for that region and quarter, `0.0` when the pair never occurs;
- the `total` column holds each row's sum, and the `"total"` row holds each column's sum (its `total` cell is the grand total).

## Example
```
  region quarter  revenue
0   east      Q1     10.0          region    Q1    Q2  total
1   east      Q1      5.0     ->     east  15.0   0.0   15.0
2   west      Q2      8.0            west   0.0   8.0    8.0
                                    total  15.0   8.0   23.0
```

## Input
`df` has at least one row, and no region or quarter is named `"total"`.
"""

STARTER = '''import pandas as pd

def revenue_pivot(df):
    """Region x quarter revenue sums, 0.0 for gaps, with a 'total' column and a final 'total' row."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def revenue_pivot(df):
    grid = df.pivot_table(index='region', columns='quarter', values='revenue', aggfunc='sum',
                          fill_value=0.0, margins=True, margins_name='total')
    # pivot_table names the column axis "quarter"; drop that so the flattened frame is clean.
    grid.columns.name = None
    return grid.reset_index().astype({c: float for c in grid.columns})
'''

HINTS = [
    '`df.pivot_table(index=..., columns=..., values=..., aggfunc="sum")` turns one column\'s distinct values into new columns and aggregates the duplicates.',
    '`fill_value=0.0` replaces the empty cells, and `margins=True, margins_name="total"` appends both the total column and the total row.',
    'Rows and columns come out sorted already, with the margins last. Finish with `reset_index()` so `region` becomes a column.',
]

_r = np.random.default_rng(12)
_n = 40
_sales = pd.DataFrame({'region': _r.choice(['north', 'south', 'east', 'west', 'central'], _n),
                       'quarter': _r.choice(['Q1', 'Q2', 'Q3', 'Q4'], _n),
                       'revenue': _r.integers(1, 200, _n) * 0.5})


def _df(rows):
    return pd.DataFrame(rows, columns=['region', 'quarter', 'revenue']).astype({'revenue': float})


TESTS = [
    (_df([['east', 'Q1', 10.0], ['east', 'Q1', 5.0], ['west', 'Q2', 8.0]]),),
    (_df([['b', 'Q3', 1.0], ['a', 'Q1', 2.0], ['a', 'Q3', 3.0], ['b', 'Q1', 4.0]]),),
    (_df([['only', 'Q4', 99.5]]),),
    (_df([['x', 'Q2', -50.0], ['x', 'Q2', 50.0], ['y', 'Q1', 1e9]]),),
    (_df([['r', '2024-Q1', 1.0], ['r', '2023-Q4', 2.0], ['s', '2024-Q2', 3.0]]),),
    (_sales,),
]

EXPLANATION = r"""
## Why it works
A pivot is a group-by on two keys followed by moving one key into the columns. `pivot_table` does both, plus the two spreadsheet extras: `fill_value` for combinations that never occur, and `margins` for the total row and column. The margins are computed from the **raw data**, so the grand total is the sum of all revenue, not a sum of rounded cells. Region and quarter labels come out sorted, and the margin label is appended after them.

The same table by hand: `df.groupby(["region", "quarter"])["revenue"].sum().unstack(fill_value=0.0)`, then add `total` with `.sum(axis=1)` and the total row with `.sum()`.

## Complexity
$O(N)$ to group, plus $O(R \cdot Q)$ for the grid of $R$ regions and $Q$ quarters.

## Common bugs
- `df.pivot(...)` raises "Index contains duplicate entries" as soon as a pair repeats (test 1). `pivot` reshapes; `pivot_table` reshapes **and** aggregates.
- The default `aggfunc` is `"mean"`, not `"sum"`.
- Without `fill_value`, missing cells are NaN and the margin sums still work, but the grid shows NaN where the statement wants 0.0.
- Quarter labels sort as **strings**: `"2023-Q4"` comes before `"2024-Q1"` (test 5), which is right for that format but not for `"Q10"` vs `"Q2"`.
- Forgetting `reset_index()`: the regions stay in the index, which the comparison ignores.

## Interview follow-ups
- Show each cell as a percentage of its region's total instead. (Divide the grid by its `total` column, row-wise, with `div(..., axis=0)`.)
- Run the same pivot in SQL. Why do you need either one `CASE WHEN` per quarter or a database-specific `PIVOT`, and why is `pandas` easier when the quarters are not known in advance?
"""
