import numpy as np
import pandas as pd

META = dict(title='Rank filtered rows and keep their origin', tags=['Data Processing'], diff='Medium', runtime='pandas',
            related=['loc-vs-iloc', 'setting-index', 'boolean-indexing'])

STATEMENT = r"""
After filtering and sorting, a frame's index is a jumble of the original row numbers. Sometimes you want to throw it away; sometimes it is the only link back to the source rows and must be kept as data.

`df` has the default index `0..len-1`, a numeric column named by `column`, and any other columns. Write `ranked_rows(df, column, threshold)` that:

1. keeps the rows where `df[column] > threshold` (strictly greater);
2. sorts them by `column` **descending**; equal values keep their original order;
3. returns a DataFrame whose columns are, in this order:
   - `rank`: `1, 2, 3, ...` in the sorted order;
   - `row_id`: the row's 0-based position in the original `df`;
   - then all columns of `df` in their original order.

If no row passes, return a frame with those columns and no rows.

## Input
- `column` names an int or float column with no missing values. `df` has no columns called `rank` or `row_id`.
"""

STARTER = '''import pandas as pd

def ranked_rows(df, column, threshold):
    """Rows with df[column] > threshold, sorted desc (stable), with rank and original row_id columns first."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def ranked_rows(df, column, threshold):
    kept = df[df[column] > threshold].sort_values(column, ascending=False, kind='stable')
    # reset_index() without drop=True moves the old labels (original row numbers) into a column.
    out = kept.rename_axis('row_id').reset_index()
    out.insert(0, 'rank', range(1, len(out) + 1))
    return out
'''

HINTS = [
    'Filter with a boolean mask, then `sort_values(column, ascending=False, kind="stable")`. The surviving rows keep their original index labels.',
    '`reset_index()` (without `drop=True`) turns those labels into a regular column and gives the frame a fresh `0..k-1` index.',
    'Name the index `row_id` first (`rename_axis("row_id")`) so the new column gets the right name, then `insert(0, "rank", range(1, len(out) + 1))`.',
]

_r = np.random.default_rng(9)
_scores = pd.DataFrame({'model': [f'm{i}' for i in range(20)], 'auc': _r.integers(50, 100, 20) / 100,
                        'params': _r.integers(1, 50, 20)})
TESTS = [
    (pd.DataFrame({'name': ['a', 'b', 'c', 'd'], 'score': [5, 1, 9, 7]}), 'score', 4),
    (pd.DataFrame({'name': ['a', 'b', 'c'], 'score': [2.0, 2.0, 3.0]}), 'score', 0),
    (pd.DataFrame({'name': ['a', 'b'], 'score': [1, 2]}), 'score', 2),
    (pd.DataFrame({'v': [-5.0, -1.0, -3.0]}), 'v', -4.0),
    (pd.DataFrame({'id': [10], 'x': [1e12]}), 'x', -1e12),
    (_scores, 'auc', 0.7),
]

EXPLANATION = r"""
## Why it works
Filtering and sorting never change a row's label, only which rows are present and in what order. So after both steps the index is precisely "where did this row come from". `reset_index()` moves that index into a column (named after the index, hence `rename_axis`) and replaces it with `0..k-1`; `reset_index(drop=True)` would discard it instead. The rank is then simply position + 1 in the sorted frame.

## Complexity
$O(N)$ to filter, $O(K \log K)$ to sort the $K$ survivors.

## Common bugs
- `reset_index(drop=True)` before recording the origin loses `row_id` for good.
- Plain `reset_index()` on an unnamed index produces a column called `index`.
- `rank` from `df[column].rank(ascending=False)` gives **average** ranks for ties (2.5, 2.5). The statement wants ordinal ranks that follow the stable sort order (test 2).
- Forgetting `kind="stable"`: tied rows may swap, changing both `rank` and `row_id` order.
- `>=` instead of `>` keeps the boundary row (test 3 must come back empty).

## Interview follow-ups
- Use `df[column].rank(method="first", ascending=False)` to compute the same ranks without sorting. Why does `"first"` match the stable sort?
- `groupby(...).sum()` returns the group keys as the index. Compare `reset_index()` afterwards with `groupby(..., as_index=False)`.
"""
