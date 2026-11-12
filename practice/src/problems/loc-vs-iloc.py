import numpy as np
import pandas as pd

META = dict(title='Labels versus positions after a sort', tags=['Data Processing'], diff='Easy', runtime='pandas',
            related=['head-and-tail-operations', 'resetting-index', 'setting-index'])

STATEMENT = r"""
`.iloc` selects by **position** (0, 1, 2, ... from the top), `.loc` selects by **index label**. On a fresh frame the two coincide, so bugs hide. After a sort they come apart.

Take a leaderboard `df` with columns `name` (str) and `score` (number) and the default index `0..len-1`. Sort it by `score` **descending**, keeping the original order for equal scores, and **do not** reset the index. Call the result `board`.

Write `label_vs_position(df, k)` that returns a tuple of three values:

1. `by_position`: the `name` of the row at position `k` of `board`;
2. `by_label`: the `name` of the row of `board` whose index label is `k` (that is, row `k` of the original `df`);
3. `through_label`: a list of the names of `board` from its first row **down to and including** the row labelled `k`, in `board` order.

## Input
- `df`: at least one row.
- `k`: an int with `0 <= k < len(df)`.

## Example
```
   name  score            board:    name  score
0  ann   3                       1  bob   9
1  bob   9                       2  cat   5
2  cat   5                       0  ann   3
```
`label_vs_position(df, 0)` is `('bob', 'ann', ['bob', 'cat', 'ann'])`.
"""

STARTER = '''import pandas as pd

def label_vs_position(df, k):
    """Sort by score desc (stable), keep the index. Return (name at position k, name at label k, names up to label k)."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def label_vs_position(df, k):
    # kind="stable" keeps the original order among equal scores.
    board = df.sort_values('score', ascending=False, kind='stable')
    by_position = board['name'].iloc[k]
    by_label = board['name'].loc[k]
    # A .loc slice includes its end label and follows the frame's current row order.
    through_label = board['name'].loc[:k].tolist()
    return by_position, by_label, through_label
'''

HINTS = [
    'Sort with `sort_values("score", ascending=False, kind="stable")`. The default quicksort does not promise to keep ties in their original order.',
    'After sorting, `board.iloc[k]` and `board.loc[k]` are different rows in general. Which one follows the original row numbers?',
    'Label slices are inclusive: `board["name"].loc[:k]` runs from the first row of `board` to the row labelled `k`, wherever that row ended up.',
]

_r = np.random.default_rng(13)
_names = [f'p{i:02d}' for i in range(15)]
_board = pd.DataFrame({'name': _names, 'score': _r.integers(0, 6, 15)})
TESTS = [
    (pd.DataFrame({'name': ['ann', 'bob', 'cat'], 'score': [3, 9, 5]}), 0),
    (pd.DataFrame({'name': ['ann', 'bob', 'cat'], 'score': [3, 9, 5]}), 2),
    (pd.DataFrame({'name': ['solo'], 'score': [1.5]}), 0),
    (pd.DataFrame({'name': ['a', 'b', 'c', 'd'], 'score': [7, 7, 7, 7]}), 2),
    (pd.DataFrame({'name': ['a', 'b', 'c', 'd', 'e'], 'score': [-1.0, 4.5, 4.5, -3.0, 10.0]}), 1),
    (_board, 4),
    (_board, 11),
]

EXPLANATION = r"""
## Why it works
`sort_values` reorders rows but carries each row's label with it, so the label `k` still means "row `k` of the original data". `iloc[k]` ignores labels and counts from the top of the current order. A `.loc` slice `[:k]` is label-based and therefore **inclusive** of `k`, but it walks the rows in their current physical order, from the first row to wherever label `k` now sits.

## Complexity
$O(N \log N)$ for the sort. Each lookup afterwards is $O(1)$ for `iloc` and an index lookup for `loc`.

## Common bugs
- Treating `df.loc[k]` and `df.iloc[k]` as the same thing because they agree on a fresh frame. Tests 1 and 2 separate them.
- Expecting `loc[:k]` to stop **before** `k`, like Python and `iloc` slices do.
- Using the default sort algorithm. Ties (test 4) can come out in any order, so the answer would not be determined.
- Calling `reset_index(drop=True)` after sorting: then labels equal positions again and `by_label` is wrong.

## Interview follow-ups
- What does `df.loc[2:5]` do on an index of floats or strings, and when does it raise a `KeyError`? (It needs the endpoints to exist unless the index is sorted.)
- Chained indexing `df[df.score > 3]["name"] = "x"` may silently do nothing. Why, and how does `df.loc[df.score > 3, "name"] = "x"` fix it?
"""
