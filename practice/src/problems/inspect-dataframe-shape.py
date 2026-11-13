import numpy as np
import pandas as pd

META = dict(title='First-look profile of a table', tags=['Data Processing'], diff='Easy', runtime='pandas',
            lessons=['air-fanout-v1:11.12'], related=['data-types-overview', 'handle-missing-values'])

STATEMENT = r"""
Before cleaning a new dataset you want four numbers: how big it is and how much of it is missing.

Write `table_profile(df)` that returns a dict with exactly these keys, each an `int`:

| key | meaning |
|---|---|
| `rows` | number of rows |
| `cols` | number of columns |
| `missing` | number of missing cells (NaN or None) in the whole frame |
| `complete_rows` | number of rows with **no** missing cell |

## Input
- `df`: any DataFrame. It may have zero rows; it has at least one column. Text columns may contain `None`, numeric columns may contain `NaN`.

## Example
```
   a     b
0  1.0   x
1  NaN   y
2  3.0   None
```
gives `{'rows': 3, 'cols': 2, 'missing': 2, 'complete_rows': 1}`.
"""

STARTER = '''import pandas as pd

def table_profile(df):
    """Return {'rows', 'cols', 'missing', 'complete_rows'} as ints."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def table_profile(df):
    rows, cols = df.shape
    mask = df.isna()                      # True where a cell is NaN or None
    return {
        'rows': int(rows),
        'cols': int(cols),
        'missing': int(mask.to_numpy().sum()),
        'complete_rows': int((~mask.any(axis=1)).sum()),
    }
'''

HINTS = [
    '`df.shape` is a `(rows, cols)` tuple. `len(df)` is the row count alone.',
    '`df.isna()` returns a boolean frame of the same shape that is True for both NaN and None.',
    'Summing that frame twice (or `.to_numpy().sum()`) counts missing cells. A row is complete when `mask.any(axis=1)` is False. Cast the results to plain `int`.',
]

_r = np.random.default_rng(21)
_big = pd.DataFrame(_r.normal(size=(40, 5)), columns=list('abcde'))
_big = _big.mask(_r.random(_big.shape) < 0.1)       # about 10% of cells become NaN
TESTS = [
    (pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': ['x', 'y', None]}),),
    (pd.DataFrame({'id': [1, 2, 3, 4], 'name': ['p', 'q', 'r', 's']}),),
    (pd.DataFrame({'a': pd.Series([], dtype=float), 'b': pd.Series([], dtype=object)}),),
    (pd.DataFrame({'x': [np.nan, np.nan], 'y': [np.nan, np.nan]}),),
    (pd.DataFrame({'only': [None]}),),
    (pd.DataFrame({'v': [1e300, -1e300, 0.0], 'w': [np.nan, 2.0, 3.0], 't': ['a', None, 'c']}),),
    (_big,),
]

EXPLANATION = r"""
## Why it works
`df.shape` gives both dimensions at once. `isna()` normalises the two spellings of "missing" that pandas uses (`NaN` in float columns, `None` in object columns) into one boolean mask, and every other count is a reduction of that mask: a total sum for `missing`, and a row-wise `any` for incomplete rows.

## Complexity
$O(R \cdot C)$ time and one boolean mask of the same size.

## Common bugs
- `df.size` counts cells, not rows. `len(df.columns)` and `df.shape[1]` are the column count.
- `df.isna().sum()` is a **per-column** Series. You need a second sum (or `.to_numpy().sum()`) for the grand total.
- `df.dropna()` drops rows with **any** missing value by default, so `len(df.dropna())` is a valid way to get `complete_rows`; `dropna(how="all")` is not.
- Returning NumPy integers is fine for the grader, but `json.dumps` and some APIs reject them. Cast with `int(...)`.
- `df.count()` counts **non-missing** values per column, the complement of what you want.

## Interview follow-ups
- Which columns would you drop outright, and how do you find them? (`df.isna().mean() > threshold`.)
- `df.info()` also prints memory use. Why can an object column of short strings use far more memory than an `int64` column, and what does `memory_usage(deep=True)` change?
"""
