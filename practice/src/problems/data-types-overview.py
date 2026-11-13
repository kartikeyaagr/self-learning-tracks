import numpy as np
import pandas as pd

META = dict(title='Group columns by data type', tags=['Data Processing'], diff='Easy', runtime='pandas',
            lessons=['air-fanout-v1:11.12'], related=['change-data-types', 'inspect-dataframe-shape'])

STATEMENT = r"""
A preprocessing pipeline treats columns differently by type: numbers get scaled, booleans become 0/1, text gets encoded. The first step is to sort the columns into buckets.

Write `columns_by_kind(df)` that returns a dict with exactly four keys, `"int"`, `"float"`, `"bool"` and `"text"`. Each value is the list of column names of that kind, in the order the columns appear in `df`.

| key | dtype of the column |
|---|---|
| `int` | any integer dtype (`int64`, `int32`, ...) |
| `float` | any floating dtype |
| `bool` | `bool` |
| `text` | `object` (here: strings) |

A key with no matching column maps to an empty list.

## Input
`df`: a DataFrame whose columns are each one of the four kinds above. It may have zero rows. Note that an integer column with a missing value is stored as float.
"""

STARTER = '''import pandas as pd

def columns_by_kind(df):
    """Return {'int': [...], 'float': [...], 'bool': [...], 'text': [...]} in column order."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def columns_by_kind(df):
    # dtype.kind is a one-letter code: i/u integer, f float, b bool, O object.
    kind = {'i': 'int', 'u': 'int', 'f': 'float', 'b': 'bool', 'O': 'text'}
    out = {'int': [], 'float': [], 'bool': [], 'text': []}
    for name, dtype in df.dtypes.items():
        out[kind[dtype.kind]].append(name)
    return out
'''

HINTS = [
    '`df.dtypes` is a Series mapping each column name to its dtype.',
    'Every NumPy dtype has a one-letter `.kind`: `"i"` for signed integers, `"f"` for floats, `"b"` for booleans, `"O"` for objects.',
    'Start from a dict of four empty lists and append each column name to the list for its kind. Alternatively, call `df.select_dtypes(include=...)` once per bucket.',
]

_r = np.random.default_rng(8)
TESTS = [
    (pd.DataFrame({'age': [30, 41], 'height': [1.7, 1.8], 'member': [True, False], 'city': ['Rome', 'Kyiv']}),),
    (pd.DataFrame({'score': [1.0, np.nan, 3.0], 'count': [1, 2, 3]}),),
    (pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype=object)}),),
    (pd.DataFrame({'x': ['p', 'q'], 'y': ['r', 's'], 'flag': [False, False]}),),
    (pd.DataFrame({'small': np.array([1, 2], dtype='int32'), 'f32': np.array([0.5, 1.5], dtype='float32')}),),
    (pd.DataFrame({f'c{i}': (_r.integers(0, 9, 4) if i % 3 == 0 else _r.random(4) if i % 3 == 1 else _r.random(4) > 0.5)
                   for i in range(9)}),),
]

EXPLANATION = r"""
## Why it works
Every pandas column is backed by an array with one dtype, and `df.dtypes` lists them. Mapping a dtype to a bucket through `dtype.kind` is robust: it covers `int8` through `int64` (and unsigned ints) with one letter, so you never maintain a list of dtype names. Iterating over `df.dtypes.items()` keeps the original column order.

## Complexity
$O(C)$ for $C$ columns; the data itself is never read.

## Common bugs
- Comparing `str(dtype) == "int64"` misses `int32` (test 5) and the float32 column.
- `select_dtypes("number")` excludes bool, but `pd.api.types.is_numeric_dtype` returns True for a bool column. Keep the bool check separate and explicit.
- An integer column with one missing value is `float64` (test 2 would have `score` as float even if it held whole numbers). pandas has the nullable `Int64` dtype for that case.
- Forgetting empty buckets: the result must always have all four keys.

## Interview follow-ups
- A column of numbers arrived as strings (`"12"`, `"7"`), so its dtype is `object`. How would you detect "numeric-looking" text columns? (`pd.to_numeric(s, errors="coerce").notna().mean()` close to 1.)
- When is the `category` dtype worth it, and what does it do to memory and to `groupby`?
"""
