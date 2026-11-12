import numpy as np
import pandas as pd

META = dict(title='Parse a table that arrived as text', tags=['Data Processing'], diff='Medium', runtime='pandas',
            lessons=['air-fanout-v1:11.12'], related=['data-types-overview', 'replace-values'])

STATEMENT = r"""
A CSV was read with every column as text, so nothing can be summed or filtered numerically. `df` has four columns, all strings:

| column | examples | convert to |
|---|---|---|
| `id` | `"007"`, `"12"` | `int64` |
| `price` | `"$1,234.50"`, `"-$5.00"`, `"3"` | `float64` (drop `$` and `,`) |
| `qty` | `"3"`, `" 12 "`, `""`, `"n/a"`, `"2.5"` | `float64`; anything that is not a number becomes NaN |
| `active` | `"true"`, `"False"`, `" TRUE"` | `bool` (case and surrounding spaces ignored) |

Write `parse_types(df)` that returns a new DataFrame with the same four columns, in the order `id, price, qty, active`, holding the converted values. Rows are unchanged.

## Input
- `id` always holds digits; `price` is always a valid amount once `$` and `,` are removed; `active` is always `true` or `false` in some casing, possibly padded with spaces. Only `qty` can be invalid.
"""

STARTER = '''import pandas as pd

def parse_types(df):
    """All columns are strings. Return id int64, price float64, qty float64 (bad -> NaN), active bool."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def parse_types(df):
    return pd.DataFrame({
        'id': df['id'].astype('int64'),                     # int("007") == 7
        'price': df['price'].str.replace(r'[$,]', '', regex=True).astype('float64'),
        # errors="coerce" turns unparsable text into NaN instead of raising.
        'qty': pd.to_numeric(df['qty'].str.strip(), errors='coerce').astype('float64'),
        # astype(bool) would make every non-empty string True, including "false".
        'active': df['active'].str.strip().str.lower() == 'true',
    })
'''

HINTS = [
    'Each column needs its own conversion. `astype("int64")` parses digit strings; leading zeros are fine.',
    'For `price`, clean the text first with `.str.replace(r"[$,]", "", regex=True)`. For `qty`, `pd.to_numeric(..., errors="coerce")` is the tolerant parser.',
    '`astype(bool)` on strings tests for emptiness, so `"false"` becomes True. Compare the cleaned text instead: `s.str.strip().str.lower() == "true"`.',
]

_r = np.random.default_rng(90)
_n = 20
_raw = pd.DataFrame({
    'id': [f'{v:04d}' for v in _r.integers(0, 5000, _n)],
    'price': [f'{"-" if v < 0 else ""}${abs(v):,.2f}' for v in _r.normal(0, 3000, _n).round(2)],
    'qty': _r.choice(['1', ' 2', '3 ', '', 'n/a', '10', 'x', '0.5'], _n),
    'active': _r.choice(['true', 'FALSE', ' True', 'false '], _n),
})


def _df(rows):
    return pd.DataFrame(rows, columns=['id', 'price', 'qty', 'active'])


TESTS = [
    (_df([['007', '$1,234.50', '3', 'true'], ['12', '-$5.00', 'n/a', 'False']]),),
    (_df([['1', '3', ' 12 ', ' TRUE'], ['2', '$0.99', '', 'false']]),),
    (_df([['0', '$1,000,000.00', '2.5', 'FALSE']]),),
    (_df([['42', '$7', 'seven', 'false'], ['43', '$7.10', '-1', 'true'], ['44', '$,7', '1e3', 'TrUe']]),),
    (_df([]),),
    (_raw,),
]

EXPLANATION = r"""
## Why it works
Every column needs a different parser, so convert them one by one and assemble a new frame:

- `astype("int64")` on digit strings calls Python's `int`, which accepts leading zeros.
- Prices need cleaning before parsing; one regex removes both `$` and the thousands separator.
- `pd.to_numeric(errors="coerce")` is the tool for "parse what you can, mark the rest missing". Missing numbers force a float column.
- Booleans are the trap. Python's `bool("false")` is True (any non-empty string is), so `astype(bool)` is wrong for text. Comparing against `"true"` after cleaning yields a real bool Series.

## Complexity
$O(N)$ per column; the string operations dominate.

## Common bugs
- `df["active"].astype(bool)` makes every row True (test 1).
- `astype(float)` on `qty` raises on `"n/a"`; `errors="ignore"` (deprecated) would silently keep the column as text.
- `.str.replace("[$,]", "")` without `regex=True` searches for the literal text `[$,]` (pandas 2 defaults to `regex=False`) and removes nothing. Be explicit about `regex`.
- Removing `-` together with `$` loses the sign of refunds (test 1).
- `pd.to_numeric` happily parses `"1e3"` as 1000.0. That matches the rule here ("is a number"), but decide deliberately (test 4).

## Interview follow-ups
- With `pd.read_csv`, how do `dtype=`, `converters=`, `thousands=","` and `true_values=` let you do most of this at load time?
- `qty` should be an integer count but has missing values. What does the nullable `Int64` dtype give you over float64, and what breaks if you use it?
"""
