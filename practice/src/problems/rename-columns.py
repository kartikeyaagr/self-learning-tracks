import pandas as pd

META = dict(title='Normalise messy column headers', tags=['Data Processing'], diff='Easy', runtime='pandas',
            related=['column-selection', 'concatenate-dataframes'])

STATEMENT = r"""
Exported spreadsheets come with headers like `" Unit Price ($)"` or `"Order-ID"`. Convert them to `snake_case` so the rest of the code can use predictable names.

Write `tidy_columns(df, overrides)` that returns a copy of `df` with renamed columns (the data is unchanged):

1. **Normalise** each header: strip leading and trailing whitespace, lower-case it, replace every run of characters that are **not** `a-z` or `0-9` with a single `_`, then remove `_` from both ends.
2. Then apply `overrides`, a dict mapping a **normalised** name to the final name you want. Keys that match no column are ignored.

| raw header | normalised |
|---|---|
| `" Unit Price ($)"` | `unit_price` |
| `"Order-ID"` | `order_id` |
| `"__weird  name__"` | `weird_name` |
| `"2nd Score"` | `2nd_score` |

## Input
- `df`: a DataFrame with string headers. The test data guarantees the final names are distinct and non-empty.
- `overrides`: a dict of `str -> str`, possibly empty.
"""

STARTER = '''import pandas as pd

def tidy_columns(df, overrides):
    """Normalise headers to snake_case, then rename with overrides (keyed by normalised name)."""
    raise NotImplementedError
'''

SOLUTION = '''import re
import pandas as pd

def _snake(name):
    return re.sub(r'[^a-z0-9]+', '_', name.strip().lower()).strip('_')

def tidy_columns(df, overrides):
    # rename() accepts a function (applied to every label) or a dict (unknown keys ignored).
    return df.rename(columns=_snake).rename(columns=overrides)
'''

HINTS = [
    '`df.rename(columns=f)` calls `f` on every header; `df.rename(columns={"old": "new"})` renames just the listed ones and ignores the rest.',
    'A regular expression does the middle step in one call: `re.sub(r"[^a-z0-9]+", "_", s)` replaces each **run** of other characters.',
    'Order matters: strip and lower-case before the substitution, `.strip("_")` after it, and apply the overrides last because their keys are normalised names.',
]

TESTS = [
    (pd.DataFrame({' Unit Price ($)': [1.5], 'Order-ID': [7], 'qty': [2]}), {}),
    (pd.DataFrame({'Customer Name': ['a'], 'E-mail': ['x@y']}), {'e_mail': 'email'}),
    (pd.DataFrame({'__weird  name__': [1], '2nd Score': [2.0], 'ALLCAPS': [3]}), {'missing_key': 'nope'}),
    (pd.DataFrame({'already_clean': [1, 2, 3]}), {}),
    (pd.DataFrame({'Temp (°C)': [21.5, 22.0], 'Rel. Humidity %': [40, 41], 'Station#': ['n', 's']}),
     {'temp_c': 'temp', 'station': 'station_id'}),
    (pd.DataFrame({'a.b.c': [1], ' A  B ': [2], 'x--y__z': [3]}), {'a_b': 'ab'}),
]

EXPLANATION = r"""
## Why it works
`DataFrame.rename(columns=...)` takes either a mapping or a function. A function suits a rule that applies to every header; a dict suits a few explicit exceptions, and a dict rename silently ignores keys that aren't present. Chaining the two keeps the rule and the exceptions separate. `rename` returns a new frame, so `df` is untouched.

The normaliser order matters: lower-case before the regex (so `A-Z` survive as letters), collapse **runs** with `+` (so `"weird  name"` gives one underscore, not two), and strip underscores last (so a leading `"__"` or a trailing `")"` leaves no edge underscore).

## Complexity
$O(C \cdot L)$ for $C$ headers of length $L$; no data is copied beyond the new frame's metadata.

## Common bugs
- `[^a-z0-9]` without `+` turns `"a  b"` into `"a__b"`.
- Running the regex before `lower()` replaces every capital letter with `_`.
- Applying `overrides` to the raw names: their keys are normalised names, so nothing matches (tests 2 and 5).
- Assigning `df.columns = [...]` modifies the caller's frame in place. Use `rename`, or copy first.
- `str.isalnum` keeps non-ASCII letters such as `°` or `é`; the rule here keeps only ASCII letters and digits (test 5).

## Interview follow-ups
- Two headers normalise to the same name (`"Order ID"` and `"order-id"`). Detect it and resolve it with `_2`, `_3` suffixes in a deterministic way.
- Headers like `"OrderID"` or `"unitPrice"` are camelCase. How would you split them into `order_id` and `unit_price`? (Insert `_` before an upper-case letter that follows a lower-case letter or digit.)
"""
