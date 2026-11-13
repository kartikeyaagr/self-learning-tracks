import pandas as pd

META = dict(title='Pick a column by a forgiving name', tags=['Data Processing'], diff='Easy', runtime='pandas',
            related=['multi-column-selection', 'rename-columns'])

STATEMENT = r"""
Spreadsheet exports have messy headers: `" Total Sales"`, `"count"`, `"Unit price"`. You want to fetch a column by name without caring about case or stray spaces at either end.

Write `pick_column(df, query)` that returns the column of `df` whose header matches `query` once both are **stripped** of leading and trailing whitespace and **lower-cased**.

## Input
- `df`: a DataFrame with string column names. Headers can contain spaces and can collide with DataFrame method names (`count`, `sum`, `size`).
- `query`: a string. Exactly one column matches it after normalising.

## Output
A pandas Series with that column's values in row order (its name and index are not checked).

## Example
With columns `["id", " Total Sales", "count"]`, `pick_column(df, "total sales")` returns the second column and `pick_column(df, "COUNT ")` the third.
"""

STARTER = '''import pandas as pd

def pick_column(df, query):
    """Return the column whose header equals query after strip() + lower()."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def pick_column(df, query):
    key = query.strip().lower()
    name = next(c for c in df.columns if c.strip().lower() == key)
    # Bracket access works for any header; df.count would return the bound method instead.
    return df[name]
'''

HINTS = [
    'Normalise the query once, then compare it with each normalised header in `df.columns`.',
    'Once you have the real header, use bracket access `df[name]`. Attribute access (`df.count`) breaks on spaces and on names that are also methods.',
    '`next(c for c in df.columns if c.strip().lower() == key)` finds the original header in one line.',
]

TESTS = [
    (pd.DataFrame({'id': [1, 2, 3], ' Total Sales': [10.5, 3.0, 7.25], 'count': [4, 1, 2]}), 'total sales'),
    (pd.DataFrame({'id': [1, 2, 3], ' Total Sales': [10.5, 3.0, 7.25], 'count': [4, 1, 2]}), 'COUNT '),
    (pd.DataFrame({'sum': [1, 2], 'size': [3, 4], 'mean': [5, 6]}), 'size'),
    (pd.DataFrame({'Name': ['ann'], 'NAME2': ['bob']}), '  name'),
    (pd.DataFrame({'Unit price': pd.Series([], dtype=float), 'qty': pd.Series([], dtype=int)}), 'unit PRICE'),
    (pd.DataFrame({'  city  ': ['Paris', 'Lima', 'Oslo', 'Pune'], 'pop': [2.1, 9.7, 0.7, 3.1]}), 'City'),
]

EXPLANATION = r"""
## Why it works
`df[name]` is the general way to select one column: it accepts any hashable label, including ones with spaces or ones that shadow DataFrame methods. The only real work is finding the exact label, so normalise both sides with `strip().lower()` and compare.

## Complexity
$O(C)$ string comparisons for $C$ columns. Selecting a column is $O(1)$: it returns a view-like Series without copying the data.

## Common bugs
- `df.count` is the `DataFrame.count` method, not the column called `count`. Attribute access silently returns the wrong object.
- `df.Total Sales` is a syntax error; attribute access needs a valid identifier.
- `str.strip()` only removes **leading and trailing** whitespace. Inner spaces in `Total Sales` stay, which is what the statement asks for.
- `df.columns.str.lower() == key` returns a boolean array; indexing `df.loc[:, mask]` gives a one-column **DataFrame**, not a Series.

## Interview follow-ups
- How do you rename every header to a clean form once, so the rest of the pipeline can use plain names? (`df.columns = df.columns.str.strip().str.lower()`, or see the rename problem.)
- Two headers normalise to the same key. What should the function do, and how would you detect it up front? (Check `normalised.duplicated().any()` and raise.)
"""
