import numpy as np
import pandas as pd

META = dict(title='Assemble a feature matrix by column name', tags=['Data Processing', 'Feature Engineering'],
            diff='Easy', runtime='pandas', related=['column-selection', 'boolean-indexing'])

STATEMENT = r"""
A training table mixes identifiers, labels and many feature columns whose names share a prefix such as `f_`. To build the model input you select a few key columns plus every feature column.

Write `select_columns(df, keep, prefix)` that returns a new DataFrame with:

1. the columns named in the list `keep`, in the order given;
2. then every other column whose name **starts with** `prefix`, in the order they appear in `df`.

A column that is in `keep` and also starts with `prefix` appears **once**, in its `keep` position. Rows are unchanged.

## Input
- `df`: a DataFrame with string column names.
- `keep`: a list of distinct column names, all present in `df`; it may be empty. The result always has at least one column.
- `prefix`: a non-empty string; matching is case-sensitive.

## Example
Columns `["id", "f_age", "label", "f_bmi", "note"]`, `keep = ["label", "id"]`, `prefix = "f_"` give columns `["label", "id", "f_age", "f_bmi"]`.
"""

STARTER = '''import pandas as pd

def select_columns(df, keep, prefix):
    """Columns in keep (given order), then the remaining prefix columns (df order)."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def select_columns(df, keep, prefix):
    extra = [c for c in df.columns if c.startswith(prefix) and c not in keep]
    # A list inside [] selects (and orders) several columns and returns a DataFrame.
    return df[list(keep) + extra]
'''

HINTS = [
    'Passing a **list** of names, `df[["a", "b"]]`, returns a DataFrame with those columns in that order.',
    'Build the list of prefix columns with a comprehension over `df.columns` and `str.startswith`.',
    'Skip names already in `keep` so nothing is duplicated, then select `df[keep + extra]`.',
]

_r = np.random.default_rng(17)
_cols = ['user', 'f_clicks', 'y', 'f_dwell', 'g_score', 'f_', 'xf_bad', 'F_upper', 'f_time']
_wide = pd.DataFrame(_r.integers(0, 10, size=(6, len(_cols))), columns=_cols)
TESTS = [
    (pd.DataFrame({'id': [1, 2], 'f_age': [30, 40], 'label': [0, 1], 'f_bmi': [22.5, 27.1], 'note': ['a', 'b']}),
     ['label', 'id'], 'f_'),
    (pd.DataFrame({'id': [1, 2], 'f_age': [30, 40], 'label': [0, 1], 'f_bmi': [22.5, 27.1], 'note': ['a', 'b']}),
     ['f_bmi', 'id'], 'f_'),
    (pd.DataFrame({'a': [1], 'b': [2]}), ['b'], 'z'),
    (pd.DataFrame({'x_1': [1.0, 2.0, 3.0], 'x_2': [4.0, 5.0, 6.0]}), [], 'x_'),
    (_wide, ['y', 'user'], 'f_'),
    (_wide, ['f_time'], 'f'),
]

EXPLANATION = r"""
## Why it works
`df[list_of_names]` is the multi-column selector: it returns a new DataFrame whose columns follow the list's order exactly, which is how you both pick and reorder columns. The prefix columns come from a plain comprehension over `df.columns`, which preserves the frame's order, and the `c not in keep` test removes the overlap so no column is selected twice.

## Complexity
$O(C \cdot |keep|)$ for the comprehension (use a set for `keep` when it is long), plus copying the selected data.

## Common bugs
- `df["label", "id"]` (a tuple, not a list) looks for one column named `("label", "id")` and raises `KeyError`.
- `df.filter(like="f_")` matches the substring **anywhere**, so it would pick up `xf_bad` in tests 5 and 6. `df.filter(regex="^f_")` is the anchored version.
- Concatenating `keep + extra` without removing the overlap gives a frame with duplicate column labels (test 2).
- An empty `keep` is not a special case: `[] + extra` is just the prefix columns (test 4).

## Interview follow-ups
- How do you select columns by dtype instead of by name, for example every numeric feature? (`df.select_dtypes("number")`.)
- At serving time a feature column is missing. Should `df[cols]` fail loudly, or should `df.reindex(columns=cols)` fill NaN? Argue for one.
"""
