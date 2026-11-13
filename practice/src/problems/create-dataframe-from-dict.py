import numpy as np
import pandas as pd

META = dict(title='User profiles from a nested dict', tags=['Data Processing'], diff='Easy', runtime='pandas',
            related=['inspect-dataframe-shape', 'rename-columns'])

STATEMENT = r"""
An API returns user profiles as a nested dict: the outer keys are user ids, and each value is a dict of numeric fields for that user. Not every user has every field.

```
{"u2": {"age": 31, "visits": 4},
 "u1": {"visits": 9, "spend": 12.5}}
```

Write `profiles_to_frame(profiles)` that turns this into a table.

## Input
- `profiles`: a dict mapping a user id (`str`) to a dict of `field -> number` (int or float). A user's dict may be empty. No field is named `user`.

## Output
A new DataFrame with:
- first a column `user` holding the user id, then one column per field that appears in **any** profile, sorted alphabetically;
- one row per user, sorted by `user` ascending;
- a field missing from a user's profile filled with `0` (a column with any gap therefore becomes float).

For the example above the result is

| user | age | spend | visits |
|---|---|---|---|
| u1 | 0.0 | 12.5 | 9 |
| u2 | 31.0 | 0.0 | 4 |

An empty `profiles` gives a frame with the single column `user` and no rows.
"""

STARTER = '''import pandas as pd

def profiles_to_frame(profiles):
    """profiles: {user_id: {field: number}}. Return a DataFrame: user, then fields A-Z; missing -> 0."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def profiles_to_frame(profiles):
    # orient="index": outer keys become row labels, inner keys become columns.
    # A user with an empty dict would vanish here, so add the full user list back with reindex.
    df = pd.DataFrame.from_dict(profiles, orient='index').reindex(list(profiles))
    df = df.reindex(columns=sorted(df.columns)).fillna(0).sort_index()
    df.insert(0, 'user', df.index)
    return df.reset_index(drop=True)
'''

HINTS = [
    '`pd.DataFrame(profiles)` puts the users in **columns**. Look at `pd.DataFrame.from_dict` and its `orient` argument.',
    'With `orient="index"` the outer keys become the index. Missing fields show up as NaN; `fillna(0)` fixes them, and `reindex(columns=sorted(...))` orders the columns.',
    'Move the index into a `user` column (`insert(0, "user", df.index)` or `reset_index` plus a rename) after sorting by it. Check what happens to a user whose profile is `{}`.',
]

_r = np.random.default_rng(3)
_fields = ['clicks', 'age', 'spend', 'visits', 'rating']
_rand = {f'user{int(i):03d}': {f: int(_r.integers(0, 100)) for f in _fields if _r.random() < 0.6}
         for i in _r.permutation(12)}
TESTS = [
    ({'u2': {'age': 31, 'visits': 4}, 'u1': {'visits': 9, 'spend': 12.5}},),
    ({'b': {'x': 1}, 'a': {'x': 2}, 'c': {'x': 3}},),
    ({},),
    ({'solo': {'height': 1.8, 'weight': 72.0, 'age': 40}},),
    ({'u1': {'k': 5}, 'u3': {}, 'u2': {'j': -2.5}},),
    ({'z9': {'b': 1e9, 'a': -1e9}, 'a1': {'c': 0.001}, 'm5': {'a': 7, 'c': 8, 'b': 9}},),
    (_rand,),
]

EXPLANATION = r"""
## Why it works
`pd.DataFrame(d)` treats the outer keys of a dict as **columns**, which is right for column-oriented data such as `{"age": [..], "visits": [..]}` but wrong here, where the outer keys are records. `DataFrame.from_dict(d, orient="index")` flips that: outer keys become row labels and the union of inner keys becomes the columns, with NaN wherever a record lacks a field. The rest is presentation: fill the gaps, order the columns and rows, and turn the index into an ordinary `user` column.

## Complexity
$O(U \cdot F)$ to build the table for $U$ users and $F$ distinct fields, plus $O(U \log U)$ for the sort.

## Common bugs
- Using `pd.DataFrame(profiles)` and getting users as columns. `.T` repairs the shape but is easy to forget.
- A user whose profile is `{}` contributes no cells, so `from_dict` drops that row. `reindex(list(profiles))` puts it back (test 5).
- Filling with 0 turns integer columns that had gaps into floats. That is expected: NaN forced them to float first.
- Leaving the ids in the index. The comparison ignores the index, so the ids would simply be missing.

## Interview follow-ups
- The API instead returns a list of records `[{"user": "u1", "visits": 9}, ...]`. What is the one-line constructor now? (`pd.DataFrame(records)` or `pd.json_normalize` for nested records.)
- Some fields are themselves dicts (`{"address": {"city": ...}}`). How do you flatten them into `address.city` columns?
"""
