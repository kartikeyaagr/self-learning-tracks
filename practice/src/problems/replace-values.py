import numpy as np
import pandas as pd

META = dict(title='Standardise survey answers and sentinels', tags=['Data Processing'], diff='Medium', runtime='pandas',
            lessons=['air-fanout-v1:11.12'], related=['change-data-types', 'handle-missing-values', 'drop-duplicates'])

STATEMENT = r"""
A survey export has two problems. The yes/no column was typed by hand, and the numeric score uses magic numbers for "no answer".

`df` has columns `id` (int), `answer` (str or `None`) and `score` (int). Write `clean_survey(df)` that returns a new DataFrame with columns `id, answer, score`:

**`answer`**: strip whitespace and lower-case, then map

| cleaned text | result |
|---|---|
| `yes`, `y`, `true`, `1` | `"yes"` |
| `no`, `n`, `false`, `0` | `"no"` |
| anything else, including `""` and `None` | `"unknown"` |

**`score`**: the values `-999` and `-1` mean "missing" and become NaN; every other value (including other negatives) is kept. The result is a float column.

Rows and `id` are unchanged. Don't modify `df`.
"""

STARTER = '''import numpy as np
import pandas as pd

def clean_survey(df):
    """Map answers to yes/no/unknown and turn score sentinels (-999, -1) into NaN."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
import pandas as pd

YES = ['yes', 'y', 'true', '1']
NO = ['no', 'n', 'false', '0']

def clean_survey(df):
    text = df['answer'].str.strip().str.lower()          # None stays missing through .str
    # map() sends unmatched values to NaN, which is exactly the "anything else" bucket.
    answer = text.map({**dict.fromkeys(YES, 'yes'), **dict.fromkeys(NO, 'no')}).fillna('unknown')
    score = df['score'].astype(float).replace([-999.0, -1.0], np.nan)
    return pd.DataFrame({'id': df['id'], 'answer': answer, 'score': score})
'''

HINTS = [
    'Clean the text first with the `.str` accessor (`.str.strip().str.lower()`); missing values pass through it unchanged.',
    '`Series.replace(dict)` leaves values it does not know untouched, while `Series.map(dict)` turns them into NaN. Which one gives you the "anything else" bucket for free?',
    'After `map`, `fillna("unknown")` catches both unmatched text and `None`. For the score, `replace([-999, -1], np.nan)` on a float copy of the column.',
]

_r = np.random.default_rng(55)
_n = 30
_raw = pd.DataFrame({'id': np.arange(_n),
                     'answer': _r.choice(['Yes', ' y', 'NO', 'n ', 'True', '0', 'maybe', '', 'nope', 'FALSE'], _n),
                     'score': _r.choice([-999, -1, 0, 3, 7, 10, -5], _n)})
TESTS = [
    (pd.DataFrame({'id': [1, 2, 3, 4], 'answer': ['Yes', ' n', 'maybe', None], 'score': [7, -999, 10, -1]}),),
    (pd.DataFrame({'id': [10, 11], 'answer': ['TRUE ', '0'], 'score': [0, -2]}),),
    (pd.DataFrame({'id': [5], 'answer': [''], 'score': [-1]}),),
    (pd.DataFrame({'id': [1, 2, 3], 'answer': [None, None, None], 'score': [1, 2, 3]}),),
    (pd.DataFrame({'id': [7, 8, 9, 10], 'answer': ['yes please', 'Y', 'no!', 'f'], 'score': [-9990, 999, -998, 5]}),),
    (pd.DataFrame({'id': pd.Series([], dtype=int), 'answer': pd.Series([], dtype=object), 'score': pd.Series([], dtype=int)}),),
    (_raw,),
]

EXPLANATION = r"""
## Why it works
The two columns need opposite behaviour for unknown values:

- For `answer`, every value must land in one of three buckets. `map` with a dict does that in one step: listed values get their label, everything else (and missing input) becomes NaN, which `fillna("unknown")` relabels.
- For `score`, only two specific values change and everything else must survive. That is what `replace` is for: unmatched values pass through untouched. Replacing with NaN needs a float column, because NumPy integers have no NaN.

## Complexity
$O(N)$: each step is one vectorised pass with a hash lookup per value.

## Common bugs
- Using `replace` for the answers: `"maybe"` survives instead of becoming `"unknown"`.
- Comparing before cleaning: `" n"` and `"TRUE "` miss the table.
- Mapping `"yes please"` or `"no!"` to yes/no by substring search. The rule is an exact match after cleaning (test 5).
- `df["score"] < 0` as the sentinel test also wipes legitimate negatives such as `-2` and `-998`.
- `df.replace(-1, np.nan)` on the whole frame also rewrites an `id` of `-1`. Target the column.

## Interview follow-ups
- The sentinel list differs per column (`-1` for age, `9999` for income). How would you express that with one `df.replace({...})` call using a nested dict?
- Why might you keep a separate `score_missing` flag rather than only NaN, and when does "missing" carry signal for a model?
"""
