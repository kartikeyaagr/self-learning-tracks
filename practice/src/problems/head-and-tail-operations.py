import numpy as np
import pandas as pd

META = dict(title='Preview both ends of a table', tags=['Data Processing'], diff='Easy', runtime='pandas',
            related=['inspect-dataframe-shape', 'loc-vs-iloc'])

STATEMENT = r"""
A log viewer shows the first `n` and the last `n` rows of a table so you can see where it starts and where it ends. When the table is short the two windows overlap, and an overlapping row must be shown only **once**.

Write `peek(df, n)` that returns a new DataFrame with:

- a first column `row` holding each selected row's 0-based position in `df`;
- then all columns of `df`, in their original order;
- the rows that are among the first `n` **or** the last `n` rows of `df`, each once, in their original order.

## Input
- `df`: a DataFrame with a default `0..len-1` index; it may be empty. It has no column called `row`.
- `n`: an int, `n >= 0`. `n = 0` selects nothing.

## Example
For a 5-row frame and `n = 2`, the result has rows at positions `0, 1, 3, 4`. For `n = 3` it has all five, not six.
"""

STARTER = '''import pandas as pd

def peek(df, n):
    """Rows among the first n or last n of df, each once, with a leading 'row' position column."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def peek(df, n):
    out = pd.concat([df.head(n), df.tail(n)])
    out = out[~out.index.duplicated()]     # overlap: keep each original row once
    out = out.sort_index()
    out.insert(0, 'row', out.index)
    return out.reset_index(drop=True)
'''

HINTS = [
    '`df.head(n)` and `df.tail(n)` return the two windows and keep the original row labels in their index.',
    'Stack them with `pd.concat`. Rows from the overlap now appear twice, with the same index label.',
    'Drop repeats with `out[~out.index.duplicated()]`, then copy the index into a `row` column. Watch out: `df.tail(0)` is empty, but slicing `df.iloc[-0:]` is the whole frame.',
]

_r = np.random.default_rng(5)
_log = pd.DataFrame({'ts': np.arange(100, 130), 'level': _r.choice(['INFO', 'WARN', 'ERROR'], 30),
                     'latency': _r.exponential(20, 30).round(2)})
TESTS = [
    (pd.DataFrame({'ts': [1, 2, 3, 4, 5], 'msg': ['a', 'b', 'c', 'd', 'e']}), 2),
    (pd.DataFrame({'ts': [1, 2, 3, 4, 5], 'msg': ['a', 'b', 'c', 'd', 'e']}), 3),
    (pd.DataFrame({'ts': [1, 2, 3], 'msg': ['a', 'b', 'c']}), 0),
    (pd.DataFrame({'ts': pd.Series([], dtype=int), 'msg': pd.Series([], dtype=object)}), 4),
    (pd.DataFrame({'x': [9.5]}), 1),
    (pd.DataFrame({'x': [9.5, -1.0]}), 10),
    (_log, 4),
]

EXPLANATION = r"""
## Why it works
`head` and `tail` keep the row labels of the original frame, so after concatenation a row that sits in both windows appears twice **with the same label**. That makes the overlap easy to spot: `index.duplicated()` marks the second copy. Sorting by the index restores the original order, and with a default index the label is the 0-based position.

A purely positional version also works: `pos = sorted(set(range(min(n, L))) | set(range(max(L - n, 0), L)))` then `df.iloc[pos]`.

## Complexity
$O(n)$ rows are copied, independent of the table size, which is why `head`/`tail` are cheap even on large frames.

## Common bugs
- `pd.concat([df.head(n), df.tail(n)])` alone duplicates the overlap (test 2).
- `df.iloc[-n:]` with `n = 0` is `df.iloc[0:]`, the whole frame. `tail(0)` gets this right (test 3).
- Calling `reset_index()` before recording positions loses them.
- `drop_duplicates()` on the **values** would wrongly merge two different rows that happen to hold identical data. Deduplicate on the index.

## Interview follow-ups
- `df.sample(n, random_state=0)` gives a random preview. Why is a random sample often a better first look than `head`, for example on a file sorted by date?
- How would you peek at the first and last rows **per group** (say, per user)? (`groupby(...).head(n)` and `.tail(n)`, same overlap trick on the index.)
"""
