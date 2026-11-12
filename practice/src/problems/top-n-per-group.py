import pandas as pd

META = dict(title='Top-N items per user with pandas', tags=['Data Processing', 'Recommender Systems'], diff='Medium', runtime='pandas',
            lessons=[], related=['sql-running-total', 'precision-recall-at-k'])

STATEMENT = r"""
You have a DataFrame of model scores with columns `user`, `item` and `score`. Build the list each user would see: their `n` highest-scoring items.

Write `top_n(df, n)` that returns a new DataFrame with:

- the same three columns, in the order `user, item, score`;
- at most `n` rows per user, ranked by `score` descending, ties broken by `item` ascending;
- rows sorted by `user` ascending, then by that rank;
- a fresh `0..k-1` index.

Users with fewer than `n` rows keep all of them. Don't modify `df`.
"""

STARTER = '''import pandas as pd

def top_n(df, n):
    """df: columns user, item, score. Return the top-n rows per user (see statement)."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def top_n(df, n):
    ranked = df.sort_values(['user', 'score', 'item'], ascending=[True, False, True])
    return ranked.groupby('user', sort=False).head(n)[['user', 'item', 'score']].reset_index(drop=True)
'''

HINTS = [
    'Do all the ordering with one `sort_values` over three keys, with a different direction per key: `ascending=[True, False, True]`.',
    'After sorting, `groupby("user").head(n)` keeps the first n rows of each group and keeps their order.',
    'Finish with `reset_index(drop=True)` so the index is 0..k-1, and select the columns in the required order.',
]

def _df(rows):
    return pd.DataFrame(rows, columns=['user', 'item', 'score'])

TESTS = [
    (_df([['u1', 'a', 0.9], ['u1', 'b', 0.5], ['u1', 'c', 0.7], ['u2', 'a', 0.1]]), 2),
    (_df([['u1', 'z', 1.0], ['u1', 'a', 1.0], ['u1', 'm', 1.0]]), 2),
    (_df([['b', 'x', 0.3], ['a', 'y', 0.2], ['b', 'w', 0.9], ['a', 'v', 0.8]]), 1),
    (_df([['u1', 'a', 0.5]]), 5),
    (_df([['u3', 'i1', 2.0], ['u1', 'i2', -1.0], ['u3', 'i3', 3.0], ['u1', 'i4', 0.0], ['u2', 'i5', 1.5], ['u3', 'i6', 2.0]]), 2),
    (_df([['u1', 'a', 0.1], ['u1', 'b', 0.2], ['u1', 'c', 0.3], ['u1', 'd', 0.4]]), 3),
]

EXPLANATION = r"""
## Sort once, then take the head of each group
Sorting by `(user, -score, item)` puts every user's rows in final display order. `groupby(...).head(n)` then only has to cut each group, and it preserves row order, so no second sort is needed. Cost: $O(N \log N)$ for the sort.

## Alternatives
- `df.groupby('user')['score'].rank(method='first', ascending=False) <= n` gives a boolean mask. It works, but the tie-break on `item` needs the pre-sort anyway.
- `nlargest` per group via `apply` is slower and harder to make deterministic on ties.

## Why deterministic ties matter
Recommendation logs, offline evaluation and A/B tests all compare lists. If ties resolve differently from run to run, metrics jitter and diffs show changes that didn't happen.

## Interview follow-ups
- Do the same in SQL with `ROW_NUMBER() OVER (PARTITION BY user ORDER BY score DESC, item)`.
- The table has 2B rows. How do you compute top-N per user without one global sort? (Partition by user, then keep a bounded heap per partition.)
"""
