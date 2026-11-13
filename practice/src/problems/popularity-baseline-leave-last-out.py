import pandas as pd

META = dict(title='Leave-last-out evaluation of a popularity baseline', tags=['Recommender Systems', 'Data Processing', 'Metrics & Evaluation'],
            diff='Hard', runtime='pandas', lessons=['sdsd:09.09', 'sdsd:16.05'],
            related=['top-n-per-group', 'hit-rate-and-mrr', 'catalog-coverage-and-novelty'])

STATEMENT = r"""
Before trusting any recommender, compare it to "recommend the most popular items you haven't seen". Build that baseline and its offline evaluation end to end.

Write `popularity_loo(df, k)` where `df` has columns `user` (str), `item` (str) and `ts` (int timestamp), one row per interaction.

## 1. Split
- Order each user's rows by `ts` ascending, ties by `item` ascending. A user's **last** row in that order is their held-out test interaction.
- Users with fewer than 2 rows are **not evaluated**; all their rows stay in training.
- Training data is every row that isn't held out.

## 2. Popularity
An item's popularity is the number of **distinct users** who have it in the training data (repeat interactions by one user count once). Only items that appear in training are candidates.

## 3. Recommend
For each evaluated user, the list is the `k` most popular candidates, by popularity descending, ties by `item` ascending, **excluding every item the user has in training**. The list may be shorter than `k` if candidates run out.

## 4. Return
A DataFrame with columns `user`, `held_out`, `rank`, one row per evaluated user, sorted by `user` ascending (string order), with a fresh index:
- `held_out`: the user's held-out item.
- `rank`: the 1-based position of `held_out` in the user's list, or `0` if it isn't there, as `int64`. A held-out item the user already has in training can never be in the list, so it gets `0`.

At least one user has 2 or more rows. Don't modify `df`.
"""

STARTER = '''import pandas as pd

def popularity_loo(df, k):
    """Return DataFrame[user, held_out, rank] (see statement)."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd
from itertools import islice

def popularity_loo(df, k):
    d = df.sort_values(['user', 'ts', 'item'], kind='mergesort').reset_index(drop=True)
    size = d.groupby('user')['item'].transform('size')
    is_last = ~d.duplicated('user', keep='last')            # last row per user in (ts, item) order
    test_mask = is_last & (size >= 2)
    test, train = d[test_mask], d[~test_mask]

    pop = train.drop_duplicates(['user', 'item']).groupby('item').size().rename('n').reset_index()
    ranked = pop.sort_values(['n', 'item'], ascending=[False, True])['item'].tolist()
    seen = train.groupby('user')['item'].agg(set)

    rows = []
    for user, target in zip(test['user'], test['item']):
        have = seen[user]                                   # evaluated users always have training rows
        top = list(islice((it for it in ranked if it not in have), k))
        rows.append((user, target, top.index(target) + 1 if target in top else 0))
    out = pd.DataFrame(rows, columns=['user', 'held_out', 'rank'])
    out['rank'] = out['rank'].astype('int64')
    return out.sort_values('user').reset_index(drop=True)
'''

HINTS = [
    'Sort once by `user, ts, item`. Then `~d.duplicated("user", keep="last")` marks each user\'s last row, and `groupby("user").transform("size")` tells you who has at least 2 rows.',
    'Popularity counts distinct users: `drop_duplicates(["user", "item"])` before `groupby("item").size()`. Sort the result by count descending, item ascending, once for everybody.',
    'Per user, walk the global ranking and skip items in their training set until you have `k`. The held-out item can be one they already had, in which case it is skipped too and the rank is 0.',
]

def _df(rows):
    return pd.DataFrame(rows, columns=['user', 'item', 'ts'])

import random as _random
_g = _random.Random(41)
_items = [f'i{n:02d}' for n in range(15)]
_w = [1 / (n + 1) for n in range(15)]
_rows = [[f'u{_g.randrange(30)}', _g.choices(_items, _w)[0], _g.randrange(1000)] for _ in range(160)]

TESTS = [
    (_df([['a', 'x', 1], ['a', 'y', 4], ['b', 'y', 2], ['b', 'z', 3], ['b', 'x', 6], ['c', 'z', 1]]), 2),
    (_df([['u1', 'p', 10], ['u1', 'q', 10], ['u2', 'p', 1], ['u2', 'r', 2], ['u3', 'q', 4], ['u3', 'r', 9]]), 1),
    (_df([['a', 'x', 1], ['a', 'x', 2], ['b', 'x', 1], ['b', 'y', 3]]), 3),
    (_df([['solo', 'x', 1], ['b', 'x', 2], ['b', 'y', 3], ['c', 'y', 1], ['c', 'z', 2]]), 5),
    (_df([['b', 'x', 1], ['b', 'y', 2], ['a', 'y', 1], ['a', 'z', 2], ['a', 'x', 3], ['c', 'x', 7], ['c', 'w', 8]]), 1),
    (_df([['u10', 'a', 1], ['u10', 'b', 2], ['u2', 'a', 1], ['u2', 'b', 2], ['u2', 'c', 3]]), 2),
    (_df(_rows), 3),
    (_df(_rows), 10),
]

EXPLANATION = r"""
## Why it works
Sorting by `(user, ts, item)` fixes a total order inside each user, so "last" is well defined even when two events share a timestamp (test 2). Splitting by time per user mimics deployment: the model only knows the past and must predict the next interaction. Counting distinct users makes popularity robust to one user replaying an item many times (test 3: user `a` has `x` twice). Because the global ranking is the same for everyone, it is computed once; each user's list is that ranking with their own training items skipped.

## Complexity
$O(N \log N)$ for the sorts, plus $O(U \cdot (k + s_u))$ for the per-user walks, where $s_u$ is how many of the top items a user already has.

## Common bugs
- Splitting by global time instead of per user, or by row order without sorting by `ts`.
- Computing popularity on **all** data, test rows included: the held-out items get a popularity boost they can't have at prediction time (label leakage). This alone can make the baseline look much stronger.
- Not excluding seen items, so the list is full of things the user already has.
- Sorting users as numbers in your head: the required order is string order, so `u10` sorts before `u2` (test 6).

## Interview follow-ups
- Your model beats popularity by 30% on this protocol but loses online. What could be wrong? (Leakage in features, a popularity baseline computed with a different time window, users with long histories dominating the average, or the offline protocol ignoring position bias.)
- How would you turn the output into HR@k and MRR@k, and report them separately for users with short and long histories? (`(rank > 0).mean()`, `(1 / rank).where(rank > 0, 0).mean()`; join history length and group by buckets.)
"""
