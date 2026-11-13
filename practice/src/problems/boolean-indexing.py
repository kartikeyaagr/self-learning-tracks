import numpy as np
import pandas as pd

META = dict(title='Filter orders with combined conditions', tags=['Data Processing'], diff='Easy', runtime='pandas',
            related=['multi-column-selection', 'resetting-index'])

STATEMENT = r"""
A finance team wants the orders that need a manual check. The table `df` has columns `id` (int), `region` (str), `amount` (float) and `refunded` (bool).

Write `orders_to_review(df, min_amount, regions)` that keeps the rows where **all** of these hold:

- `amount >= min_amount`;
- `region` is one of the strings in the list `regions`;
- the order is **not** refunded.

Return a DataFrame with the same columns in the same order, holding the kept rows in their original order. If nothing matches (for example when `regions` is empty) return a frame with those columns and no rows. Don't modify `df`.
"""

STARTER = '''import pandas as pd

def orders_to_review(df, min_amount, regions):
    """Rows with amount >= min_amount, region in regions, and not refunded."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def orders_to_review(df, min_amount, regions):
    # Element-wise &, | and ~ combine masks; each comparison needs its own parentheses.
    mask = (df['amount'] >= min_amount) & df['region'].isin(regions) & ~df['refunded']
    return df[mask]
'''

HINTS = [
    'Each condition is a boolean Series: `df["amount"] >= min_amount` has one True/False per row.',
    'Membership in a list is `df["region"].isin(regions)`. Negate a boolean Series with `~`, not `not`.',
    'Combine with `&`, wrapping comparisons in parentheses because `&` binds tighter than `>=`: `(a >= x) & b.isin(r) & ~c`. Then `df[mask]`.',
]

_r = np.random.default_rng(31)
_n = 25
_orders = pd.DataFrame({'id': np.arange(1, _n + 1), 'region': _r.choice(['EU', 'US', 'APAC', 'LATAM'], _n),
                        'amount': _r.uniform(0, 500, _n).round(2), 'refunded': _r.random(_n) < 0.3})


def _df(rows):
    return pd.DataFrame(rows, columns=['id', 'region', 'amount', 'refunded'])


TESTS = [
    (_df([[1, 'EU', 120.0, False], [2, 'US', 80.0, False], [3, 'EU', 300.0, True], [4, 'APAC', 150.0, False]]), 100.0, ['EU', 'APAC']),
    (_df([[1, 'EU', 100.0, False], [2, 'EU', 99.99, False]]), 100.0, ['EU']),
    (_df([[1, 'EU', 500.0, False]]), 0.0, []),
    (_df([[7, 'US', -20.0, False], [8, 'US', 0.0, False], [9, 'US', 5.0, True]]), -50.0, ['US']),
    (_df([[1, 'eu', 900.0, False], [2, 'EU', 900.0, False]]), 1.0, ['EU']),
    (_orders, 200.0, ['EU', 'US']),
    (_orders, 0.0, ['APAC', 'LATAM', 'EU', 'US']),
]

EXPLANATION = r"""
## Why it works
A comparison on a Series produces a boolean Series aligned to the rows. `&`, `|` and `~` combine such masks element by element, and `df[mask]` keeps the rows where the mask is True, in their original order and with their original labels.

## Complexity
$O(N)$ for the numeric comparisons; `isin` hashes the list once, so it is $O(N + |R|)$ rather than $O(N \cdot |R|)$.

## Common bugs
- `df.amount >= 100 & df.refunded`: without parentheses Python evaluates `100 & df.refunded` first.
- `and`, `or`, `not` on a Series raise "The truth value of a Series is ambiguous".
- `~df.refunded` is right only for a real bool column. On an int column of 0/1, `~` flips bits and gives -1 and -2, which are both truthy.
- `df.region == regions` compares element-wise with a list of the same length (or raises). Membership is `isin`.
- `>` instead of `>=` misses the boundary row in test 2. Matching is case-sensitive, as test 5 checks.

## Interview follow-ups
- Write the same filter with `df.query("amount >= @min_amount and region in @regions and not refunded")`. When is `query` nicer, and when is it a liability?
- How would you flag the rows instead of dropping them, adding a `needs_review` column? (Assign the mask: `df.assign(needs_review=mask)`.)
"""
