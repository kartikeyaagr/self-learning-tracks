import numpy as np
import pandas as pd

META = dict(title='Enrich orders from two lookup tables', tags=['Data Processing'], diff='Medium', runtime='pandas',
            lessons=['sdsd:03.01'], related=['setting-index', 'concatenate-dataframes', 'sql-running-total'])

STATEMENT = r"""
Orders reference customers and products by id. Join them into one reporting table, choosing the right join for each lookup.

| frame | columns |
|---|---|
| `orders` | `order_id` (int, unique), `customer_id` (int), `sku` (str), `qty` (int) |
| `customers` | `customer_id` (int, unique), `name` (str), `tier` (str) |
| `products` | `sku` (str, unique), `price` (float) |

Write `enrich_orders(orders, customers, products)` that returns a DataFrame with columns `order_id, name, tier, sku, qty, line_total`, where:

- an order whose `sku` is **not** in `products` is dropped (it cannot be priced);
- an order whose `customer_id` is **not** in `customers` is kept, with `name` and `tier` set to `"unknown"`;
- `line_total = qty * price`;
- rows are sorted by `order_id` ascending.

Any frame may have zero rows. Don't modify the inputs.
"""

STARTER = '''import pandas as pd

def enrich_orders(orders, customers, products):
    """Inner-join products, left-join customers ('unknown' if missing). Columns: order_id, name, tier, sku, qty, line_total."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def enrich_orders(orders, customers, products):
    # validate= raises if a lookup key is unexpectedly duplicated, which would multiply orders.
    out = orders.merge(products, on='sku', how='inner', validate='many_to_one')
    out = out.merge(customers, on='customer_id', how='left', validate='many_to_one')
    out[['name', 'tier']] = out[['name', 'tier']].fillna('unknown')
    out['line_total'] = out['qty'] * out['price']
    cols = ['order_id', 'name', 'tier', 'sku', 'qty', 'line_total']
    return out.sort_values('order_id')[cols].reset_index(drop=True)
'''

HINTS = [
    '`a.merge(b, on="key", how=...)` is a SQL join. Decide per lookup: dropping unmatched orders is `"inner"`, keeping them is `"left"`.',
    'After the left join, unmatched customers have NaN in `name` and `tier`. Fill those two columns, and only those, with `"unknown"`.',
    'Merging does not keep the original row order in every case, so sort by `order_id` at the end and select the columns in the required order.',
]

_r = np.random.default_rng(5150)
_cust = pd.DataFrame({'customer_id': np.arange(1, 9), 'name': [f'c{i}' for i in range(1, 9)],
                      'tier': _r.choice(['gold', 'silver', 'bronze'], 8)})
_prod = pd.DataFrame({'sku': [f'P{i}' for i in range(6)], 'price': _r.integers(100, 5000, 6) / 100})
_ord = pd.DataFrame({'order_id': _r.permutation(np.arange(100, 130)), 'customer_id': _r.integers(1, 11, 30),
                     'sku': _r.choice([f'P{i}' for i in range(8)], 30), 'qty': _r.integers(1, 6, 30)})


def _o(rows):
    return pd.DataFrame(rows, columns=['order_id', 'customer_id', 'sku', 'qty'])


def _c(rows):
    return pd.DataFrame(rows, columns=['customer_id', 'name', 'tier']).astype({'customer_id': int})


def _p(rows):
    return pd.DataFrame(rows, columns=['sku', 'price']).astype({'price': float})


TESTS = [
    (_o([[2, 10, 'A', 3], [1, 11, 'B', 1], [3, 99, 'A', 2]]), _c([[10, 'ann', 'gold'], [11, 'bob', 'basic']]),
     _p([['A', 2.5], ['B', 10.0]])),
    (_o([[1, 10, 'A', 1], [2, 10, 'Z', 5]]), _c([[10, 'ann', 'gold']]), _p([['A', 4.0]])),
    (_o([[5, 1, 'A', 2]]), _c([]), _p([['A', 1.25]])),
    (_o([[5, 1, 'A', 2]]), _c([[1, 'x', 'y']]), _p([])),
    (_o([[7, 1, 'A', 0], [6, 2, 'A', 1000000], [8, 1, 'B', 3]]), _c([[1, 'p', 'gold'], [2, 'q', 'gold'], [3, 'r', 'unused']]),
     _p([['A', 0.01], ['B', 7.0], ['C', 1.0]])),
    (_ord, _cust, _prod),
]

EXPLANATION = r"""
## Why it works
Each lookup answers a different business question, so it gets a different join:

| lookup | rule | join |
|---|---|---|
| products | no price, no line total: drop | `inner` |
| customers | unknown buyer is still revenue: keep | `left` |

Both lookups are **many-to-one**: many orders per customer or product, at most one row per key in the lookup table. That guarantee is what keeps the row count at most `len(orders)`. `validate="many_to_one"` turns the assumption into a check, so a duplicated product row fails loudly instead of silently doubling revenue.

## Complexity
Hash joins: $O(N + M)$ per merge for $N$ orders and $M$ lookup rows, plus $O(N \log N)$ to sort.

## Common bugs
- One `how="left"` for both joins: unpriceable orders survive with `line_total = NaN` (test 2).
- One `how="inner"` for both: orders from unknown customers disappear (test 1).
- `out.fillna("unknown")` on the whole frame also writes the string into numeric columns such as `price`.
- Duplicate keys in a lookup table multiply rows. Run `validate=` or check `key.is_unique` first.
- Relying on merge output order. Sort explicitly.

## Interview follow-ups
- Return the orders that were dropped and why, in one pass. (`merge(..., how="left", indicator=True)` and filter on `_merge == "left_only"`.)
- The products table has price history with `valid_from` dates. How do you join each order to the price in force on its date? (`pd.merge_asof` by sku on the date.)
"""
