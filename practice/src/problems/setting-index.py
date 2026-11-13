import numpy as np
import pandas as pd

META = dict(title='Align two price snapshots by key', tags=['Data Processing'], diff='Medium', runtime='pandas',
            related=['loc-vs-iloc', 'merge-dataframes', 'resetting-index'])

STATEMENT = r"""
You have yesterday's and today's price lists, `old` and `new`, each with columns `sku` (str) and `price` (float). The rows are in no particular order, and a product can appear in only one of them (it was added or discontinued).

Write `price_changes(old, new)` that compares the two **by `sku`**, not by row position. Return a DataFrame with columns, in this order:

| column | value |
|---|---|
| `sku` | every sku that appears in `old` or `new` |
| `old_price` | the price in `old`, NaN if absent |
| `new_price` | the price in `new`, NaN if absent |
| `change` | `new_price - old_price`, NaN if either is absent |

Rows sorted by `sku` ascending.

## Input
- Within each frame, `sku` values are unique. Either frame may have zero rows.

The intended tool is the index: once `sku` is the index of both prices, arithmetic between them lines up rows by label automatically.
"""

STARTER = '''import pandas as pd

def price_changes(old, new):
    """Compare prices by sku. Columns: sku, old_price, new_price, change; sorted by sku."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def price_changes(old, new):
    o = old.set_index('sku')['price']
    n = new.set_index('sku')['price']
    # Building a frame from Series aligns them on the union of their index labels.
    out = pd.DataFrame({'old_price': o, 'new_price': n})
    out['change'] = out['new_price'] - out['old_price']
    out = out.sort_index().rename_axis('sku').reset_index()
    return out
'''

HINTS = [
    'Subtracting two columns from different frames lines them up by **index label**, not by position. Make `sku` the label with `set_index("sku")`.',
    'Turn each frame into a price Series indexed by sku. `pd.DataFrame({"old_price": o, "new_price": n})` aligns both on the union of skus and fills gaps with NaN.',
    'Compute `change` as a column difference (NaN propagates by itself), `sort_index()`, then bring `sku` back as a column with `reset_index()`; make sure the index is named `sku` first.',
]

_r = np.random.default_rng(42)
_skus = [f'SKU{i:03d}' for i in range(30)]
_old = pd.DataFrame({'sku': _r.permutation(_skus[:24]), 'price': _r.uniform(1, 100, 24).round(2)})
_new = pd.DataFrame({'sku': _r.permutation(_skus[6:]), 'price': _r.uniform(1, 100, 24).round(2)})


def _p(rows):
    return pd.DataFrame(rows, columns=['sku', 'price']).astype({'price': float})


TESTS = [
    (_p([['b', 2.0], ['a', 1.0], ['c', 5.0]]), _p([['a', 1.5], ['c', 4.0], ['d', 9.0]])),
    (_p([['x', 10.0], ['y', 20.0]]), _p([['y', 25.0], ['x', 10.0]])),
    (_p([]), _p([['new1', 3.0]])),
    (_p([['gone', 7.5]]), _p([])),
    (_p([['k', 1e9], ['m', -0.5]]), _p([['m', 0.5], ['k', 1e9 + 1]])),
    (_old, _new),
]

EXPLANATION = r"""
## Why it works
pandas aligns on the index before any element-wise operation. With `sku` as the index, `n - o` (or a DataFrame built from both Series) matches each sku with its counterpart wherever it sits, takes the **union** of labels, and puts NaN where a label exists on one side only. NaN then propagates through the subtraction, which is exactly the "absent" rule. Without the index, `new["price"] - old["price"]` would subtract row 0 from row 0, comparing unrelated products (test 2 is the same products in a different order).

## Complexity
$O(N \log N)$: alignment uses a hash join on the index, and the final sort dominates.

## Common bugs
- Subtracting the raw columns: alignment then happens on the default `0..n-1` index, i.e. by position.
- `reset_index()` on an unnamed index creates a column called `index`. Name it with `rename_axis("sku")` (or rename afterwards).
- `set_index` on a frame with duplicate skus still "works", but alignment then multiplies rows. Check `index.is_unique` (or use `verify_integrity=True`) when the data isn't guaranteed clean.
- Filling missing prices with 0 makes a new product look like a price increase from 0.

## Interview follow-ups
- Write the same thing with `pd.merge(old, new, on="sku", how="outer", suffixes=("_old", "_new"))`. When is an index join cheaper than repeated merges?
- Return only skus whose price moved by more than 10%, treating new and removed products separately.
"""
