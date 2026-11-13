import numpy as np
import pandas as pd

META = dict(title='Stack batches with schema drift', tags=['Data Processing', 'MLOps'], diff='Medium', runtime='pandas',
            lessons=['sdsd:03.12'], related=['merge-dataframes', 'drop-duplicates', 'rename-columns'])

STATEMENT = r"""
A feature pipeline writes one DataFrame per daily batch. Over time columns were added and removed, and some ids were re-sent with corrected values in a later batch.

`batches` is a list of DataFrames, oldest first. Each has a column `id` (str, unique within that batch) and any number of numeric columns. Write `stack_batches(batches)` that returns one DataFrame with:

- a first column `batch` (int): the position of the row's source frame in `batches`, starting at 0;
- then `id`;
- then every other column, in the order it **first appears** scanning the batches in order and each batch's columns left to right;
- NaN where a row's batch lacks a column;
- for an `id` that occurs in several batches, **only the row from the latest batch**;
- rows ordered by `batch`, then by their order within the batch.

## Input
`batches` has at least one frame, and every frame has at least one row.

## Example
```
batch 0:  id  a        batch 1:  id  b    a
          x   1                  y   5  2.0
          y   2                                   ->   batch id    a    b
                                                           0  x  1.0  NaN
                                                           1  y  2.0  5.0
```
"""

STARTER = '''import pandas as pd

def stack_batches(batches):
    """Concatenate batches (oldest first): batch, id, other columns by first appearance; latest row per id wins."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def stack_batches(batches):
    # Outer concat: the column set is the union, in order of first appearance; gaps become NaN.
    out = pd.concat([b.assign(batch=i) for i, b in enumerate(batches)], ignore_index=True)
    out = out.drop_duplicates('id', keep='last')        # rows are in batch order, so last = latest
    rest = [c for c in out.columns if c not in ('batch', 'id')]
    return out[['batch', 'id'] + rest].reset_index(drop=True)
'''

HINTS = [
    '`pd.concat(list_of_frames, ignore_index=True)` stacks rows. Columns are matched **by name**, and a column missing from one frame becomes NaN for its rows.',
    'Tag each frame before stacking: `b.assign(batch=i)` inside a comprehension over `enumerate(batches)`.',
    'After stacking, rows are already in batch order, so `drop_duplicates("id", keep="last")` keeps the latest version of each id. Then reorder the columns as `["batch", "id", ...rest]`.',
]

_r = np.random.default_rng(31415)
_pool = ['f_age', 'f_clicks', 'f_spend', 'f_days', 'f_ratio']
_rand = []
for _i in range(4):
    _cols = [c for c in _pool if _r.random() < 0.7] or ['f_age']
    _ids = _r.choice([f'id{k:02d}' for k in range(12)], int(_r.integers(2, 7)), replace=False)
    _rand.append(pd.DataFrame({'id': _ids, **{c: _r.integers(0, 50, len(_ids)) for c in _r.permutation(_cols)}}))

TESTS = [
    ([pd.DataFrame({'id': ['x', 'y'], 'a': [1, 2]}), pd.DataFrame({'id': ['y'], 'b': [5], 'a': [2.0]})],),
    ([pd.DataFrame({'id': ['p'], 'v': [1.5]}), pd.DataFrame({'id': ['q'], 'v': [2.5]}), pd.DataFrame({'id': ['p'], 'v': [9.5]})],),
    ([pd.DataFrame({'id': ['only'], 'z': [0], 'y': [1]})],),
    ([pd.DataFrame({'id': ['a', 'b'], 'm': [1, 2]}), pd.DataFrame({'id': ['c'], 'n': [3]}), pd.DataFrame({'id': ['b', 'a'], 'o': [4, 5]})],),
    ([pd.DataFrame({'id': ['k'], 'big': [1e15]}), pd.DataFrame({'id': ['k'], 'big': [-1e15], 'new': [0.5]})],),
    (_rand,),
]

EXPLANATION = r"""
## Why it works
`pd.concat` aligns frames **by column label**, not by position, and takes the union of labels (an outer join on the columns). With the default `sort=False`, the union keeps the order in which labels are first seen, which is exactly the required column order. Cells for columns a batch did not have are NaN, which forces those columns to float.

Tagging rows with their batch before stacking keeps provenance, and because the stacked rows are in batch order, "latest wins" is just `drop_duplicates(keep="last")`.

## Complexity
$O(\text{total cells})$ to stack, plus $O(N)$ hashing for the deduplication.

## Common bugs
- `ignore_index=False` keeps every batch's own `0..k-1` index, so labels repeat. Harmless here, but surprising later with `.loc`.
- `pd.concat(..., axis=1)` puts frames side by side instead of stacking them.
- `keep="first"` keeps the stale version of a re-sent id (tests 2 and 5).
- `sort=True`, or sorting columns afterwards, breaks the first-appearance order (test 3).
- Building the frame with `DataFrame.append` in a loop: removed in pandas 2, and quadratic anyway.

## Interview follow-ups
- A column was **renamed** between batches (`f_spend` became `spend_usd`). How do you stop the stack from silently holding two half-empty columns? (Map names to a canonical schema before concatenating, and fail on unknown columns.)
- Do you keep the dropped stale rows anywhere? Discuss append-only storage with a "latest view" versus overwriting in place.
"""
