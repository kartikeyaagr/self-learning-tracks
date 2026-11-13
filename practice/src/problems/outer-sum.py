import numpy as np

META = dict(title='Pair totals under a budget', tags=['Data Processing'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.03'], related=['pairwise-differences', 'row-scaling'])

STATEMENT = r"""
A shop sells `m` mains and `n` sides, with prices `a` and `b`. Write `pair_totals(a, b, budget)` that prices **every** (main, side) combination at once and counts how many fit the budget.

## Input
- `a`: float array of shape `(m,)`; `b`: float array of shape `(n,)`, with `m, n >= 1`.
- `budget`: a Python float.

## Output
A tuple `(totals, count)`:
- `totals`: `float64` array of shape `(m, n)` with `totals[i, j] = a[i] + b[j]`.
- `count`: Python `int`, the number of pairs with `totals[i, j] <= budget`.

Use broadcasting; no Python loops.

## Example
`a = [3, 5]`, `b = [1, 2, 4]`, `budget = 6`:

```
totals = [[4., 5., 7.],
          [6., 7., 9.]]
count  = 3
```
"""

STARTER = '''import numpy as np

def pair_totals(a, b, budget):
    """Return (totals (m, n) with a[i] + b[j], count of totals <= budget as int)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pair_totals(a, b, budget):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    totals = a[:, None] + b[None, :]          # (m, 1) + (1, n) -> (m, n)
    return totals, int((totals <= budget).sum())
'''

HINTS = [
    'Adding a `(m,)` array to an `(n,)` array fails or gives the wrong thing. Give them different axes so broadcasting stretches each one along the other.',
    '`a[:, None]` has shape `(m, 1)` and `b[None, :]` has shape `(1, n)`. Their sum is `(m, n)`. `np.add.outer(a, b)` does the same.',
    'Counting is a boolean reduction: `(totals <= budget).sum()`, converted with `int(...)`.',
]

_r = np.random.default_rng(31)
TESTS = [
    (np.array([3.0, 5.0]), np.array([1.0, 2.0, 4.0]), 6.0),
    (np.array([1.0]), np.array([1.0]), 1.5),
    (np.array([2.0, 2.0]), np.array([0.5]), 2.5),
    (np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 1.0]), 0.0),
    (np.array([1e15, 2e15]), np.array([1.0, -1e15]), 1e15),
    (np.array([10.0, 20.0]), np.array([30.0, 40.0]), 5.0),
    (_r.uniform(5, 20, size=6).round(2), _r.uniform(1, 8, size=4).round(2), 18.0),
]

EXPLANATION = r"""
## Why it works
Broadcasting compares shapes from the right. `(m, 1)` against `(1, n)`: each size-1 axis is stretched to match the other, so the sum behaves as if `a` were repeated across `n` columns and `b` down `m` rows, without materialising either copy. The result is the full table of pair sums, the "outer sum".

The comparison `totals <= budget` broadcasts a scalar, giving an `(m, n)` boolean table. `True` counts as 1 when summed.

## Complexity
$O(mn)$ time and memory, unavoidable since the table itself has $mn$ entries.

## Common bugs
- `a + b` with `m == n` runs without error and returns element-wise sums, shape `(m,)`, which is not what you want. Shape bugs that don't crash are the dangerous ones.
- `a[None, :] + b[:, None]` gives the transpose, `(n, m)`.
- Returning `np.int64` where the statement asks for an `int`. The grader accepts it, but JSON serialisers and type checks elsewhere often don't.

## Interview follow-ups
- How would you count pairs within budget **without** building the `(m, n)` table? (Sort `b`; for each `a[i]`, `np.searchsorted(b, budget - a[i], side='right')` counts the valid sides: $O((m+n)\log n)$ time and no table.)
- Generalise to an outer product or outer max. (`a[:, None] * b`, `np.maximum.outer(a, b)`; every binary ufunc has `.outer`.)
"""
