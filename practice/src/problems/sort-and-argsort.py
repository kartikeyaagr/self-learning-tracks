import numpy as np

META = dict(title='Ranks with averaged ties', tags=['Probability and Statistics', 'Data Processing'], diff='Medium',
            runtime='numpy', related=['aggregation-functions', 'concat-and-correlate', 'precision-recall-at-k'])

STATEMENT = r"""
Rank-based statistics (Spearman correlation, the Mann-Whitney test, AUC) replace each value by its **rank**. The smallest value gets rank 1, the next rank 2, and so on. Values that tie share the **average** of the ranks they would occupy.

Write `average_ranks(x)`.

## Input
`x`: a 1-D numeric array of length `n >= 1`, with no NaN.

## Output
A `float64` array of shape `(n,)` where `out[i]` is the rank of `x[i]`, between 1 and `n`. Tied values all receive the mean of their positions in sorted order.

## Example
```
x      = [40, 10, 30, 10, 50]
sorted = [10, 10, 30, 40, 50]   (positions 1..5)
ranks  = [ 4, 1.5,  3, 1.5,  5]
```
The two 10s occupy positions 1 and 2, so each gets 1.5.

## Requirements
Vectorise it with sorting tools (`argsort`, `unique`, cumulative sums). No Python loop over elements or tie groups.
"""

STARTER = '''import numpy as np

def average_ranks(x):
    """1-based ranks of x with ties averaged. float64 (n,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def average_ranks(x):
    x = np.asarray(x)
    # Distinct values come back sorted; inverse[i] says which distinct value x[i] is; counts gives each group's size.
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)                 # 1-based sorted position of each group's last member
    first = last - counts + 1                # ... and of its first member
    return ((first + last) / 2.0)[inverse]   # the average of consecutive integers is the midpoint
'''

HINTS = [
    'Without ties, the rank of `x[i]` is its position in sorted order plus 1. `np.argsort` gives the sorted order, and you need its inverse: the position of each element.',
    'With ties, think in groups of equal values. `np.unique(x, return_inverse=True, return_counts=True)` tells you each element\'s group and the size of every group, in sorted order.',
    'A group with `count` members ending at sorted position `last` (a cumulative sum of counts) spans positions `last - count + 1 .. last`. Its average rank is the midpoint of those two. Index the per-group ranks with `inverse`.',
]

_r = np.random.default_rng(67)
TESTS = [
    (np.array([40, 10, 30, 10, 50]),),
    (np.array([3.0, 1.0, 2.0]),),
    (np.array([7, 7, 7, 7]),),
    (np.array([42]),),
    (np.array([-1e12, 1e12, 0.0, -1e12, 1e12, 1e12]),),
    (np.array([5, 4, 3, 2, 1]),),
    (_r.integers(0, 6, size=20),),
    (_r.normal(size=15),),
]

EXPLANATION = r"""
## Why it works
`np.unique` sorts the distinct values and returns, for each input element, the index of its distinct value (`inverse`) plus how many times each value occurs (`counts`). In sorted order the groups sit back to back, so the cumulative sum of `counts` is the 1-based position of each group's last member, and subtracting `count - 1` gives its first. The average of the consecutive integers `first .. last` is $(\text{first} + \text{last})/2$. Indexing that per-group vector with `inverse` scatters each group's rank back to the original positions.

## The argsort route
Without ties, ranks are the **inverse permutation** of `argsort`:

```
order = np.argsort(x, kind='stable')
ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
```

Note that `np.argsort(np.argsort(x)) + 1` computes the same thing. Ties then need a second pass to average within groups, which is what the `unique` route does in one step.

## Complexity
$O(n \log n)$ for the sort inside `np.unique`; everything else is $O(n)$.

## Common bugs
- Returning `argsort(x) + 1`. That is the order (which element is smallest), not the rank (where each element lands).
- Giving ties the minimum rank (`1, 1, 3`, as in sports rankings) or the order they appear (`1, 2, 3`). Both are valid conventions but not the average.
- 0-based ranks.

## Interview follow-ups
- How do you compute Spearman's rank correlation from this? (Pearson correlation of `average_ranks(x)` and `average_ranks(y)`.)
- How would you rank each row of a 2-D matrix independently? (`argsort` along `axis=1` and scatter with `np.put_along_axis`; tie averaging needs a per-row group id, for example by offsetting each row's values.)
"""
