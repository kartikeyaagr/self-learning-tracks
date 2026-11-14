import numpy as np

META = dict(title='Rank and Gaussian-rank transforms', tags=['Feature Engineering'], diff='Easy', runtime='numpy',
            related=['quantile-binning', 'feature-scaling', 'rank-and-dense-rank'])

STATEMENT = r"""
Write `rank_transform(x, method)` for a 1-D float array.

- `method="average"`: ranks $1, \dots, n$ in increasing order, with tied values all receiving the **average** of the ranks they would occupy (so `[10, 20, 20, 30]` gives `[1, 2.5, 2.5, 4]`).
- `method="pct"`: the average rank divided by $n$.
- `method="gauss"`: the Gaussian-rank transform $\Phi^{-1}\big((r - 0.5)/n\big)$ where $r$ is the average rank and $\Phi^{-1}$ is the standard normal quantile function. Implement $\Phi^{-1}$ with `statistics.NormalDist().inv_cdf`.

Return a float array of the same length.
"""

STARTER = '''import numpy as np

def rank_transform(x, method):
    """Return average ranks, percentile ranks or Gaussian ranks."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from statistics import NormalDist

def rank_transform(x, method):
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="stable")
    r = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    if method == "pct":
        return r / n
    if method == "gauss":
        inv = NormalDist().inv_cdf
        return np.array([inv((v - 0.5) / n) for v in r])
    return r
'''

HINTS = [
    'Sort once with `argsort`, then walk through runs of equal values in sorted order.',
    'A run occupying sorted positions i..j (0-based) gets rank (i + j) / 2 + 1.',
    'For the Gaussian version, (r - 0.5) / n keeps the argument strictly between 0 and 1, so the normal quantile is finite.',
]

_r = np.random.default_rng(12)
TESTS = [
    (np.array([10.0, 20.0, 20.0, 30.0]), "average"),
    (np.array([3.0, 1.0, 2.0]), "pct"),
    (np.array([5.0, 5.0, 5.0]), "average"),
    (np.array([1.0, 100.0, 1000.0, 1e6]), "gauss"),
    (_r.exponential(1.0, 9), "gauss"),
    (np.array([2.0, 1.0, 2.0, 1.0, 3.0]), "pct"),
]

EXPLANATION = r"""
## What ranks buy you
A rank transform keeps only the order of values, so outliers and skew disappear: the largest value becomes rank $n$ whether it is 1,000 or a billion. The Gaussian-rank version then maps the ranks onto a bell curve, which neural networks and linear models often train on more easily.

## Ties
Averaging tied ranks keeps the transform symmetric and matches the convention used by Spearman correlation. Dense ranking (1, 2, 2, 3) is another valid choice for other purposes.

## Applying to new data
Ranks are relative to a dataset. For test data, interpolate each new value's position within the sorted training values, rather than ranking the test set on its own.

## Complexity
$O(n \log n)$ for the sort.

## Interview follow-ups
- How is Spearman correlation related to this transform?
"""
