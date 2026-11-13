import numpy as np

META = dict(title='Quantiles by linear interpolation', tags=['Probability and Statistics'], diff='Easy', runtime='numpy',
            lessons=['ieng:04.05'], related=['mean-median-mode', 'bootstrap-confidence-interval'])

STATEMENT = r"""
Latency dashboards report p50, p95 and p99: quantiles of the observed values. Implement the most common definition yourself.

Sort the data into $x_{(0)} \le x_{(1)} \le \dots \le x_{(n-1)}$. For a level $q \in [0, 1]$, compute the fractional position $h = (n-1)\,q$ and interpolate between its neighbours:

$$Q(q) = x_{(\lfloor h \rfloor)} + (h - \lfloor h \rfloor)\,\big(x_{(\lceil h \rceil)} - x_{(\lfloor h \rfloor)}\big)$$

Write `quantiles(x, qs)`.

## Requirements
- `x`: non-empty 1-D float array. `qs`: 1-D float array of levels in $[0, 1]$, in any order, possibly empty.
- Return a float array with the same shape as `qs`, where entry `i` is $Q(q_i)$.
- Vectorise over `qs`. Don't call `np.quantile` or `np.percentile` (this definition matches their default, so use them to check your work).
"""

STARTER = '''import numpy as np

def quantiles(x, qs):
    """x: (n,), qs: (k,) levels in [0, 1]. Return (k,) float array."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def quantiles(x, qs):
    xs = np.sort(np.asarray(x, dtype=float))
    h = (len(xs) - 1) * np.asarray(qs, dtype=float)
    lo = np.floor(h).astype(int)
    hi = np.minimum(lo + 1, len(xs) - 1)           # at q = 1 there is no right neighbour; frac is 0 there anyway
    frac = h - lo
    return xs[lo] + frac * (xs[hi] - xs[lo])
'''

HINTS = [
    'Sort once, then every quantile is a lookup plus an interpolation.',
    'Compute `h = (n - 1) * qs` for all levels at once. `np.floor(h).astype(int)` gives the left index and `h - floor(h)` the weight.',
    'Use `min(lo + 1, n - 1)` as the right index so `q = 1` (and `n = 1`) never index past the end. The weight is 0 in those cases.',
]

_r = np.random.default_rng(95)
TESTS = [
    (np.array([10.0, 20.0, 30.0, 40.0]), np.array([0.0, 0.5, 1.0])),
    (np.array([7.0, 1.0, 3.0]), np.array([0.25, 0.75])),
    (np.array([42.0]), np.array([0.0, 0.3, 1.0])),
    (np.array([5.0, 5.0, 5.0, 1.0, 9.0]), np.array([0.1, 0.5, 0.9])),
    (np.array([-2.0, -8.0, 4.0]), np.array([])),
    (np.array([1.0, 2.0]), np.array([0.99, 0.01, 0.5])),
    (_r.exponential(30.0, size=200), np.array([0.5, 0.9, 0.95, 0.99, 0.999])),
]

EXPLANATION = r"""
## Why interpolate
With $n$ sorted points, the $k$-th one is treated as sitting at level $k/(n-1)$, so the minimum is the 0-quantile and the maximum the 1-quantile. Between two points, the quantile function is a straight line. This makes $Q$ continuous and monotone in $q$, and $Q(0.5)$ equals the usual median.

There are at least nine conventions (NumPy's `method=` lists them). They differ mostly in the tails with small $n$, which is exactly where p99 lives. Always say which one a dashboard uses.

## Complexity
$O(n \log n)$ for the sort, then $O(k)$ for $k$ levels. For one quantile, a selection algorithm (`np.partition`) is $O(n)$.

## Common bugs
- Using $h = nq$ instead of $(n-1)q$, which shifts every quantile and indexes out of range at $q=1$.
- Forgetting to sort.
- Indexing `lo + 1` without clamping, which crashes when $n = 1$.

## Interview follow-ups
- p99 latency over a billion requests per day: how do you compute it without storing every value? (A quantile sketch such as t-digest or KLL, or fixed histogram buckets.)
- Can you average p99s from ten servers to get the fleet p99? (No. Quantiles don't average. Merge the sketches or histograms instead.)
"""
