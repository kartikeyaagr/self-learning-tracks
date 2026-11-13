import numpy as np

META = dict(title='Benchmark summary statistics', tags=['Probability and Statistics', 'Data Processing'], diff='Easy',
            runtime='numpy', lessons=['ieng:04.05', 'mlmaths:10.04'], related=['random-array-generation', 'row-extremes', 'winsorize'])

STATEMENT = r"""
You timed several model configurations, running each one repeatedly. The timings form a matrix `times` of shape `(runs, configs)`: row `r` is run number `r`, column `c` is configuration `c`.

Write `benchmark_summary(times)` that returns a tuple `(mean, std, median, p95)`, each a `float64` array of shape `(configs,)` with one value per **column**:

| Output | Definition |
|---|---|
| `mean` | arithmetic mean of the column |
| `std` | **sample** standard deviation, `ddof=1` |
| `median` | middle value (average of the two middle values for an even number of runs) |
| `p95` | 95th percentile with NumPy's default linear interpolation, `np.percentile(..., 95)` |

## Input
`times`: float array with `runs >= 2` and `configs >= 1`, no NaN.

No loops over columns: each statistic is one reduction along the right axis.
"""

STARTER = '''import numpy as np

def benchmark_summary(times):
    """times: (runs, configs). Return (mean, std ddof=1, median, p95), each shape (configs,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def benchmark_summary(times):
    t = np.asarray(times, dtype=float)
    # axis=0 reduces over runs (rows), leaving one number per configuration (column).
    return (t.mean(axis=0), t.std(axis=0, ddof=1),
            np.median(t, axis=0), np.percentile(t, 95, axis=0))
'''

HINTS = [
    'The `axis` argument names the axis that disappears. You want one result per column, so the rows must be collapsed.',
    'Every statistic takes `axis=0`. `np.std` defaults to `ddof=0`, the population formula, so pass `ddof=1` explicitly.',
    '`t.mean(axis=0)`, `t.std(axis=0, ddof=1)`, `np.median(t, axis=0)` and `np.percentile(t, 95, axis=0)`.',
]

_r = np.random.default_rng(61)
TESTS = [
    (np.array([[10.0, 1.0], [12.0, 3.0], [14.0, 2.0]]),),
    (np.array([[1.0], [2.0], [3.0], [100.0]]),),
    (np.array([[5.0, 5.0], [5.0, 5.0]]),),
    (np.array([[1e9, -3.0, 0.5], [1e9 + 4.0, 3.0, 0.25]]),),
    (np.arange(20.0).reshape(20, 1),),
    (_r.lognormal(3.0, 0.4, size=(25, 4)),),
    (_r.normal(100.0, 15.0, size=(8, 6)),),
]

EXPLANATION = r"""
## Why it works
For a 2-D array, `axis=0` means "combine along the rows", so a `(runs, configs)` input reduces to `(configs,)`: one summary per configuration. All four functions take the same `axis` argument, so the whole summary is four vectorised reductions.

- `ddof` ("delta degrees of freedom") sets the divisor to $n - \text{ddof}$. `ddof=1` gives the unbiased variance estimate for a sample.
- `np.percentile(x, q)` with the default `method='linear'` sorts the values, computes the fractional position $h = (n-1)\,q/100$ and interpolates between the neighbours. With 20 runs, the 95th percentile sits at $h = 18.05$, between the 19th and 20th smallest values.

## Complexity
Mean and std are $O(nc)$. Median and percentile need a partial sort per column, $O(nc)$ on average with `np.partition`-style selection.

## Common bugs
- `axis=1`, giving one value per run instead of per configuration.
- Leaving `ddof` at 0. For a handful of runs the difference is large (with 2 runs, the two estimates differ by a factor of $\sqrt{2}$).
- Passing `95` to `np.quantile`, which expects a fraction (`0.95`).

## Interview follow-ups
- Why report p95 or p99 latency rather than the mean? (Latency distributions are right-skewed. Users feel the tail, and one slow dependency call dominates fan-out requests.)
- With only 25 runs, how reliable is a p95 estimate, and how would you put an interval on it? (It rests on the top one or two observations. Bootstrap the percentile, or collect more runs.)
"""
