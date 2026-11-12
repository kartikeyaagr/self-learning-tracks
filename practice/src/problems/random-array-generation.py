import numpy as np

META = dict(title='Seeded bootstrap of the mean', tags=['Probability and Statistics', 'Data Processing'], diff='Easy',
            runtime='numpy', lessons=['mlmaths:12.01'], related=['aggregation-functions', 'fancy-indexing'])

STATEMENT = r"""
The bootstrap estimates how much a statistic would vary across datasets by resampling the one dataset you have. Each bootstrap replicate draws $n$ indices **with replacement** from the $n$ observations and recomputes the statistic.

Write `bootstrap_means(x, n_boot, seed)` that returns `(means, se)`.

## Exact procedure (it decides the random numbers, so follow it exactly)
1. `rng = np.random.default_rng(seed)`
2. Draw every index in **one** call: `idx = rng.integers(0, n, size=(n_boot, n))`, where `n = len(x)`. Row `b` holds the indices for replicate `b`.
3. `means[b]` is the mean of `x` at the indices in row `b`.
4. `se` is the sample standard deviation of `means` with `ddof=1` (the bootstrap standard error).

## Shapes and types
- `x`: 1-D float array with `n >= 1`. `n_boot >= 2` and `seed` are Python ints.
- `means`: `float64` array of shape `(n_boot,)`. `se`: a Python `float`.
- Use NumPy only, with no loop over replicates.
"""

STARTER = '''import numpy as np

def bootstrap_means(x, n_boot, seed):
    """Return (means (n_boot,), se float) from a seeded bootstrap of the mean."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def bootstrap_means(x, n_boot, seed):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))   # high is exclusive: values 0 .. n-1
    means = x[idx].mean(axis=1)                            # x[idx] has shape (n_boot, n)
    return means, float(means.std(ddof=1))
'''

HINTS = [
    'Use the `Generator` API (`np.random.default_rng`), not the legacy `np.random.seed` / `np.random.randint` functions. They produce different streams.',
    '`rng.integers(low, high, size=...)` excludes `high`, so `rng.integers(0, n, ...)` gives valid indices. Indexing `x` with the whole `(n_boot, n)` index matrix resamples every replicate at once.',
    '`means = x[idx].mean(axis=1)` and `se = means.std(ddof=1)`. Return the standard error as a Python float.',
]

_r = np.random.default_rng(21)
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0]), 5, 0),
    (np.array([10.0, 10.0, 10.0]), 3, 1),
    (np.array([7.0]), 4, 2),
    (np.array([-5.0, 5.0]), 6, 3),
    (np.array([1e9, 1e9 + 1.0, 1e9 + 2.0]), 4, 4),
    (_r.normal(50.0, 10.0, size=30), 200, 42),
    (_r.exponential(2.0, size=12), 50, 7),
]

EXPLANATION = r"""
## Why it works
`default_rng(seed)` creates a PCG64 generator whose stream is fully determined by the seed, so the same seed gives the same indices on every machine running the same NumPy version. One call to `integers` with `size=(n_boot, n)` fills the index matrix row by row. Fancy indexing `x[idx]` then gathers an `(n_boot, n)` matrix of resampled values, and one `mean(axis=1)` gives every replicate's statistic.

The spread of the bootstrap means approximates the standard error of the sample mean, roughly $\sigma / \sqrt{n}$. `ddof=1` is the usual choice because the replicates are treated as a sample.

## Complexity
$O(B \cdot n)$ time and memory for the index matrix, where $B$ is `n_boot`. For very large $B \cdot n$, process replicates in chunks.

## Common bugs
- Mixing APIs. `np.random.seed(0); np.random.randint(...)` uses the legacy Mersenne Twister and gives different numbers.
- `rng.integers(0, n - 1, ...)` never picks the last element, because `high` is already exclusive.
- Drawing one replicate per loop iteration with `size=n`. The statement fixes one call; any other sequence of calls is not guaranteed to produce the same stream.

## Interview follow-ups
- How do you get a 95% percentile bootstrap confidence interval? (`np.percentile(means, [2.5, 97.5])`.)
- When does the bootstrap fail? (Heavy-tailed data, statistics like the maximum, and dependent data such as time series, which need a block bootstrap.)
"""
