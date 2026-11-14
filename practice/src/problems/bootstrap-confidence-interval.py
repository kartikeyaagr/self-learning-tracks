import numpy as np

META = dict(title='Bootstrap confidence intervals, vectorised', tags=['Probability and Statistics'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:12.17', 'ieng:00.04'], related=['standard-error-comparison', 'percentiles-quantiles'])

STATEMENT = r"""
The bootstrap estimates the uncertainty of a statistic $\hat\theta$ from a single sample by resampling it with replacement. Implement the whole pipeline.

**Resample.** Use exactly this generator call, so results are reproducible:

```
rng = np.random.default_rng(seed)
idx = rng.integers(0, n, size=(B, n))   # row b holds the indices of bootstrap sample b
```

**Recompute.** Apply the statistic to every row of `x[idx]`, giving $\hat\theta^*_1, \dots, \hat\theta^*_B$.

**Summarise,** with $q_{\text{lo}}$ and $q_{\text{hi}}$ the $\alpha/2$ and $1 - \alpha/2$ quantiles of the $\hat\theta^*$ (use `np.quantile` with its default linear method):

| output | formula |
| --- | --- |
| `theta` | the statistic on the original `x` |
| `se` | standard deviation of the $\hat\theta^*$, `ddof=1` |
| `bias` | $\operatorname{mean}(\hat\theta^*) - \hat\theta$ |
| `percentile` | $(q_{\text{lo}},\; q_{\text{hi}})$ |
| `basic` | $(2\hat\theta - q_{\text{hi}},\; 2\hat\theta - q_{\text{lo}})$ |

Write `bootstrap(x, stat, B, alpha, seed)` that returns the tuple `(theta, se, bias, percentile, basic)`. The first three are Python floats and the last two are tuples of two floats.

## Requirements
- `x`: 1-D float array with $n \ge 2$. `B`: int $\ge 2$. `alpha`: float in $(0, 1)$. `seed`: int.
- `stat` is one of `'mean'`, `'median'` or `'std'` (sample standard deviation, `ddof=1`).
- No Python loop over the $B$ resamples: compute the statistic along `axis=1`.
"""

STARTER = '''import numpy as np

def bootstrap(x, stat, B, alpha, seed):
    """Return (theta, se, bias, (pct_lo, pct_hi), (basic_lo, basic_hi))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

STATS = {
    'mean': lambda a, axis=None: np.mean(a, axis=axis),
    'median': lambda a, axis=None: np.median(a, axis=axis),
    'std': lambda a, axis=None: np.std(a, axis=axis, ddof=1),
}

def bootstrap(x, stat, B, alpha, seed):
    x = np.asarray(x, dtype=float)
    f = STATS[stat]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    boot = f(x[idx], axis=1)                          # (B,) bootstrap replicates
    theta = float(f(x))
    lo, hi = (float(q) for q in np.quantile(boot, [alpha / 2, 1 - alpha / 2]))
    # Basic (pivotal) interval: reflect the percentile interval around theta.
    return theta, float(boot.std(ddof=1)), float(boot.mean() - theta), (lo, hi), (2 * theta - hi, 2 * theta - lo)
'''

HINTS = [
    '`x[idx]` with `idx` of shape `(B, n)` builds every bootstrap sample at once as a `(B, n)` matrix.',
    'Map the `stat` name to a function that accepts `axis=`, so the same function computes $\\hat\\theta$ on `x` and the replicates along `axis=1`. Remember `ddof=1` for `std`.',
    'The basic interval reflects the percentile one around $\\hat\\theta$: its **lower** end uses the **upper** quantile. Create the generator inside the function from `seed`, and make exactly the one `integers` call shown.',
]

_r = np.random.default_rng(77)
TESTS = [
    (np.array([2.0, 4.0, 4.0, 5.0, 9.0]), 'mean', 8, 0.5, 0),
    (np.array([1.0, 3.0, 8.0]), 'median', 6, 0.2, 1),
    (np.array([10.0, 10.0]), 'mean', 5, 0.1, 3),
    (_r.exponential(50.0, size=40), 'mean', 2000, 0.05, 42),
    (_r.exponential(50.0, size=40), 'median', 2000, 0.1, 7),
    (_r.normal(0.0, 3.0, size=25), 'std', 1500, 0.05, 11),
    (np.array([-5.0, 0.0, 1.0, 1.0, 2.0, 30.0]), 'std', 300, 0.32, 5),
]

EXPLANATION = r"""
## Why resampling works
The sample is our best picture of the population. Drawing from it with replacement mimics drawing fresh samples from the population, so the spread of $\hat\theta^*$ around $\hat\theta$ approximates the spread of $\hat\theta$ around the true $\theta$. This works for any statistic you can compute, with no formula needed.

## Percentile versus basic
The **percentile** interval reads the quantiles of $\hat\theta^*$ directly. The **basic** interval assumes $\hat\theta - \theta$ is distributed like $\hat\theta^* - \hat\theta$. If the replicates sit to the right of $\hat\theta$, the true $\theta$ is probably to the left, so it flips the interval around $\hat\theta$. They agree when the bootstrap distribution is symmetric and differ for skewed statistics such as the mean of exponential data. BCa intervals go further and correct for both bias and skew.

## Complexity
$O(Bn)$ time and memory for the index matrix. For huge $B \cdot n$, process the resamples in chunks.

## Common bugs
- Resampling **without** replacement, which just permutes $x$ and gives zero variance for the mean.
- Creating the generator once at module level, so a second call gives different results.
- Swapping the quantiles in the basic interval.
- Default `ddof=0` for the `std` statistic.

## Interview follow-ups
- When does the bootstrap fail? (Extremes like the max, very small $n$, heavy tails with infinite variance, and dependent data, which needs a block bootstrap.)
- How would you bootstrap the difference in conversion rate between two A/B arms? (Resample each arm independently, compute the difference per replicate, and take its quantiles.)
"""
