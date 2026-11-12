import numpy as np

META = dict(title='Stratified estimate of a population mean', tags=['Probability and Statistics', 'Data Processing'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:12.01'], related=['sampling-distribution-of-mean', 'standard-error-comparison'])

STATEMENT = r"""
A population of $N$ units is split into $H$ **strata** (for example user tiers), with $N_h$ units in stratum $h$. You sample $n_h$ units at random without replacement from each stratum. With stratum weights $W_h = N_h / N$, sample means $\bar y_h$ and sample variances $s_h^2$ (`ddof=1`):

$$\bar y_{\text{st}} = \sum_h W_h\,\bar y_h, \qquad \widehat{\mathrm{SE}}_{\text{st}} = \sqrt{\sum_h W_h^2 \left(1 - \frac{n_h}{N_h}\right) \frac{s_h^2}{n_h}}$$

For comparison, the standard error you'd report if you ignored the strata and treated all $n = \sum n_h$ values as one simple random sample is

$$\widehat{\mathrm{SE}}_{\text{srs}} = \sqrt{\left(1 - \frac{n}{N}\right)\frac{s^2}{n}}$$

where $s^2$ is the sample variance (`ddof=1`) of all $n$ values pooled together.

Write `stratified(values, strata, sizes)` that returns the tuple `(estimate, se_strat, se_srs)` as three Python floats.

## Requirements
- `values`: 1-D float array of the $n$ sampled values. `strata`: int array of the same length with labels in `0..H-1`.
- `sizes`: int array of shape `(H,)`, where `sizes[h]` $= N_h$. Every stratum has $2 \le n_h \le N_h$.
- No Python loops over strata: use `np.bincount` or similar grouped sums.
"""

STARTER = '''import numpy as np

def stratified(values, strata, sizes):
    """Return (stratified_mean, se_stratified, se_simple_random) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def stratified(values, strata, sizes):
    y = np.asarray(values, dtype=float)
    g = np.asarray(strata)
    Nh = np.asarray(sizes, dtype=float)
    H, N, n = len(Nh), Nh.sum(), len(y)
    nh = np.bincount(g, minlength=H)
    means = np.bincount(g, weights=y, minlength=H) / nh
    s2 = np.bincount(g, weights=(y - means[g]) ** 2, minlength=H) / (nh - 1)   # per-stratum variance, ddof=1
    W = Nh / N
    se_st = np.sqrt(np.sum(W ** 2 * (1 - nh / Nh) * s2 / nh))
    se_srs = np.sqrt((1 - n / N) * y.var(ddof=1) / n)
    return float(W @ means), float(se_st), float(se_srs)
'''

HINTS = [
    'Everything is a per-stratum sum. `np.bincount(strata, weights=w, minlength=H)` adds up `w` within each stratum.',
    'Get the counts $n_h$ and the means first. Then `means[strata]` broadcasts each unit\'s own stratum mean, so the squared deviations can be summed per stratum too.',
    'Weight by **population** shares $N_h/N$, not by how many you sampled. $(1 - n_h/N_h)$ is the finite population correction: it is 0 for a stratum you sampled completely.',
]

_r = np.random.default_rng(8)
_g = np.repeat([0, 1, 2], [30, 20, 10])
_y = np.concatenate([_r.normal(20, 2, 30), _r.normal(50, 5, 20), _r.normal(200, 20, 10)])
TESTS = [
    (np.array([1.0, 3.0, 10.0, 14.0]), np.array([0, 0, 1, 1]), np.array([100, 100])),
    (np.array([5.0, 7.0, 6.0, 20.0, 30.0]), np.array([1, 0, 0, 1, 0]), np.array([10, 4])),
    (np.array([2.0, 4.0, 9.0, 11.0]), np.array([0, 0, 1, 1]), np.array([2, 1000])),
    (np.array([3.0, 3.0, 8.0, 9.0, 1.0, 1.0]), np.array([0, 0, 1, 1, 2, 2]), np.array([50, 30, 20])),
    (_y, _g, np.array([6000, 3000, 1000])),
    (_y, _g, np.array([1000, 1000, 8000])),
    (_r.normal(0, 1, 40), _r.permutation(np.repeat([0, 1, 2, 3], 10)), np.array([10, 20, 30, 40])),
]

EXPLANATION = r"""
## Why stratify
The variance of $\bar y_{\text{st}}$ only involves the variance **within** each stratum. Differences **between** strata, which can be huge (free users versus enterprise accounts), drop out entirely. A simple random sample has to pay for that between-stratum spread, so when strata differ a lot the stratified standard error is much smaller. Compare `se_strat` with `se_srs` on the three-tier test.

The comparison is rough: `se_srs` treats the pooled values as if they were a simple random sample, which they aren't when strata were sampled at different rates. It's the error you'd **claim** by ignoring the design.

## Allocation
Given a total budget $n$, proportional allocation sets $n_h \propto N_h$. Neyman allocation sets $n_h \propto N_h S_h$, sampling more where the population is large **and** noisy, and minimises the variance.

## Complexity
$O(n + H)$ with `bincount`.

## Common bugs
- Weighting stratum means by the sample counts $n_h/n$. That's only right under proportional allocation.
- Dropping the finite population correction, which matters when a stratum is mostly or fully sampled (test 3).
- `ddof=0` within strata.

## Interview follow-ups
- Your A/B test has 5% enterprise users with very different spend. How can stratification (or CUPED) reduce the variance of the treatment effect estimate?
- Derive Neyman allocation with a Lagrange multiplier on $\sum n_h = n$.
"""
