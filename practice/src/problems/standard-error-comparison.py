import numpy as np

META = dict(title='Standard error of the mean versus the median', tags=['Probability and Statistics'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:12.01'], related=['sampling-distribution-of-mean', 'bootstrap-confidence-interval'])

STATEMENT = r"""
The mean and the median both estimate the centre of a symmetric distribution. Which one is more **precise** depends on the tails. You can measure it directly by repeating an experiment: each of $R$ rows of `samples` is an independent sample of size $n$.

For each row compute its mean and its median. The **empirical standard error** of an estimator is the standard deviation (with `ddof=1`) of its $R$ values. The **relative efficiency** of the median versus the mean is

$$\mathrm{eff} = \frac{\mathrm{SE}_{\text{mean}}^2}{\mathrm{SE}_{\text{median}}^2}$$

so $\mathrm{eff} > 1$ means the median is more precise.

Write `compare_se(samples)` that returns the tuple `(se_mean, se_median, eff)` as three Python floats.

## Requirements
- `samples`: float array of shape `(R, n)` with $R \ge 2$ and $n \ge 1$. Each row is one experiment.
- For even $n$, a row's median is the average of its two middle values.
- Vectorise over rows. The tests always give $\mathrm{SE}_{\text{median}} > 0$.
"""

STARTER = '''import numpy as np

def compare_se(samples):
    """samples: (R, n). Return (se_mean, se_median, efficiency) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def compare_se(samples):
    s = np.asarray(samples, dtype=float)
    se_mean = s.mean(axis=1).std(ddof=1)          # spread of the estimator across experiments
    se_median = np.median(s, axis=1).std(ddof=1)
    return float(se_mean), float(se_median), float(se_mean ** 2 / se_median ** 2)
'''

HINTS = [
    'An estimator\'s standard error is how much it varies from one experiment to the next. Compute the estimator for every row, then take the spread of those values.',
    '`samples.mean(axis=1)` and `np.median(samples, axis=1)` each give one value per row. Then `.std(ddof=1)`.',
    'Efficiency compares **variances**, so square both standard errors before dividing.',
]

_r = np.random.default_rng(2024)
TESTS = [
    (np.array([[1.0, 2.0, 9.0], [2.0, 3.0, 4.0], [0.0, 5.0, 6.0]]),),
    (np.array([[1.0, 3.0], [2.0, 2.0], [4.0, 8.0], [0.0, 1.0]]),),
    (np.array([[3.0], [5.0], [10.0]]),),
    (_r.normal(0.0, 1.0, size=(400, 25)),),
    (_r.laplace(0.0, 1.0, size=(400, 25)),),
    (_r.uniform(-1.0, 1.0, size=(300, 20)),),
    (_r.standard_t(2, size=(300, 31)) + 1e3,),
]

EXPLANATION = r"""
## What theory predicts
For normal data, $\mathrm{SE}_{\text{mean}} = \sigma/\sqrt n$ and $\mathrm{SE}_{\text{median}} \approx \sqrt{\pi/2}\,\sigma/\sqrt n$, so $\mathrm{eff} \approx 2/\pi \approx 0.64$: the median wastes about a third of the data. For Laplace data (heavier tails) the median wins: $\mathrm{eff} \to 2$ as $n$ grows (about 1.3 at $n = 25$), and for Student-t with 2 degrees of freedom the mean's variance is infinite, so the median wins by a lot. For uniform data (light tails) the median is worse still. Run the tests and check.

In general, the median's large-sample variance is $\frac{1}{4 n f(m)^2}$, where $f(m)$ is the density at the median: a sharp peak makes the median precise.

## Why simulate
For most estimators (trimmed means, quantiles, ratios) there's no simple formula, but you can always compute the estimator on many independent samples and look at its spread. When you only have **one** sample, the bootstrap fakes the repetitions by resampling it.

## Complexity
$O(Rn)$ for the means. `np.median` partitions each row, $O(Rn)$ on average.

## Common bugs
- Taking the standard deviation over all $R \times n$ raw values. That's the population spread, $\sqrt n$ times too large.
- Comparing standard errors directly instead of variances.
- `ddof=0`, which slightly understates the standard error for small $R$.

## Interview follow-ups
- Your latency data has rare 10-second timeouts. Would you report the mean or the median, and how do you compare two versions' medians? (The median or a trimmed mean. Compare medians with a bootstrap.)
- What's the efficiency of a 10% trimmed mean on normal data? (About 0.94. It keeps most of the mean's precision and gains robustness.)
"""
