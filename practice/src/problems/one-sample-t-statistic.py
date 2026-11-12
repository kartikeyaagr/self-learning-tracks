import numpy as np

META = dict(title='One-sample t statistic and confidence interval', tags=['Probability and Statistics'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:12.01'], related=['precision-recall-at-k'])

STATEMENT = r"""
You measured a model's latency on $n$ requests and want to test whether the true mean differs from a target $\mu_0$.

$$\bar x = \tfrac1n \textstyle\sum x_i, \qquad s = \sqrt{\tfrac{1}{n-1}\sum (x_i - \bar x)^2}, \qquad t = \frac{\bar x - \mu_0}{s / \sqrt n}$$

The $(1-\alpha)$ confidence interval for the mean is $\bar x \pm t^{*}\, s/\sqrt n$, where $t^*$ is the critical value. Here you are **given** $t^*$ (looking it up needs a t-table).

Write `t_test(x, mu0, t_crit)` and return the tuple `(t, df, lo, hi)`: the statistic, the degrees of freedom ($n-1$, an int), and the interval's bounds.

## Requirements
- `x` has at least 2 values and is not constant.
- Use the **sample** standard deviation ($n-1$ in the denominator).
"""

STARTER = '''import numpy as np

def t_test(x, mu0, t_crit):
    """Return (t, df, lo, hi)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def t_test(x, mu0, t_crit):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean, se = x.mean(), x.std(ddof=1) / np.sqrt(n)    # standard error of the mean
    return float((mean - mu0) / se), n - 1, float(mean - t_crit * se), float(mean + t_crit * se)
'''

HINTS = [
    '`np.std` defaults to `ddof=0`. The sample standard deviation needs `ddof=1`.',
    'The standard error $s/\\sqrt{n}$ appears in both the statistic and the interval, so compute it once.',
    'Degrees of freedom for a one-sample test are `n - 1`, returned as a plain int.',
]

_r = np.random.default_rng(8)
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0, 2.776),
    (np.array([10.0, 12.0]), 0.0, 12.706),
    (np.array([5.1, 4.9, 5.3, 5.0, 4.8, 5.2]), 5.5, 2.571),
    (_r.normal(loc=120, scale=15, size=30), 110.0, 2.045),
    (_r.normal(loc=0, scale=1, size=100), 0.0, 1.984),
    (np.array([2.0, 2.0, 2.0, 3.0]), 2.0, 3.182),
]

EXPLANATION = r"""
## What the t statistic measures
It's the distance between the sample mean and $\mu_0$, in units of standard error. The standard error shrinks like $1/\sqrt n$, so the same mean difference becomes more significant with more data.

## Why t and not z
$s$ is itself estimated from the data. With small $n$, that extra uncertainty gives heavier tails than a normal. The t distribution with $n-1$ degrees of freedom accounts for it and approaches the normal as $n$ grows. Compare $t^*$ = 12.7 at $n=2$ with 1.98 at $n=100$ in the tests.

## Test and interval agree
At level $\alpha$, the two-sided test rejects $H_0: \mu = \mu_0$ exactly when $\mu_0$ lies outside the $(1-\alpha)$ interval. Reporting the interval tells you more: it gives the size of the effect, not only whether it exists.

## Interview follow-ups
- Latency is right-skewed. When is the t-test still OK? (Large $n$ via the CLT; otherwise use a bootstrap or a log transform.)
- How would you compare two models' latencies on the **same** requests? (A paired t-test on the differences.)
"""
