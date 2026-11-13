import numpy as np

META = dict(title='One-sample z-test with known sigma', tags=['Probability and Statistics'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:12.01'], related=['normal-distribution', 't-test-one-sample', 'one-sample-t-statistic'])

STATEMENT = r"""
When the population standard deviation $\sigma$ is **known** (for example, a sensor's documented noise level), you can test whether the mean equals $\mu_0$ with

$$z = \frac{\bar x - \mu_0}{\sigma / \sqrt n}$$

Under $H_0$, $z \sim \mathcal N(0, 1)$. The p-value depends on the alternative hypothesis, with $\Phi$ the standard normal cdf:

| `alternative` | $H_1$ | p-value |
| --- | --- | --- |
| `'two-sided'` | $\mu \ne \mu_0$ | $2\,(1 - \Phi(\lvert z\rvert))$ |
| `'greater'` | $\mu > \mu_0$ | $1 - \Phi(z)$ |
| `'less'` | $\mu < \mu_0$ | $\Phi(z)$ |

Write `z_test(x, mu0, sigma, alternative='two-sided')` that returns the tuple `(z, p)` as two Python floats.

## Requirements
- `x`: non-empty 1-D float array. `sigma` $> 0$.
- Compute tail probabilities with `math.erfc`, using $1 - \Phi(z) = \tfrac12\,\mathrm{erfc}(z/\sqrt2)$, so small p-values keep their precision.
"""

STARTER = '''import math
import numpy as np

def z_test(x, mu0, sigma, alternative='two-sided'):
    """Return (z, p_value) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

def z_test(x, mu0, sigma, alternative='two-sided'):
    x = np.asarray(x, dtype=float)
    z = float((x.mean() - mu0) / (sigma / math.sqrt(len(x))))
    upper = lambda v: 0.5 * math.erfc(v / math.sqrt(2))     # P(Z > v)
    if alternative == 'two-sided':
        p = 2 * upper(abs(z))
    elif alternative == 'greater':
        p = upper(z)
    else:
        p = upper(-z)                                       # P(Z < z) = P(Z > -z)
    return z, p
'''

HINTS = [
    'The standard error of the mean is $\\sigma/\\sqrt n$. The statistic is how many standard errors $\\bar x$ sits from $\\mu_0$.',
    'Write one helper for the upper tail, $P(Z > v) = \\tfrac12\\,\\mathrm{erfc}(v/\\sqrt2)$. Every alternative is that helper applied to $|z|$, $z$ or $-z$.',
    'The two-sided p-value doubles the tail beyond $|z|$. For `"less"`, use the symmetry $\\Phi(z) = P(Z > -z)$.',
]

_r = np.random.default_rng(19)
TESTS = [
    (np.array([5.2, 4.8, 5.6, 5.4]), 5.0, 0.5),
    (np.array([98.0, 102.0, 101.0, 99.5, 103.0]), 100.0, 2.0, 'greater'),
    (np.array([98.0, 102.0, 101.0, 99.5, 103.0]), 100.0, 2.0, 'less'),
    (np.array([3.0]), 3.0, 1.0),
    (np.array([10.0, 10.0, 10.0]), 0.0, 1.0, 'two-sided'),
    (_r.normal(0.3, 1.0, size=200), 0.0, 1.0),
    (_r.normal(0.3, 1.0, size=200), 0.0, 1.0, 'less'),
]

EXPLANATION = r"""
## What the p-value is
It's the probability, **assuming $H_0$ is true**, of a statistic at least as extreme as the one observed, where "extreme" is defined by $H_1$. It is not the probability that $H_0$ is true. A small p-value says the data would be surprising under $H_0$.

## Why erfc
For $z = 17.3$ (test 5), $1 - \Phi(z)$ computed as `1 - cdf` is exactly 0, while `erfc` returns the true value, about $10^{-67}$. The answer matters when you rank thousands of features by p-value or apply a multiple-testing correction.

## When to use z versus t
Only use z when $\sigma$ is genuinely known, or when $n$ is large enough that the sample $s$ is a near-exact stand-in. With an estimated $s$ and small $n$, use the t-test: its heavier tails account for the uncertainty in $s$.

## Complexity
$O(n)$ for the mean.

## Common bugs
- Dividing by $\sigma$ instead of $\sigma / \sqrt n$.
- Doubling the one-sided p-value for the wrong tail. With `'greater'` and a negative $z$, the p-value is above 0.5. It doesn't flip sign.
- Calling a one-sided test after looking at which direction the data went. That doubles the real false-positive rate.

## Interview follow-ups
- How many samples do you need to detect a shift of $\delta$ with power $1 - \beta$ at level $\alpha$? ($n = \left(\frac{(z_{1-\alpha/2} + z_{1-\beta})\,\sigma}{\delta}\right)^2$.)
- What's the link between this test and a 95% confidence interval for $\mu$? (Reject at 5% exactly when $\mu_0$ is outside $\bar x \pm 1.96\,\sigma/\sqrt n$.)
"""
