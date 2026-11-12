import numpy as np

META = dict(title='Exact p-value for a one-sample t-test', tags=['Probability and Statistics'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:12.01', 'mlmaths:12.03'], related=['one-sample-t-statistic', 'z-test-one-sample', 'p-values-and-significance'])

STATEMENT = r"""
The t statistic $t = \dfrac{\bar x - \mu_0}{s / \sqrt n}$ (with sample standard deviation $s$, `ddof=1`) follows a Student-t distribution with $\nu = n - 1$ degrees of freedom under $H_0$. To turn it into a two-sided p-value without SciPy, use the closed form for **integer** $\nu$.

Let $\theta = \arctan\!\big(|t| / \sqrt\nu\big)$, $c = \cos\theta$ and $\sigma = \sin\theta$. The probability $A = P(|T| < |t|)$ is

- $\nu = 1$: $\quad A = \dfrac{2\theta}{\pi}$
- $\nu$ odd, $\nu \ge 3$: $\quad A = \dfrac{2}{\pi}\Big(\theta + \sigma c\,\big(a_0 + a_1 c^2 + \dots + a_{(\nu-3)/2}\, c^{\nu-3}\big)\Big)$, with $a_0 = 1$ and $a_k = a_{k-1}\,\dfrac{2k}{2k+1}$
- $\nu$ even: $\quad A = \sigma\,\big(b_0 + b_1 c^2 + \dots + b_{(\nu-2)/2}\, c^{\nu-2}\big)$, with $b_0 = 1$ and $b_k = b_{k-1}\,\dfrac{2k-1}{2k}$

and the two-sided p-value is $p = 1 - A$.

Write `t_test_pvalue(x, mu0)` that returns the tuple `(t, df, p)`: the statistic as a float, $\nu$ as an int and the p-value as a float.

## Requirements
- `x`: 1-D float array with $n \ge 2$ values, not all equal.
- Accuracy: the p-value must match to within about $10^{-8}$ in absolute terms.
"""

STARTER = '''import math
import numpy as np

def t_test_pvalue(x, mu0):
    """Return (t, df, two_sided_p)."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

def _t_abs_cdf(t, nu):
    """A = P(|T| < t) for t >= 0 and integer nu >= 1 (finite series)."""
    theta = math.atan(t / math.sqrt(nu))
    c, s = math.cos(theta), math.sin(theta)
    if nu == 1:
        return 2 * theta / math.pi
    total, term = 0.0, 1.0
    if nu % 2:                                   # odd: a_k = a_{k-1} * 2k / (2k + 1)
        for k in range((nu - 3) // 2 + 1):
            if k:
                term *= c * c * (2 * k) / (2 * k + 1)
            total += term
        return 2 / math.pi * (theta + s * c * total)
    for k in range((nu - 2) // 2 + 1):           # even: b_k = b_{k-1} * (2k - 1) / (2k)
        if k:
            term *= c * c * (2 * k - 1) / (2 * k)
        total += term
    return s * total

def t_test_pvalue(x, mu0):
    x = np.asarray(x, dtype=float)
    n = len(x)
    t = float((x.mean() - mu0) / (x.std(ddof=1) / math.sqrt(n)))
    return t, n - 1, 1.0 - _t_abs_cdf(abs(t), n - 1)
'''

HINTS = [
    'Start with the statistic and $\\nu = n - 1$. Only $|t|$ enters the two-sided p-value.',
    'Build each series term from the previous one, multiplying by $c^2$ and the coefficient ratio, instead of computing powers and products from scratch.',
    'Check the small cases by hand. $\\nu = 2$ gives $A = \\sin\\theta = |t|/\\sqrt{t^2 + 2}$, and $\\nu = 3$ gives $A = \\tfrac{2}{\\pi}(\\theta + \\sin\\theta\\cos\\theta)$. A t of 2.776 with $\\nu = 4$ should give $p \\approx 0.05$.',
]

_r = np.random.default_rng(33)
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 1.0),
    (np.array([10.0, 12.0]), 0.0),
    (np.array([4.0, 6.0, 5.0]), 5.0),
    (np.array([2.0, 2.5, 1.5, 2.0, 3.0, 1.0]), 1.0),
    (np.array([0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.16]), -1.0),
    (_r.normal(0.4, 1.0, size=30), 0.0),
    (_r.normal(100.0, 15.0, size=101), 98.0),
    (_r.normal(0.0, 1.0, size=12), 0.0),
]

EXPLANATION = r"""
## Where the series comes from
The Student-t density is proportional to $(1 + t^2/\nu)^{-(\nu+1)/2}$. Substituting $t = \sqrt\nu \tan\theta$ turns it into a power of $\cos\theta$, and integrating $\cos^m\theta$ by parts gives a recurrence that steps $m$ down by 2 each time. For integer $\nu$ it stops after about $\nu/2$ terms, which are the coefficients $a_k$ and $b_k$. For non-integer $\nu$ (Welch's test) you need the regularised incomplete beta function instead.

## Sanity checks
- $\nu = 1$ is the Cauchy distribution. Test 2 has $t = 11$ and $p = 1 - \tfrac{2}{\pi}\arctan 11 \approx 0.058$: a mean 11 standard errors from 0 is still unconvincing with two points.
- For large $\nu$, the p-value approaches the z-test's $\mathrm{erfc}(|t|/\sqrt2)$.

## Complexity
$O(n)$ for the statistic plus $O(\nu)$ for the series.

## Common bugs
- `ddof=0` for $s$, which inflates $t$.
- Off-by-one in the number of series terms: the last power is $c^{\nu-3}$ (odd) or $c^{\nu-2}$ (even).
- Using $1 - \Phi(|t|)$ from the normal, which understates p for small samples.

## Interview follow-ups
- How would you get a one-sided p-value from $A$? (For $H_1: \mu > \mu_0$, $p = (1 - A)/2$ if $t > 0$, otherwise $1 - (1 - A)/2$.)
- The t-test assumes normality. How robust is it, and what's a non-parametric alternative? (Robust for moderate $n$ unless heavily skewed. Use the Wilcoxon signed-rank test or a bootstrap.)
"""
