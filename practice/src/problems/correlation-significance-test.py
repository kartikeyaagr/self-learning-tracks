import numpy as np

META = dict(title="Is this correlation real? Fisher's z", tags=['Probability and Statistics'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:11.21', 'mlmaths:12.17'], related=['pearson-correlation-matrix', 'z-test-one-sample'])

STATEMENT = r"""
A sample correlation $r$ from $n$ pairs is noisy, and its sampling distribution is skewed when $|r|$ is near 1. Fisher's transformation fixes both problems:

$$z_r = \operatorname{artanh}(r) = \tfrac12 \ln\frac{1+r}{1-r}$$

is approximately normal with standard error $1/\sqrt{n-3}$, whatever the true correlation $\rho$.

- **Test** $H_0: \rho = 0$ with $z = z_r\sqrt{n-3}$ and two-sided p-value $p = \mathrm{erfc}(|z|/\sqrt2)$.
- **Interval** for $\rho$: build it on the $z_r$ scale, then map both ends back with $\tanh$:

$$\Big(\tanh\big(z_r - z^{*}/\sqrt{n-3}\big),\; \tanh\big(z_r + z^{*}/\sqrt{n-3}\big)\Big)$$

Write `corr_test(x, y, z_crit=1.959963984540054)` that returns the tuple `(r, z, p, lo, hi)` as five Python floats, where $r$ is the Pearson correlation of `x` and `y`.

## Requirements
- `x`, `y`: 1-D float arrays of the same length $n \ge 4$. Neither is constant, and $|r| < 1$.
"""

STARTER = '''import math
import numpy as np

def corr_test(x, y, z_crit=1.959963984540054):
    """Return (r, z_statistic, p_value, ci_lo, ci_hi) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

def corr_test(x, y, z_crit=1.959963984540054):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx, dy = x - x.mean(), y - y.mean()
    r = float(dx @ dy / math.sqrt((dx @ dx) * (dy @ dy)))
    zr, se = math.atanh(r), 1 / math.sqrt(len(x) - 3)
    z = zr / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return r, z, p, math.tanh(zr - z_crit * se), math.tanh(zr + z_crit * se)
'''

HINTS = [
    'Compute $r$ from centred vectors: their dot product over the product of their norms.',
    '`math.atanh(r)` is Fisher\'s $z_r$. Its standard error $1/\\sqrt{n-3}$ is the same for the test and the interval.',
    'Make the interval symmetric on the $z_r$ scale, then apply `math.tanh` to each end. The result is **not** symmetric around $r$.',
]

_r = np.random.default_rng(27)
_x = _r.normal(size=60)
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 1.0, 4.0, 3.0, 5.0])),
    (np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 3.0, 1.0, 2.0])),
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])),
    (_x, 0.9 * _x + 0.3 * _r.normal(size=60)),
    (_x, _r.normal(size=60)),
    (_x, -0.2 * _x + _r.normal(size=60), 2.5758293035489004),
    (np.arange(10.0) + 1e6, np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0])),
]

EXPLANATION = r"""
## Why transform
The correlation is bounded by $\pm 1$. When $\rho = 0.9$, the sample $r$ can't go far above 0.9 but can easily fall to 0.7, so its distribution is skewed, and its variance $\approx (1 - \rho^2)^2/n$ depends on the unknown $\rho$. $\operatorname{artanh}$ stretches the region near $\pm1$ exactly enough to make the distribution nearly normal with a constant variance. That's why the interval is lopsided around $r$ after mapping back.

## Alternative test
The exact test for $\rho = 0$ under bivariate normality uses $t = r\sqrt{\frac{n-2}{1 - r^2}}$ with $n - 2$ degrees of freedom. Fisher's z is the approximation that also gives intervals and tests against $\rho_0 \ne 0$.

## Complexity
$O(n)$.

## Common bugs
- Building the interval as $r \pm z^*/\sqrt{n-3}$ on the raw scale. It can go past $\pm1$ and has the wrong coverage.
- Using $\sqrt n$ instead of $\sqrt{n - 3}$.
- Reading significance as strength: with $n = 10^6$, $r = 0.003$ is "significant" and still useless.

## Interview follow-ups
- You test 1,000 features for correlation with the label and keep those with $p < 0.05$. What goes wrong? (About 50 null features pass by chance. Correct for multiple testing.)
- How would you test whether two correlations from independent samples differ? ($\frac{z_1 - z_2}{\sqrt{1/(n_1-3) + 1/(n_2-3)}}$ is approximately standard normal.)
"""
