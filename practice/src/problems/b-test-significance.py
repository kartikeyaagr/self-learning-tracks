import numpy as np

META = dict(title='A/B test: two-proportion z-test', tags=['Probability and Statistics', 'Metrics & Evaluation'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:12.17'], related=['z-test-one-sample', 'p-values-and-significance', 'binomial-distribution'])

STATEMENT = r"""
Variant A converted $x_A$ of $n_A$ users and variant B converted $x_B$ of $n_B$. With $\hat p_A = x_A / n_A$ and $\hat p_B = x_B / n_B$, the difference is $d = \hat p_B - \hat p_A$.

**Test** $H_0: p_A = p_B$. Under $H_0$ both arms share one rate, so estimate it from the pooled data, $\hat p = \frac{x_A + x_B}{n_A + n_B}$:

$$z = \frac{d}{\sqrt{\hat p(1 - \hat p)\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}, \qquad p = \mathrm{erfc}\!\left(\frac{|z|}{\sqrt 2}\right) \text{ (two-sided)}$$

**Interval** for the difference. Here the rates are not assumed equal, so use the unpooled standard error:

$$d \pm z^{*}\sqrt{\frac{\hat p_A(1 - \hat p_A)}{n_A} + \frac{\hat p_B(1 - \hat p_B)}{n_B}}$$

Write `ab_test(x_a, n_a, x_b, n_b, z_crit=1.959963984540054)` that returns the tuple `(d, z, p, lo, hi)` as five Python floats. The default `z_crit` gives a 95% interval.

## Requirements
- Counts are ints with $0 \le x \le n$ and $n \ge 1$.
- If the pooled rate is 0 or 1 (no conversions at all, or all users converted), the pooled standard error is 0. Return `z = 0.0` and `p = 1.0`.
- If the unpooled standard error is 0, the interval collapses to `(d, d)`.
"""

STARTER = '''import math

def ab_test(x_a, n_a, x_b, n_b, z_crit=1.959963984540054):
    """Return (diff, z, p_value, ci_lo, ci_hi) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import math

def ab_test(x_a, n_a, x_b, n_b, z_crit=1.959963984540054):
    pa, pb = x_a / n_a, x_b / n_b
    d = pb - pa
    pooled = (x_a + x_b) / (n_a + n_b)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))   # SE under H0 (shared rate)
    z = d / se0 if se0 > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    se1 = math.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)      # SE for the interval (rates may differ)
    return d, z, p, d - z_crit * se1, d + z_crit * se1
'''

HINTS = [
    'Two different standard errors appear: the **pooled** one for the test (it assumes $H_0$) and the **unpooled** one for the interval.',
    'The two-sided normal p-value is $2(1 - \\Phi(|z|)) = \\mathrm{erfc}(|z|/\\sqrt2)$, one call to `math.erfc`.',
    'Guard the division: when nobody (or everybody) converted, the pooled SE is 0, and `z = 0.0`, `p = 1.0` by definition. With `se1 = 0` the formula already gives `(d, d)`.',
]

TESTS = [
    (200, 2000, 250, 2000),
    (30, 1000, 33, 1000),
    (0, 500, 0, 500),
    (10, 10, 10, 10),
    (0, 50, 5, 50),
    (1200, 100000, 1350, 100000),
    (48, 120, 30, 115, 2.5758293035489004),
    (5, 5, 0, 5),
]

EXPLANATION = r"""
## Why two standard errors
The p-value asks how surprising $d$ would be **if $H_0$ were true**, so its standard error is computed under $H_0$, with one shared rate. The interval asks where the true difference plausibly lies, with no assumption that it's 0, so each arm keeps its own rate. This is why the test and the interval can disagree slightly near the boundary.

## Reading the result
Test 2 (3.0% versus 3.3% on 1,000 users each) has $p \approx 0.70$: the interval for $d$ covers everything from about $-1.2$ to $+1.8$ points. Test 6 has a similar relative lift at 100x the traffic, and it's highly significant. Small lifts on low base rates need a lot of users.

## Complexity
$O(1)$ from the counts. Computing the counts is the expensive part.

## Common bugs
- Using the unpooled SE in the test statistic. It's an acceptable variant (Wald test), but it isn't the pooled test described here.
- Forgetting the $1/n_A + 1/n_B$ structure and dividing by $n_A + n_B$.
- Applying a z-test with tiny expected counts (test 5): the normal approximation is shaky there. Fisher's exact test is safer.

## Interview follow-ups
- How many users per arm do you need to detect a lift from 10% to 11% with 80% power at $\alpha = 0.05$? ($n \approx \frac{(1.96 + 0.84)^2\,(p_1 q_1 + p_2 q_2)}{(p_2 - p_1)^2} \approx 14{,}700$.)
- Your metric is revenue per user, not conversion. What changes? (Use a Welch t-test on per-user values. Heavy tails may call for winsorising, CUPED or a bootstrap.)
"""
