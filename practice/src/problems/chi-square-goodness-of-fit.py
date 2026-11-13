import numpy as np

META = dict(title='Chi-square goodness-of-fit test', tags=['Probability and Statistics'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:12.03'], related=['independence-testing', 'p-values-and-significance'])

STATEMENT = r"""
You rolled a die 600 times, or logged which of $k$ servers a load balancer picked. Do the counts match the distribution you expected?

With observed counts $O_i$ and expected counts $E_i = N\,\pi_i$ (where $N = \sum O_i$ and $\pi$ is the hypothesised distribution),

$$X^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

is approximately $\chi^2$ with $\nu = k - 1 - \texttt{ddof}$ degrees of freedom, where `ddof` counts parameters estimated from the same data. The p-value is the upper tail $Q(X^2; \nu) = P(\chi^2_\nu \ge X^2)$. For integer $\nu$ it follows from

$$Q(x; 1) = \mathrm{erfc}\big(\sqrt{x/2}\big), \qquad Q(x; 2) = e^{-x/2}, \qquad Q(x; \nu + 2) = Q(x; \nu) + \frac{(x/2)^{\nu/2}\, e^{-x/2}}{\Gamma(\nu/2 + 1)}$$

Write `chi2_gof(observed, expected, ddof=0)` that returns the tuple `(stat, df, p)`: $X^2$ as a float, $\nu$ as an int and the p-value as a float.

## Requirements
- `observed`: 1-D array of non-negative counts with a positive total. `expected`: same length, all **positive**, and not necessarily normalised. Rescale it to sum to $N$.
- $\nu \ge 1$ always holds.
- Evaluate the recurrence's extra term in log space, using `math.lgamma`, so large $x$ and $\nu$ don't overflow. If $X^2 = 0$, the p-value is exactly `1.0`.
"""

STARTER = '''import math
import numpy as np

def chi2_gof(observed, expected, ddof=0):
    """Return (chi2_statistic, df, p_value)."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

def chi2_sf(x, df):
    """P(chi2_df >= x) for integer df >= 1, by stepping df up two at a time."""
    if x <= 0:
        return 1.0
    q, k = (math.erfc(math.sqrt(x / 2)), 1) if df % 2 else (math.exp(-x / 2), 2)
    while k < df:
        q += math.exp(k / 2 * math.log(x / 2) - x / 2 - math.lgamma(k / 2 + 1))
        k += 2
    return min(q, 1.0)

def chi2_gof(observed, expected, ddof=0):
    O = np.asarray(observed, dtype=float)
    E = np.asarray(expected, dtype=float)
    E = E / E.sum() * O.sum()                 # expected counts on the same total
    stat = float(((O - E) ** 2 / E).sum())
    df = len(O) - 1 - ddof
    return stat, df, chi2_sf(stat, df)
'''

HINTS = [
    'Normalise `expected` to probabilities and multiply by the observed total. Then the statistic is one vectorised line.',
    'For the p-value, start at $\\nu = 1$ (erfc) or $\\nu = 2$ (exp) depending on parity, and add one extra term per step of 2 until you reach the target $\\nu$.',
    'Compute each extra term as `exp(k/2 * log(x/2) - x/2 - lgamma(k/2 + 1))`. Return `1.0` when the statistic is 0, which avoids `log(0)`.',
]

_r = np.random.default_rng(66)
TESTS = [
    (np.array([18, 22, 20, 25, 15, 20]), np.ones(6)),
    (np.array([50, 30, 20]), np.array([0.5, 0.3, 0.2])),
    (np.array([10, 10]), np.array([1.0, 3.0])),
    (np.array([5, 0, 0, 0]), np.ones(4)),
    (np.array([40, 35, 25, 20, 5]), np.array([30.0, 30.0, 20.0, 20.0, 10.0]), 1),
    (np.array([1000, 1100]), np.array([1.0, 1.0])),
    (_r.multinomial(5000, np.full(12, 1 / 12)), np.ones(12)),
    (_r.multinomial(400, [0.3, 0.25, 0.2, 0.15, 0.1]), np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 2),
]

EXPLANATION = r"""
## Why the statistic is chi-square
Each count is roughly normal with variance near $E_i$, so $(O_i - E_i)/\sqrt{E_i}$ is roughly standard normal and $X^2$ is a sum of squares. The counts must add up to $N$, which removes one degree of freedom. Each parameter fitted from the data (such as a Poisson rate) removes one more, which is what `ddof` is for.

## Why the recurrence
Integrating the $\chi^2_{\nu+2}$ density by parts gives $Q(x; \nu + 2) = Q(x; \nu) + 2 f_{\nu+2}(x)$, where $f$ is the density. Starting from the closed forms for 1 and 2 degrees of freedom gives every integer $\nu$. Doing the term in log space keeps $(x/2)^{\nu/2}$ and $\Gamma$ from overflowing.

## Complexity
$O(k)$ for the statistic and $O(\nu)$ for the p-value.

## Common bugs
- Passing probabilities as expected counts without scaling by $N$.
- Using $k$ degrees of freedom instead of $k - 1$.
- Trusting the approximation when some $E_i < 5$ (test 4). Merge sparse bins or use an exact multinomial test.
- Squaring before subtracting, or dividing by $O_i$ instead of $E_i$.

## Interview follow-ups
- How would you test whether a feature's distribution in production has drifted from training? (Bin it and run this test with the training proportions, or use PSI or a KS test. With huge $N$, any tiny drift becomes significant, so also look at effect size.)
- How do you turn this into a test of independence for a two-way table? (Expected counts from the product of marginals, and $\nu = (r-1)(c-1)$.)
"""
