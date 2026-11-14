import numpy as np

META = dict(title='Moments of a discrete distribution', tags=['Probability'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:10.15', 'mlmaths:10.25'], related=['skewness-and-kurtosis', 'binomial-distribution'])

STATEMENT = r"""
A discrete random variable takes value $v_i$ with probability $p_i$. Its moments are expectations:

$$\mu = E[X] = \sum_i p_i v_i, \qquad \mu_k = E\big[(X - \mu)^k\big] = \sum_i p_i (v_i - \mu)^k$$

From the central moments: variance $\mu_2$, standard deviation $\sigma = \sqrt{\mu_2}$, skewness $\mu_3 / \sigma^3$ and excess kurtosis $\mu_4 / \sigma^4 - 3$.

Write `discrete_moments(values, weights)` that returns the tuple `(mean, var, std, skew, ex_kurt)` as five Python floats.

## Requirements
- `values`: 1-D float array of support points (they need not be sorted or distinct).
- `weights`: 1-D array of the same length, non-negative but **not necessarily normalised**: weight $w_i$ means $p_i = w_i / \sum_j w_j$. Zero weights are allowed.
- At least two distinct values have positive weight, so $\sigma > 0$.
- Compute central moments from deviations $v_i - \mu$, not from raw moments like $E[X^2] - \mu^2$, so data with a large offset stays accurate.
"""

STARTER = '''import numpy as np

def discrete_moments(values, weights):
    """Return (mean, var, std, skew, excess_kurtosis) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def discrete_moments(values, weights):
    v = np.asarray(values, dtype=float)
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    mean = p @ v
    d = v - mean
    var, m3, m4 = p @ d ** 2, p @ d ** 3, p @ d ** 4   # weighted central moments
    std = np.sqrt(var)
    return float(mean), float(var), float(std), float(m3 / std ** 3), float(m4 / var ** 2 - 3.0)
'''

HINTS = [
    'Normalise the weights first. After that, every moment is a dot product `p @ something`.',
    'Compute the mean, then the deviations `d = v - mean`, then `p @ d**2`, `p @ d**3`, `p @ d**4`.',
    'Excess kurtosis subtracts 3, so a normal distribution scores 0. Repeated support values need no special handling: their weights simply add up inside the dot product.',
]

_r = np.random.default_rng(14)
TESTS = [
    (np.array([0.0, 1.0]), np.array([0.7, 0.3])),
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), np.ones(6)),
    (np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])),
    (np.array([0.0, 10.0, 0.0]), np.array([2.0, 1.0, 3.0])),
    (np.array([1e9, 1e9 + 1.0, 1e9 + 2.0]), np.array([1.0, 2.0, 7.0])),
    (np.arange(11.0), np.array([1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1], dtype=float)),
    (_r.normal(size=8) * 5, _r.random(8)),
]

EXPLANATION = r"""
## Reading the moments
The mean locates the distribution and the variance measures its spread. Skewness is the third moment in units of $\sigma^3$: positive when the long tail points right. Excess kurtosis compares the fourth moment with a normal's: high values mean more of the variance comes from rare large deviations.

A fair die (test 2) is symmetric, so its skewness is 0, and flat, so its kurtosis is negative ($\approx -1.27$). The Binomial(10, 0.5) weights in test 6 give variance $np(1-p) = 2.5$ and excess kurtosis $(1 - 6p(1-p))/(np(1-p)) = -0.2$.

## Why central, not raw, moments
$\mathrm{Var}(X) = E[X^2] - E[X]^2$ is true but numerically fragile. With values near $10^9$ you subtract two numbers near $10^{18}$, and float64 can't resolve the difference. Centring first keeps every power small.

## Complexity
$O(n)$ time, a handful of dot products.

## Common bugs
- Forgetting to normalise weights, which scales every moment by $\sum w$.
- Dividing $\mu_4$ by $\mu_2$ instead of $\mu_2^2$.
- Using sample-style $n-1$ corrections. These are exact population moments, so there's nothing to correct.

## Interview follow-ups
- Express the skewness of the Binomial($n$, $p$) in closed form and check it against this code. ($\frac{1-2p}{\sqrt{np(1-p)}}$.)
- How do you get all moments at once from the moment generating function $M(t) = E[e^{tX}]$? (The $k$-th derivative at $t = 0$ is $E[X^k]$.)
"""
