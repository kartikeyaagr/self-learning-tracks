import numpy as np

META = dict(title='Gaussian maximum likelihood fit', tags=['Probability and Statistics'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:12.10'], related=['exponential-distribution', 'sample-variance-and-standard-deviation'])

STATEMENT = r"""
Fitting a normal distribution $\mathcal N(\mu, \sigma^2)$ by maximum likelihood means choosing the parameters that make the observed data most probable. The log-likelihood of $n$ independent observations is

$$\ell(\mu, \sigma^2) = -\frac n2 \log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

Setting both partial derivatives to zero gives the estimates $\hat\mu$ and $\hat\sigma^2$.

Write `gaussian_mle(x)` that returns the tuple `(mu, var, loglik)` as three Python floats: the two estimates and the **maximised** log-likelihood $\ell(\hat\mu, \hat\sigma^2)$.

## Requirements
- `x`: 1-D float array with $n \ge 2$ values, not all equal.
- Use natural logarithms.
"""

STARTER = '''import numpy as np

def gaussian_mle(x):
    """Return (mu_hat, var_hat, max_log_likelihood) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gaussian_mle(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.mean()
    var = ((x - mu) ** 2).mean()                  # MLE divides by n, not n - 1
    # At the optimum the squared-error term equals n/2, so the log-likelihood simplifies:
    loglik = -0.5 * n * (np.log(2 * np.pi * var) + 1)
    return float(mu), float(var), float(loglik)
'''

HINTS = [
    'Differentiate $\\ell$ with respect to $\\mu$: the root is the sample mean.',
    'Differentiate with respect to $\\sigma^2$ (treat it as one variable). The root is the average squared deviation, dividing by $n$, not $n-1$.',
    'Plug the estimates back in. $\\sum (x_i - \\hat\\mu)^2 = n\\hat\\sigma^2$, so the second term is exactly $-n/2$.',
]

_r = np.random.default_rng(101)
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0]),),
    (np.array([-1.0, 1.0]),),
    (np.array([10.0, 10.0, 10.0, 11.0]),),
    (np.array([1e6 + 0.5, 1e6 - 0.5, 1e6 + 1.5]),),
    (np.array([0.001, 0.002, 0.004]),),
    (_r.normal(5.0, 2.0, size=500),),
    (_r.normal(-3.0, 0.1, size=12),),
]

EXPLANATION = r"""
## Why the estimates come out this way
$\partial\ell/\partial\mu = \frac{1}{\sigma^2}\sum(x_i - \mu) = 0$ gives $\hat\mu = \bar x$, whatever $\sigma^2$ is. Then $\partial\ell/\partial\sigma^2 = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum(x_i - \hat\mu)^2 = 0$ gives $\hat\sigma^2 = \frac1n\sum(x_i - \bar x)^2$.

## Biased but consistent
The MLE variance is biased low by a factor $(n-1)/n$, because $\bar x$ was fitted to the same data. It still converges to the truth as $n \to \infty$. Maximum likelihood optimises fit to the observed data, not unbiasedness.

## Link to least squares and cross-entropy
For fixed $\sigma$, maximising this likelihood over $\mu$ is minimising squared error, so MSE regression is Gaussian maximum likelihood. Likewise, minimising cross-entropy is maximum likelihood for a categorical output.

## Complexity
$O(n)$.

## Common bugs
- `ddof=1`, which gives the unbiased estimate, not the MLE.
- Using $\sigma$ where $\sigma^2$ is needed in $\log(2\pi\sigma^2)$.
- Summing log densities without the simplification is fine, but watch the sign and the $\frac12$.

## Interview follow-ups
- Derive the MLE for a Bernoulli parameter, then explain why it fails with zero successes and how a Beta prior fixes it.
- Why do we maximise log-likelihood rather than likelihood? (Sums are easier to differentiate than products, and a product of many densities underflows to 0.)
"""
