import numpy as np

META = dict(title='Normal pdf and cdf, accurate in the tails', tags=['Probability'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:10.35', 'mlmaths:10.04'], related=['z-test-one-sample', 'central-limit-theorem'])

STATEMENT = r"""
For $X \sim \mathcal N(\mu, \sigma^2)$, with $z = (x - \mu)/\sigma$:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-z^2/2}, \qquad F(x) = P(X \le x) = \frac12\,\mathrm{erfc}\!\left(-\frac{z}{\sqrt 2}\right)$$

where $\mathrm{erfc}(u) = 1 - \mathrm{erf}(u)$ is the complementary error function (`math.erfc` in Python).

Write `normal_pdf_cdf(x, mu=0.0, sigma=1.0)` that returns the tuple `(pdf, cdf)`.

## Requirements
- `x`: float array of any shape. `mu`: float, `sigma`: float $> 0$. Both outputs have the shape of `x`.
- The cdf must be accurate **relative to its size** in the far left tail. At $z = -10$ the true value is about $7.6 \times 10^{-24}$, and the formula $\frac12(1 + \mathrm{erf}(z/\sqrt2))$ returns exactly 0 there. This problem checks with a relative tolerance and essentially no absolute slack, so that shortcut fails.
- NumPy has no vectorised `erfc`. Apply `math.erfc` element-wise (for example with `np.vectorize`).
"""

STARTER = '''import math
import numpy as np

def normal_pdf_cdf(x, mu=0.0, sigma=1.0):
    """x: float array. Return (pdf, cdf) with the same shape as x."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

_erfc = np.vectorize(math.erfc, otypes=[float])     # otypes keeps empty inputs working

def normal_pdf_cdf(x, mu=0.0, sigma=1.0):
    z = (np.asarray(x, dtype=float) - mu) / sigma
    pdf = np.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))
    cdf = 0.5 * _erfc(-z / math.sqrt(2))            # erfc of a large positive number is tiny but exact
    return pdf, cdf
'''

HINTS = [
    'Standardise first: everything depends on $z = (x - \\mu)/\\sigma$. The pdf also needs the $1/\\sigma$ factor.',
    '`1 + erf(u)` for $u$ near $-7$ adds 1 to a number like $-0.99999999999999999999998$. The result rounds to 0. `erfc(-u)` computes the tiny tail value directly.',
    '`np.vectorize(math.erfc, otypes=[float])` gives an element-wise version that also works on empty arrays.',
]

TESTS = [
    (np.array([-1.0, 0.0, 1.0, 1.96]),),
    (np.array([10.0, 12.0, 7.0]), 10.0, 2.0),
    (np.array([-10.0, -20.0, -30.0]),),
    (np.array([8.0, 15.0]),),
    (np.array([[0.5, -0.5], [3.0, -3.0]]), 0.0, 0.5),
    (np.array([1e6, 1e6 + 3.0]), 1e6, 1.0),
    (np.array([]),),
    (np.linspace(-6.0, 6.0, 13), 1.5, 3.0),
]
META['tol'] = [1e-6, 1e-300]      # relative check only: tail values are far below the default atol

EXPLANATION = r"""
## Why erfc
Floating point has fine resolution near 0 and coarse resolution near 1: the gap between neighbouring doubles is about $10^{-16}$ at 1 but about $10^{-324}$ near 0. A left-tail probability of $10^{-24}$ can be stored exactly, but not as $1 + (\text{something near } -1)$. $\mathrm{erfc}$ computes the tail directly, so `cdf` keeps full relative precision all the way out to about $z = -38$.

The right tail has the mirror problem: $F(8) = 1 - 6 \times 10^{-16}$ is fine, but if you need the **survival** $1 - F(x)$ (a one-sided p-value), compute it as $\frac12 \mathrm{erfc}(z/\sqrt2)$, not as `1 - cdf`.

## Complexity
$O(n)$. `np.vectorize` is a Python-level loop, which is fine here. For speed, `scipy.special.ndtr` is the vectorised version.

## Common bugs
- Forgetting the $1/\sigma$ in the pdf, so it doesn't integrate to 1 when $\sigma \ne 1$.
- Using $\sigma^2$ where $\sigma$ is expected (NumPy's `rng.normal(loc, scale)` takes the standard deviation).
- Writing `erfc(z / sqrt(2))` without the minus sign, which gives the survival function instead of the cdf.

## Interview follow-ups
- How would you compute $\log F(x)$ for $z = -50$, where $F$ underflows? (Use the asymptotic expansion $\log F(z) \approx -z^2/2 - \log(-z) - \tfrac12\log 2\pi$, or `scipy.special.log_ndtr`.)
- Why do probit models and the GELU activation use $\Phi$? (GELU is $x\,\Phi(x)$: it scales each input by the probability that a standard normal falls below it.)
"""
