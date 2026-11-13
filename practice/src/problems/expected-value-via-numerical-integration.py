import numpy as np

META = dict(title="Moments of a density with Simpson's rule", tags=['Calculus', 'Probability'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:10.04', 'mlmaths:09.06'],
            related=['kl-divergence-discrete-and-continuous', 'area-under-roc-curve-via-trapezoidal-rule'])

STATEMENT = r"""
You often have a density only as values on a grid: a posterior evaluated pointwise, a histogram-smoothed estimate, an unnormalised energy model. Its normaliser, mean and variance are all integrals:

$$Z = \int p(x)\,dx, \qquad \mu = \frac1Z\int x\,p(x)\,dx, \qquad \sigma^2 = \frac1Z \int (x - \mu)^2\,p(x)\,dx$$

Write `simpson_moments(x, p)` that approximates all three with the **composite Simpson's rule** and returns the tuple `(Z, mean, var)` of Python floats.

## Composite Simpson's rule
On $m + 1$ equally spaced points $x_0, \dots, x_m$ with spacing $h$ and $m$ even,

$$\int_{x_0}^{x_m} f\,dx \approx \frac h3\Bigl(f_0 + 4f_1 + 2f_2 + 4f_3 + \dots + 2f_{m-2} + 4f_{m-1} + f_m\Bigr)$$

## Inputs and conventions
- `x`: float array, strictly increasing and equally spaced, with an **odd** number of points $\ge 3$.
- `p`: non-negative float array of the same length (not necessarily normalised), with $Z > 0$.
- Compute the variance as the integral of $(x - \mu)^2 p$ (the centred form), not as $E[X^2] - \mu^2$.
- Use `h = x[1] - x[0]`. Don't call `scipy` or `np.trapezoid`.
"""

STARTER = '''import numpy as np

def simpson_moments(x, p):
    """Return (Z, mean, variance) of the grid density p(x) using composite Simpson's rule."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def simpson_moments(x, p):
    x, p = np.asarray(x, dtype=float), np.asarray(p, dtype=float)
    h = x[1] - x[0]
    wts = np.ones(len(x))
    wts[1:-1:2] = 4                      # odd interior points
    wts[2:-1:2] = 2                      # even interior points
    integrate = lambda f: h / 3 * np.dot(wts, f)
    Z = integrate(p)
    mean = integrate(x * p) / Z
    var = integrate((x - mean) ** 2 * p) / Z
    return float(Z), float(mean), float(var)
'''

HINTS = [
    'All three quantities are the same rule applied to different integrands: $p$, $x\\,p$ and $(x-\\mu)^2 p$. Build the rule once as a weight vector.',
    'The weights are `[1, 4, 2, 4, 2, ..., 4, 1]` times $h/3$. Slicing with a step of 2 sets them: odd positions get 4, even interior positions get 2.',
    '`w = np.ones(n); w[1:-1:2] = 4; w[2:-1:2] = 2`, then each integral is `h / 3 * w @ f`. Divide by $Z$ for the mean and variance, and compute the mean before the variance.',
]

_g = np.linspace(-8, 8, 161)
_r = np.random.default_rng(2)
_xr = np.linspace(-3, 5, 41)
TESTS = [
    (np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])),
    (np.linspace(0, 1, 5), np.linspace(0, 1, 5) ** 2),
    (_g, np.exp(-0.5 * _g ** 2)),
    (_g, 3.0 * np.exp(-0.5 * ((_g - 1.5) / 0.7) ** 2)),
    (np.linspace(0, 30, 301), np.exp(-np.linspace(0, 30, 301) / 2)),
    (_g, np.exp(-2 * (_g - 3) ** 2) + 0.5 * np.exp(-2 * (_g + 3) ** 2)),
    (np.linspace(-1, 1, 3), np.array([0.0, 5.0, 0.0])),
    (_xr, _r.uniform(0, 1, size=41)),
]

EXPLANATION = r"""
## Why Simpson
Simpson fits a parabola through each triple of points and integrates it exactly. That makes it exact for cubics (test 2: $\int_0^1 x^2\,dx = 1/3$ comes out exact), with error $O(h^4)$ on smooth integrands versus $O(h^2)$ for the trapezoid rule. For the standard normal on $[-8, 8]$ with $h = 0.1$, $Z$ matches $\sqrt{2\pi}$ to about 12 digits.

## Why the centred variance
$E[X^2] - \mu^2$ subtracts two large numbers when $|\mu| \gg \sigma$, which loses precision (a density at $\mu = 10^6$ with $\sigma = 1$ loses about 12 digits). Integrating $(x - \mu)^2 p$ never forms those large terms.

## Complexity
Three weighted sums over the grid: $O(m)$.

## Common bugs
- Swapping the 4s and 2s (the 4s go on the **odd** indices, the midpoints of each parabola).
- Using an even number of points, which leaves one interval without a partner.
- Forgetting to divide by $Z$ when $p$ isn't normalised.

## Interview follow-ups
- For a Gaussian, the trapezoid rule on a wide uniform grid is also spectacularly accurate. Why? (The integrand and all its derivatives vanish at both ends, so the Euler-Maclaurin error terms vanish.)
- In 20 dimensions, a grid with 100 points per axis has $10^{40}$ points. What do you use instead, and how does its error scale with the number of samples?
"""
