import numpy as np

META = dict(title='Column-wise variance and standard deviation', tags=['Probability and Statistics'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:10.15', 'mlmaths:12.01'], related=['mean-median-mode', 'skewness-and-kurtosis'])

STATEMENT = r"""
For $n$ observations of one feature, the variance with a degrees-of-freedom correction $\delta$ (`ddof`) is

$$s^2 = \frac{1}{n - \delta}\sum_{i=1}^{n} (x_i - \bar x)^2, \qquad s = \sqrt{s^2}$$

$\delta = 1$ gives the unbiased **sample** variance, and $\delta = 0$ gives the **population** (maximum-likelihood) variance.

Write `spread(X, ddof=1)` that returns the tuple `(var, std)` for every column of a 2-D array.

## Requirements
- `X`: float array of shape `(n, d)`, rows are observations. `n > ddof` always holds.
- `var` and `std` are float arrays of shape `(d,)`.
- It must stay accurate when the data has a huge offset: a column like `1e9 + [1, 2, 3]` has sample variance exactly 1. The shortcut $\frac{1}{n}\sum x_i^2 - \bar x^2$ fails here.
- No Python loops over columns.
"""

STARTER = '''import numpy as np

def spread(X, ddof=1):
    """X: (n, d). Return (var, std), each of shape (d,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def spread(X, ddof=1):
    X = np.asarray(X, dtype=float)
    dev = X - X.mean(axis=0)                       # two-pass: centre first, then square
    var = (dev ** 2).sum(axis=0) / (len(X) - ddof)
    return var, np.sqrt(var)
'''

HINTS = [
    'Reduce over `axis=0` so you get one number per column.',
    'Subtract the column means first, then square and sum. Dividing by `n - ddof` is the only place `ddof` appears.',
    'Squaring the raw values of `1e9 + 1` loses the small part to rounding before you subtract anything. Centring first keeps the deviations small and exact.',
]

_r = np.random.default_rng(3)
TESTS = [
    (np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]),),
    (np.array([[2.0], [4.0], [4.0], [4.0], [5.0], [5.0], [7.0], [9.0]]), 0),
    (np.array([[1e9 + 1.0], [1e9 + 2.0], [1e9 + 3.0]]),),
    (np.array([[3.0, -1.0, 0.0], [3.0, 1.0, 0.0]]),),
    (np.array([[5.0, 6.0]]), 0),
    (_r.normal(0, 3, size=(50, 4)),),
    (_r.normal(-100, 0.01, size=(30, 2)), 0),
]

EXPLANATION = r"""
## Why divide by n - 1
The deviations $x_i - \bar x$ are measured from the sample's own mean, which is the point that makes their sum of squares as small as possible. That makes the raw sum of squares too small on average by exactly one observation's worth of variance, $E\left[\sum (x_i - \bar x)^2\right] = (n-1)\sigma^2$. Dividing by $n-1$ (Bessel's correction) removes the bias. The standard deviation $s$ is still slightly biased, because $\sqrt{\cdot}$ is concave.

## Why two passes
$\frac1n\sum x_i^2 - \bar x^2$ subtracts two numbers near $10^{18}$ to get a number near 1. Float64 keeps about 16 significant digits, so the answer is lost to cancellation and can even come out negative. Centring first means you only ever square small numbers. For streaming data, Welford's algorithm does the same in one pass.

## Complexity
$O(nd)$ time and $O(nd)$ memory for the centred copy.

## Common bugs
- `np.var` and `np.std` default to `ddof=0`, while pandas `.var()` and `.std()` default to `ddof=1`. Mixing them is a classic off-by-one.
- Reducing over `axis=1` (rows) instead of columns.

## Interview follow-ups
- Derive Welford's online update for the mean and the sum of squared deviations.
- How do you merge the variances of two shards computed separately? (Chan's parallel formula: combine counts, means and $M_2$ with a correction term $\delta^2 n_a n_b / n$.)
"""
