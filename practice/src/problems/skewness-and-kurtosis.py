import numpy as np

META = dict(title='Skewness and excess kurtosis', tags=['Probability and Statistics'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:10.15'], related=['sample-variance-and-standard-deviation', 'discrete-moments'])

STATEMENT = r"""
Skewness measures asymmetry and kurtosis measures how heavy the tails are. Both are built from the **central moments** of a sample,

$$m_k = \frac1n \sum_{i=1}^n (x_i - \bar x)^k$$

The plain moment estimators are

$$g_1 = \frac{m_3}{m_2^{3/2}}, \qquad g_2 = \frac{m_4}{m_2^{2}} - 3$$

where $g_2$ is the **excess** kurtosis (0 for a normal distribution). Both are biased in small samples. The standard adjusted versions are

$$G_1 = g_1\,\frac{\sqrt{n(n-1)}}{n-2}, \qquad G_2 = \frac{(n-1)\big((n+1)\,g_2 + 6\big)}{(n-2)(n-3)}$$

Write `shape_stats(x)` that returns the tuple `(g1, g2, G1, G2)` as four Python floats.

## Requirements
- `x`: 1-D float array with $n \ge 4$ values, not all equal.
- Every moment uses the $1/n$ normalisation shown above (no `ddof`).
"""

STARTER = '''import numpy as np

def shape_stats(x):
    """x: (n,), n >= 4, not constant. Return (g1, g2, G1, G2) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def shape_stats(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    g1 = m3 / m2 ** 1.5
    g2 = m4 / m2 ** 2 - 3.0
    G1 = g1 * np.sqrt(n * (n - 1)) / (n - 2)
    G2 = (n - 1) * ((n + 1) * g2 + 6) / ((n - 2) * (n - 3))
    return float(g1), float(g2), float(G1), float(G2)
'''

HINTS = [
    'Centre the data once and reuse the deviations for $m_2$, $m_3$ and $m_4$.',
    'All three moments divide by $n$. Using `np.var(x, ddof=1)` for $m_2$ quietly changes the answer.',
    'The adjusted formulas only need $n$ and the plain $g_1$, $g_2$. Remember the $-3$ in $g_2$ and that $G_2$ uses $g_2$, not $m_4/m_2^2$.',
]

_r = np.random.default_rng(42)
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0, 10.0]),),
    (np.array([-2.0, -1.0, 1.0, 2.0]),),
    (np.array([0.0, 0.0, 0.0, 1.0]),),
    (np.array([5.0, 1.0, 5.0, 1.0, 5.0, 1.0]),),
    (np.array([1e6 + 1.0, 1e6 + 2.0, 1e6 + 4.0, 1e6 + 8.0]),),
    (_r.exponential(2.0, size=300),),
    (_r.standard_t(3, size=150),),
    (-_r.lognormal(0.0, 0.5, size=60),),
]

EXPLANATION = r"""
## What the numbers mean
Dividing by $m_2^{k/2}$ makes both measures unitless and unchanged by shifting or rescaling the data. A long right tail pushes $g_1$ above 0 (exponential data gives about 2). Excess kurtosis is driven by the fourth power, so a few extreme points dominate it: Student-t with 3 degrees of freedom has infinite population kurtosis, and the sample value jumps around from run to run. A two-point distribution such as `[5, 1, 5, 1, 5, 1]` has the smallest possible kurtosis, $g_2 = -2$.

## Why adjust
$g_1$ and $g_2$ are ratios of biased moment estimates, so they are biased toward 0 for small $n$. $G_1$ and $G_2$ are the versions spreadsheets and most stats packages report by default (`scipy.stats.skew(x, bias=False)`, pandas `.skew()` and `.kurt()`). The two agree as $n \to \infty$.

## Complexity
$O(n)$ time, one pass after centring.

## Common bugs
- Reporting plain kurtosis $m_4/m_2^2$ when the question wants excess kurtosis, or the other way round.
- Mixing $1/n$ and $1/(n-1)$ moments in the same ratio.
- Raw (uncentred) moments: $E[x^3]$ is not skewness, and it loses precision with a large offset.

## Interview follow-ups
- A feature has skewness 8. What would you do before feeding it to a linear model? (Log or Box-Cox transform, or quantile binning.)
- How would you test whether data is normal using these two numbers? (Jarque-Bera: $\frac{n}{6}\left(g_1^2 + g_2^2/4\right)$ is approximately $\chi^2_2$.)
"""
