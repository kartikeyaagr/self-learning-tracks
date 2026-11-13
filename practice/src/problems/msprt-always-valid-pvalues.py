import numpy as np

META = dict(title='Always-valid p-values for A/B tests you can peek at', tags=['MLOps', 'Probability and Statistics'], diff='Hard', runtime='numpy',
            lessons=[], related=['b-test-significance', 'canary-mann-whitney'])

STATEMENT = r"""
Checking a fixed-horizon A/B test every day and stopping the first time $p < 0.05$ inflates the false-positive rate far above 5%. The **mixture sequential probability ratio test** (mSPRT) gives p-values that stay valid no matter how often you look or when you stop.

Observations arrive as paired differences $x_1, x_2, \dots$ (treatment minus control for one matched pair of users), assumed independent normal with known variance $\sigma^2$ and unknown mean $\theta$. Test $H_0: \theta = \theta_0$ with a normal mixture $\theta \sim \mathcal N(\theta_0, \tau^2)$ over the alternative. After $n$ observations with mean $\bar x_n$, the mixture likelihood ratio is

$$\Lambda_n = \sqrt{\frac{\sigma^2}{\sigma^2 + n\tau^2}}\;\exp\!\left(\frac{n^2\tau^2(\bar x_n - \theta_0)^2}{2\sigma^2(\sigma^2 + n\tau^2)}\right)$$

and the always-valid p-value is $p_0 = 1$, $p_n = \min\left(p_{n-1},\ 1/\Lambda_n\right)$.

Write `msprt(x, sigma2, tau2, theta0=0.0, alpha=0.05)` that returns the tuple `(p_values, stop)`.

## Output
- `p_values`: float array of shape `(N,)`, where entry $n-1$ is $p_n$ (the value after $n$ observations). It is non-increasing and within $[0, 1]$ (it may underflow to exactly 0 for overwhelming evidence).
- `stop`: the first 0-based index $i$ with `p_values[i] <= alpha`, or `-1` if there is none.

## Requirements
- `x` is a 1-D float array, $N \ge 1$. `sigma2 > 0`, `tau2 > 0`.
- Vectorised (no Python loop over observations).
- Work in log space: for long streams $\Lambda_n$ overflows a float. Compute $\log\Lambda_n$ and use $1/\Lambda_n = e^{-\log\Lambda_n}$, capped at 1.
"""

STARTER = '''import numpy as np

def msprt(x, sigma2, tau2, theta0=0.0, alpha=0.05):
    """Return (p_values (N,), stop index or -1)."""
    x = np.asarray(x, dtype=float)
    n = np.arange(1, len(x) + 1)
    xbar = np.cumsum(x) / n
    lam = np.sqrt(sigma2 / (sigma2 + n * tau2)) * np.exp(n * tau2 * (xbar - theta0) ** 2 / (2 * sigma2))
    p = np.minimum(1.0, 1.0 / lam)            # not the running minimum, and the exponent is wrong
    hits = np.nonzero(p <= alpha)[0]
    return p, int(hits[0]) if len(hits) else -1
'''

SOLUTION = '''import numpy as np

def msprt(x, sigma2, tau2, theta0=0.0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    n = np.arange(1, len(x) + 1, dtype=float)
    xbar = np.cumsum(x) / n                                   # running mean after n observations
    v = sigma2 + n * tau2
    log_lam = 0.5 * np.log(sigma2 / v) + n ** 2 * tau2 * (xbar - theta0) ** 2 / (2 * sigma2 * v)
    inv = np.exp(-np.maximum(log_lam, 0.0))                   # 1 / Lambda, capped at 1 (p_0 = 1)
    p = np.minimum.accumulate(inv)                            # p_n = min(p_{n-1}, 1 / Lambda_n)
    hits = np.flatnonzero(p <= alpha)
    return p, int(hits[0]) if hits.size else -1
'''

HINTS = [
    'Get every $\\bar x_n$ at once with `np.cumsum(x) / n` where `n = np.arange(1, N + 1)`. Then $\\log\\Lambda_n$ is a vectorised expression: $\\tfrac12\\log\\frac{\\sigma^2}{\\sigma^2 + n\\tau^2} + \\frac{n^2\\tau^2(\\bar x_n - \\theta_0)^2}{2\\sigma^2(\\sigma^2 + n\\tau^2)}$. Note the $n^2$.',
    'Since $p_0 = 1$, $p_n = \\min(1, 1/\\Lambda_1, \\dots, 1/\\Lambda_n)$. `np.exp(-np.maximum(log_lam, 0))` applies the cap at 1 without overflow, and `np.minimum.accumulate` gives the running minimum.',
    'The stopping index is the first position of `p <= alpha`: `np.flatnonzero(p <= alpha)`, or `-1` when it is empty.',
]

_rng = np.random.default_rng(42)
_null = _rng.normal(0.0, 1.0, 2000)
_eff = _rng.normal(0.15, 1.0, 2000)

TESTS = [
    (np.array([1.0, 1.0, 1.0, 1.0]), 1.0, 1.0),
    (np.array([0.5, -0.5, 0.5, -0.5]), 1.0, 0.5),
    (np.array([2.0, 2.5, 1.5, 3.0, 2.0]), 1.0, 1.0, 0.0, 0.01),
    (np.array([0.0]), 2.0, 0.1),
    (np.array([1.2, 0.9, 1.1, 1.0, 1.05]), 0.04, 0.01, 1.0),
    (np.full(20000, 0.5), 1.0, 0.1),
    (_null, 1.0, 0.01),
    (_eff, 1.0, 0.01),
    (_eff, 1.0, 0.01, 0.0, 0.001),
]

EXPLANATION = r"""
## Why it works
Under $H_0$ the mixture likelihood ratio $\Lambda_n$ is a non-negative martingale with mean 1. Ville's inequality bounds the chance that it *ever* reaches $1/\alpha$: $P(\sup_n \Lambda_n \ge 1/\alpha) \le \alpha$. So "stop the first time $p_n \le \alpha$" has a false-positive rate of at most $\alpha$ however often you peek. The running minimum makes $p_n$ monotone, so once a result is significant it stays significant.

The formula comes from integrating the normal likelihood ratio $\prod_i \phi(x_i; \theta)/\phi(x_i; \theta_0)$ over $\theta \sim \mathcal N(\theta_0, \tau^2)$; only $\bar x_n$ survives, which is why a cumulative sum is all you need. $\tau^2$ is a tuning knob: it should match the size of effects you expect, and it controls how quickly the test can detect them.

## Complexity
$O(N)$ time and memory: one `cumsum`, a few element-wise operations, one `minimum.accumulate`.

## Common bugs
- $n$ instead of $n^2$ in the exponent (the starter). With $\bar x_n$ the sum $\sum x_i = n\bar x_n$ is what enters the likelihood, which squares to $n^2\bar x_n^2$.
- Computing $\Lambda_n$ directly: for test 6 it reaches about $e^{2500}$ and overflows to `inf`, along with overflow warnings.
- Returning $\min(1, 1/\Lambda_n)$ without the running minimum: the p-value can go back up, so "significant" flickers on and off.
- Plugging in the observed variance as if it were known without saying so. In practice $\sigma^2$ is estimated from a pre-period or a burn-in, and the estimate should be stable.

## Interview follow-ups
- You run a standard two-sample A/B test (not paired). What is $\sigma^2$ for the difference in means of one treatment and one control user? ($\sigma_T^2 + \sigma_C^2$, with observations taken in pairs as users arrive.)
- What's the price of always-valid inference compared with a fixed-horizon test planned for the same $n$? (Lower power at that fixed $n$, i.e. a larger sample is needed on average to detect a given effect, in exchange for being able to stop early.)
"""
