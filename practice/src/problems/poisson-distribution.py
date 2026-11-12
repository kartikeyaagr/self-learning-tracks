import numpy as np

META = dict(title='Poisson pmf and cdf for arrival counts', tags=['Probability'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:10.05'], related=['binomial-distribution', 'exponential-distribution'])

STATEMENT = r"""
If events arrive independently at an average rate of $\lambda$ per window (requests per second, typos per page), the count in one window is Poisson:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k = 0, 1, 2, \dots$$

Write `poisson_dist(lam, ks)` that returns the tuple `(pmf, cdf)`, where for each requested count `ks[i]`:

- `pmf[i]` $= P(X = k_i)$,
- `cdf[i]` $= P(X \le k_i)$.

## Requirements
- `lam`: float $\ge 0$. `ks`: 1-D int array of non-negative counts, in any order, possibly empty or with repeats.
- Both outputs are float arrays with the same shape as `ks`.
- It must work for large $\lambda$ and $k$ (for example $\lambda = 1000$), where $\lambda^k$ and $k!$ overflow. Work in **log space**.
- $\lambda = 0$ is a point mass at 0: $P(X = 0) = 1$.
- No Python loops over counts.
"""

STARTER = '''import numpy as np

def poisson_dist(lam, ks):
    """lam >= 0, ks: (m,) non-negative ints. Return (pmf, cdf), each of shape (m,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def poisson_dist(lam, ks):
    ks = np.asarray(ks, dtype=int)
    kmax = int(ks.max()) if ks.size else 0
    k = np.arange(kmax + 1)
    if lam == 0:
        full = (k == 0).astype(float)
    else:
        logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, kmax + 1)))])   # log(k!)
        full = np.exp(k * np.log(lam) - lam - logfact)
    # Build the pmf on 0..kmax once; the cdf is its running sum. Then look up the requested counts.
    return full[ks], np.cumsum(full)[ks]
'''

HINTS = [
    '$\\log P(X = k) = k \\log \\lambda - \\lambda - \\log k!$. Each term stays moderate even when $\\lambda^k$ overflows.',
    'The cdf at $k$ needs every pmf value from 0 to $k$. Compute the pmf on the whole range `0..max(ks)` once, take a cumulative sum, then index both arrays with `ks`.',
    'Special-case $\\lambda = 0$ (log of 0) and empty `ks` (`ks.max()` raises on an empty array).',
]

TESTS = [
    (2.0, np.array([0, 1, 2, 3])),
    (0.5, np.array([4, 0, 4])),
    (0.0, np.array([0, 1, 5])),
    (3.0, np.array([], dtype=int)),
    (1000.0, np.array([900, 1000, 1100])),
    (12.5, np.arange(0, 40, 3)),
    (1e-3, np.array([0, 1, 2])),
]

EXPLANATION = r"""
## Where it comes from
Split the window into $n$ tiny slots, each holding an event with probability $\lambda/n$. The count is Binomial$(n, \lambda/n)$, and as $n \to \infty$ its pmf tends to $\lambda^k e^{-\lambda}/k!$. That's why the Poisson fits rare independent events, and why the mean and the variance are both $\lambda$.

## Why log space
At $\lambda = 1000$, $\lambda^{1000} = 10^{3000}$ overflows and $e^{-1000}$ underflows, so the direct formula is `inf * 0`. The logs are around $\pm 6000$, and the answer near the peak is about $0.0126$. For speed you can also build the pmf with the recurrence $P(k) = P(k-1)\,\lambda/k$, but starting from $e^{-\lambda}$ underflows when $\lambda$ is large.

## Complexity
$O(K + m)$ where $K = \max k$: one pass over the range, then $m$ lookups.

## Common bugs
- Computing `lam ** k / math.factorial(k)` and overflowing.
- Computing the cdf separately for each requested $k$, which costs $O(mK)$.
- Forgetting the empty and $\lambda = 0$ cases.

## Interview follow-ups
- A service normally gets 4 errors a minute. This minute it got 12. How surprising is that? ($P(X \ge 12) = 1 - \text{cdf}(11) \approx 0.0009$.)
- Real request counts usually have variance greater than their mean. What model would you use? (Negative binomial, a Poisson whose rate is itself Gamma-distributed.)
"""
