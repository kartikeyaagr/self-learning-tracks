import numpy as np

META = dict(title='Bernoulli and geometric distributions: PMF, CDF and moments', tags=['Probability'], diff='Easy',
            runtime='numpy', related=['binomial-distribution', 'discrete-moments', 'poisson-distribution'])

STATEMENT = r"""
Each trial succeeds with probability $p$ (a Bernoulli trial: mean $p$, variance $p(1-p)$). Let $K$ be the number of the trial on which the **first** success happens, $K \in \{1, 2, \dots\}$. Then $K$ is geometric:

$$P(K = k) = (1-p)^{k-1} p, \qquad P(K \le k) = 1 - (1-p)^k.$$

Write `geometric(k, p)` where `k` is an array of positive integers. Return a dict with:
- `"pmf"`: array of $P(K = k)$ for each entry,
- `"cdf"`: array of $P(K \le k)$,
- `"mean"`: $1/p$,
- `"var"`: $(1-p)/p^2$,
- `"bernoulli"`: a tuple `(mean, var)` for a single trial, $(p, p(1-p))$.

Assume $0 < p \le 1$. Values are floats.
"""

STARTER = '''import numpy as np

def geometric(k, p):
    """Return {"pmf", "cdf", "mean", "var", "bernoulli"} for a geometric distribution on {1, 2, ...}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def geometric(k, p):
    k = np.asarray(k, dtype=float)
    q = 1.0 - p
    return {"pmf": q ** (k - 1) * p, "cdf": 1.0 - q ** k, "mean": 1.0 / p,
            "var": q / p ** 2, "bernoulli": (float(p), float(p * q))}
'''

HINTS = [
    'The first success on trial k means k - 1 failures in a row, then a success.',
    'The CDF is easiest through the complement: K > k means the first k trials all failed.',
    'Mean and variance have closed forms; there is no need to sum a series.',
]

TESTS = [
    (np.array([1, 2, 3]), 0.5),
    (np.array([1, 5, 10, 20]), 0.1),
    (np.array([1, 2]), 1.0),
    (np.array([3]), 0.25),
    (np.arange(1, 8), 0.3),
    (np.array([100]), 0.01),
]

EXPLANATION = r"""
## Where the mean comes from
$\mathbb{E}[K] = \sum_{k\ge1} k q^{k-1} p = p / (1-q)^2 = 1/p$. Intuitively, if 10% of trials succeed you wait 10 trials on average.

## Memorylessness
$P(K > m + n \mid K > m) = P(K > n)$: having failed $m$ times tells you nothing about how much longer you will wait. The geometric is the only memoryless distribution on the positive integers, the discrete counterpart of the exponential.

## Where it appears in ML
- Number of retries until a request succeeds; number of sampled tokens until an end-of-sequence token.
- Expected number of rejection-sampling attempts, $1/p$ for acceptance rate $p$.
- Speculative decoding: the number of accepted draft tokens is a truncated geometric when each token is accepted independently.

## Common bugs
- Mixing up the two conventions: some texts count failures before the first success ($K-1$, support starting at 0), whose mean is $(1-p)/p$.
- Computing the CDF by summing the PMF in a loop; the closed form is exact and faster.
"""
