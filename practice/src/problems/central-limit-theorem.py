import numpy as np

META = dict(title='How fast does the CLT kick in?', tags=['Probability and Statistics', 'Probability'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:12.01', 'mlmaths:10.35'], related=['sampling-distribution-of-mean', 'normal-distribution'])

STATEMENT = r"""
Let $X$ take integer value `values[i]` with probability `probs[i]`, with mean $\mu$ and standard deviation $\sigma$. The central limit theorem says the standardised sum of $n$ independent copies,

$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt n}$$

approaches $\mathcal N(0, 1)$. Measure **how close** it is with the Kolmogorov distance

$$D_n = \sup_{x \in \mathbb R} \big|\,P(Z_n \le x) - \Phi(x)\,\big|$$

where $\Phi(x) = \tfrac12\,\mathrm{erfc}(-x/\sqrt2)$ is the standard normal cdf.

Write `clt_gap(values, probs, n)` that returns $D_n$ as a Python float.

## Requirements
- `values`: 1-D int array of **distinct** integers (any order). `probs`: same length, non-negative, sums to 1, with at least two values having positive probability. `n`: int $\ge 1$.
- $\sigma$ is the exact standard deviation of $X$ (divisor 1, not $n - 1$: this is a distribution, not a sample).
- Compute the exact distribution of $S_n$ (repeated convolution), not a simulation.
- $P(Z_n \le x)$ is a step function, so the supremum is approached just before or at a jump. At each possible value $z_j$ of $Z_n$, compare $\Phi(z_j)$ with **both** the cdf just before the jump and the cdf at the jump.
"""

STARTER = '''import math
import numpy as np

def clt_gap(values, probs, n):
    """Return sup_x |P(Z_n <= x) - Phi(x)| as a float."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

_erfc = np.vectorize(math.erfc, otypes=[float])

def clt_gap(values, probs, n):
    v = np.asarray(values, dtype=int)
    p = np.asarray(probs, dtype=float)
    mu = p @ v
    sigma = math.sqrt(p @ (v - mu) ** 2)
    lo = int(v.min())
    one = np.zeros(int(v.max()) - lo + 1)
    one[v - lo] = p                                   # pmf of X on an integer grid (values are distinct)
    pmf = np.array([1.0])
    for _ in range(n):
        pmf = np.convolve(pmf, one)                   # pmf of S_n
    z = (n * lo + np.arange(len(pmf)) - n * mu) / (sigma * math.sqrt(n))
    after = np.cumsum(pmf)                            # P(Z_n <= z_j)
    before = after - pmf                              # P(Z_n <  z_j), the limit from the left
    phi = 0.5 * _erfc(-z / math.sqrt(2))
    return float(max(np.abs(after - phi).max(), np.abs(before - phi).max()))
'''

HINTS = [
    'Put the pmf of $X$ on an integer grid from `min(values)` to `max(values)`, then `np.convolve` it with itself $n$ times to get the pmf of $S_n$.',
    'Grid index $i$ is the sum $s = n \\cdot \\min + i$, so $z_i = (s - n\\mu)/(\\sigma\\sqrt n)$. The cdf at the jump is `np.cumsum(pmf)`, and just before it is `cumsum - pmf`.',
    'Between jumps the empirical cdf is flat while $\\Phi$ only increases, so the largest gap is at a jump, on one side or the other. Grid points with probability 0 are harmless: both sides are equal there.',
]

TESTS = [
    (np.array([0, 1]), np.array([0.5, 0.5]), 1),
    (np.array([0, 1]), np.array([0.5, 0.5]), 4),
    (np.array([0, 1]), np.array([0.9, 0.1]), 10),
    (np.array([0, 1]), np.array([0.9, 0.1]), 100),
    (np.array([1, 2, 3, 4, 5, 6]), np.full(6, 1 / 6), 3),
    (np.array([3, -1, 0]), np.array([0.2, 0.3, 0.5]), 7),
    (np.array([0, 10]), np.array([0.99, 0.01]), 40),
    (np.array([0, 2, 5]), np.array([0.5, 0.0, 0.5]), 9),
]

EXPLANATION = r"""
## Why check both sides of each jump
The cdf of $Z_n$ jumps at each possible value $z_j$. Just to the left of $z_j$ it equals $P(Z_n < z_j)$; at $z_j$ it equals $P(Z_n \le z_j)$. On the flat stretch between two jumps, $\Phi$ keeps rising, so the gap is largest at one of the stretch's endpoints. Checking only the right-continuous value misses half of them: for one fair coin flip ($n = 1$) the gap just before the jump at $z = 1$ is $\Phi(1) - 0.5 \approx 0.34$, which is the answer.

## What the numbers show
The Berry-Esseen theorem bounds $D_n \le C\,\rho / (\sigma^3 \sqrt n)$ with $\rho = E|X - \mu|^3$, so the gap shrinks like $1/\sqrt n$, more slowly for skewed $X$. Compare the fair coin at $n = 4$ with the 90/10 coin at $n = 10$ and $n = 100$, and the 99/1 variable at $n = 40$: rare events need far more data before a normal approximation is trustworthy. That's the hidden assumption in a z-test on conversion rates.

## Complexity
The support of $S_n$ has $L = n(\max - \min) + 1$ points. $n$ convolutions cost $O(nLw)$, with $w$ the width of one draw.

## Common bugs
- Using the sample formula with $n-1$ for $\sigma$.
- Only comparing $\Phi$ with the cdf at the jumps (`after`), not just before them (`before`).
- Forgetting to standardise the sum, or standardising by $\sigma/\sqrt n$ instead of $\sigma\sqrt n$.

## Interview follow-ups
- A/B test with a 0.5% conversion rate and 1,000 users per arm. Is the normal approximation safe? (The expected count is only 5 per arm, so prefer an exact or simulation-based test.)
- What is a continuity correction, and how would it change the gap for the fair coin at $n = 4$? (Evaluate $\Phi$ at the half-integer between sums, which centres the steps on the curve.)
"""
