import numpy as np

META = dict(title='Exact sampling distribution of the mean', tags=['Probability and Statistics', 'Probability'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:12.01'], related=['central-limit-theorem', 'standard-error-comparison'])

STATEMENT = r"""
Draw $n$ values **with replacement** from a small population, each draw equally likely to pick any of the $N$ entries, and average them. The sample mean $\bar X_n$ is itself a random variable. For an integer-valued population you can compute its distribution exactly.

Write `mean_distribution(population, n)` that returns the tuple `(means, probs)`:

- `means`: float array of every value $\bar X_n$ can take with positive probability, sorted ascending,
- `probs`: float array of the same length, `probs[i]` $= P(\bar X_n = $ `means[i]`$)$.

## Requirements
- `population`: 1-D int array with $N \ge 1$ entries. Values may be negative and may repeat (a repeated value is simply more likely to be drawn).
- `n`: int $\ge 1$.
- Leave out impossible means: for population `[0, 2]` the sum of draws is always even, so only some grid points have positive probability.
- The distribution of a sum of independent draws is a **convolution**. Work with integer sums and divide by $n$ at the end, so equal means compare equal.
"""

STARTER = '''import numpy as np

def mean_distribution(population, n):
    """Return (means, probs): the exact distribution of the mean of n draws with replacement."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def mean_distribution(population, n):
    pop = np.asarray(population, dtype=int)
    lo = int(pop.min())
    one = np.bincount(pop - lo) / len(pop)        # pmf of one draw on the grid lo, lo+1, ..., max
    pmf = np.array([1.0])
    for _ in range(n):
        pmf = np.convolve(pmf, one)               # pmf of the running sum (direct method: exact zeros stay 0)
    sums = n * lo + np.arange(len(pmf))
    keep = pmf > 0
    return sums[keep] / n, pmf[keep]
'''

HINTS = [
    'First get the pmf of a single draw on an integer grid starting at the population minimum. `np.bincount(pop - pop.min()) / N` does it, repeats included.',
    'If $S_{n-1}$ has pmf $a$ and one draw has pmf $b$, then $S_n = S_{n-1} + X$ has pmf `np.convolve(a, b)`. Start from `[1.0]` and convolve $n$ times.',
    'Index $i$ of the final pmf is the sum $n \\cdot \\min + i$. Keep entries with positive probability and divide those sums by $n$.',
]

TESTS = [
    (np.array([1, 2, 3]), 2),
    (np.array([0, 2]), 3),
    (np.array([5]), 4),
    (np.array([7, 1, 4]), 1),
    (np.array([0, 0, 0, 1]), 5),
    (np.array([-3, 0, 0, 2]), 3),
    (np.array([1, 2, 3, 4, 5, 6]), 4),
    (np.random.default_rng(12).integers(0, 9, size=10), 6),
]

EXPLANATION = r"""
## Why convolution
$P(S_{n} = s) = \sum_x P(S_{n-1} = s - x)\,P(X = x)$, because the new draw is independent of the earlier ones. That sum over shifted products is exactly a discrete convolution. Each step widens the support by the width of one draw, and the final array covers every possible sum.

## What it shows
For any population the result has mean $\mu$ and variance $\sigma^2/n$, where $\sigma^2$ is the population variance with divisor $N$. Run it on a fair die with $n = 1, 2, 4$: the flat distribution becomes a triangle, then something already bell-shaped. That's the central limit theorem happening in front of you. For a skewed population like `[0, 0, 0, 1]` the bell takes longer to appear.

## Complexity
The final support has $L = n(\max - \min) + 1$ points, so the $n$ convolutions cost $O(n \cdot L \cdot w)$ with $w$ the width of one draw. Enumerating all $N^n$ samples instead would be $6^4 = 1296$ for the dice, but grows exponentially.

## Common bugs
- Using distinct values with equal weight, ignoring that repeated population entries are more likely.
- Building means with floats and calling `np.unique`: $0.1 + 0.2 \ne 0.3$ can split one mean into two.
- Using FFT convolution, which leaves values like $10^{-17}$ where the probability should be exactly 0.

## Interview follow-ups
- How would the variance change if you sampled **without** replacement? (Multiply by the finite population correction $\frac{N-n}{N-1}$.)
- Use this to compute the exact probability that the mean of 10 dice rolls is at least 4.5.
"""
