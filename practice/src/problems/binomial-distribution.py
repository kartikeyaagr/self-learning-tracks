import numpy as np

META = dict(title='Binomial pmf and cdf for large n', tags=['Probability'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:10.05'], related=['permutations-and-combinations', 'poisson-distribution', 'b-test-significance'])

STATEMENT = r"""
The number of successes in $n$ independent trials, each succeeding with probability $p$, follows a binomial distribution:

$$P(X = j) = \binom{n}{j} p^{j} (1-p)^{n-j}, \qquad j = 0, 1, \dots, n$$

Write `binomial_dist(n, p, k)` that returns the tuple `(pmf, cdf_k)`:

- `pmf`: float array of shape `(n + 1,)` with `pmf[j]` $= P(X = j)$,
- `cdf_k`: Python float, $P(X \le k)$.

## Requirements
- `n` is a non-negative int, `p` is a float in $[0, 1]$, and `k` is any int. For $k < 0$, `cdf_k` is `0.0`. For $k \ge n$, it is the sum of the whole pmf (1 up to rounding).
- It must work for $n$ in the thousands. $\binom{2000}{1000}$ is about $10^{600}$ and overflows a float, so work in **log space** and exponentiate at the end.
- Handle $p = 0$ and $p = 1$ exactly: all the mass is on $j = 0$ or on $j = n$.
- Vectorise over $j$. Don't loop in Python over the $n + 1$ outcomes.
"""

STARTER = '''import numpy as np

def binomial_dist(n, p, k):
    """Return (pmf of shape (n + 1,), P(X <= k) as a float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def binomial_dist(n, p, k):
    j = np.arange(n + 1)
    if p in (0.0, 1.0):                                  # log(0) would give 0 * -inf = nan
        pmf = (j == (0 if p == 0.0 else n)).astype(float)
    else:
        logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])   # log(i!) for i = 0..n
        log_pmf = logfact[n] - logfact[j] - logfact[n - j] + j * np.log(p) + (n - j) * np.log1p(-p)
        pmf = np.exp(log_pmf)
    cdf_k = 0.0 if k < 0 else float(pmf[:k + 1].sum())  # slicing past the end is safe
    return pmf, cdf_k
'''

HINTS = [
    'Take logs: $\\log P(X=j) = \\log n! - \\log j! - \\log (n-j)! + j \\log p + (n-j) \\log(1-p)$. Every term is a moderate number even for huge $n$.',
    'A table of $\\log i!$ for $i = 0..n$ is a cumulative sum of `np.log(np.arange(1, n + 1))` with a 0 in front. Index it with `j` and `n - j` to vectorise.',
    'Special-case $p \\in \\{0, 1\\}$ before taking logs. For the cdf, `pmf[:k + 1].sum()` handles $k \\ge n$, and return `0.0` for negative $k$.',
]

TESTS = [
    (4, 0.5, 2),
    (3, 0.2, 0),
    (0, 0.3, 0),
    (5, 0.0, 0),
    (5, 1.0, 4),
    (10, 0.7, -1),
    (10, 0.7, 25),
    (2000, 0.5, 1000),
    (500, 0.01, 3),
]

EXPLANATION = r"""
## Why log space
The pmf is a product of a gigantic binomial coefficient and tiny powers. At $n = 2000$ the coefficient overflows to `inf` and $0.5^{2000}$ underflows to 0, so the direct formula gives `inf * 0 = nan`. Their logs are about $+1382$ and $-1386$, perfectly representable, and the sum is a modest number to exponentiate.

`np.log1p(-p)` computes $\log(1-p)$ accurately when $p$ is tiny, where `np.log(1 - p)` would lose digits.

## Sanity checks
The mean is $np$ and the variance $np(1-p)$. For large $n$ the pmf looks like a normal curve, and for large $n$ with small $p$ it looks like a Poisson with $\lambda = np$ (see the $n = 500, p = 0.01$ case).

## Complexity
$O(n)$ time and memory: one cumulative sum for the log-factorials and one vectorised pass.

## Common bugs
- Calling `math.comb(n, j) * p**j * ...` in a loop: slow, and the float conversion overflows.
- Using `np.log(p)` with $p = 0$ and getting `nan` from `0 * -inf`.
- An off-by-one in the cdf slice: $P(X \le k)$ includes $j = k$, so slice to `k + 1`.

## Interview follow-ups
- How would you compute an upper tail $P(X \ge k)$ that is around $10^{-20}$ without cancellation? (Sum the tail directly instead of computing $1 - \text{cdf}$, or use log-sum-exp over the tail terms.)
- Derive the mean $np$ using indicator variables, with no sums over the pmf.
"""
