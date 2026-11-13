import numpy as np

META = dict(title='Canary judge with a Mann-Whitney U test', tags=['MLOps', 'Probability and Statistics'], diff='Hard', runtime='numpy',
            lessons=['ieng:22.06', 'sdsd:14.04'], related=['msprt-always-valid-pvalues', 'sort-and-argsort'])

STATEMENT = r"""
A new model version gets a small slice of traffic (the **canary**) next to the current version (the **baseline**). Latencies are skewed and heavy-tailed, so automated canary judges compare them with the rank-based **Mann-Whitney U** test instead of a t-test.

Write `canary_judge(baseline, canary, alpha=0.01, min_samples=10)` that returns the tuple `(u, z, p, verdict)`.

## The test
Let $n_b$ and $n_c$ be the sample sizes and $N = n_b + n_c$.

1. Rank the pooled values from 1 (smallest) to $N$. Tied values all get the **average** of the ranks they span.
2. $R_c$ is the sum of the canary's ranks, and $U = R_c - \frac{n_c(n_c + 1)}{2}$. Under $H_0$ (same distribution) its mean is $\mu = \frac{n_b n_c}{2}$.
3. Tie-corrected variance, where $t$ runs over the sizes of the groups of equal values:

$$\sigma^2 = \frac{n_b n_c}{12}\left[(N + 1) - \frac{\sum_t (t^3 - t)}{N(N-1)}\right]$$

4. With a continuity correction: $z = \dfrac{U - \mu - \tfrac12\operatorname{sign}(U - \mu)}{\sigma}$. If $\sigma = 0$ (every value equal), $z = 0$.
5. Two-sided p-value $p = \operatorname{erfc}(|z| / \sqrt 2)$, capped at 1.

## Verdict
- `'inconclusive'` if $n_b <$ `min_samples` or $n_c <$ `min_samples`.
- Otherwise `'fail'` if $p <$ `alpha` and $U > \mu$ (canary values tend to be **larger**, i.e. slower).
- Otherwise `'pass'`.

## Inputs and output
- `baseline`, `canary`: 1-D float arrays. Ties are common (latencies are logged in whole milliseconds).
- If either array is empty, return `(0.0, 0.0, 1.0, 'inconclusive')`.
- `u`, `z`, `p` are Python floats; `verdict` is a str.
"""

STARTER = '''import numpy as np

def canary_judge(baseline, canary, alpha=0.01, min_samples=10):
    """Return (u, z, p, verdict)."""
    b, c = np.asarray(baseline, dtype=float), np.asarray(canary, dtype=float)
    pooled = np.concatenate([b, c])
    ranks = np.argsort(np.argsort(pooled)) + 1.0           # ties get different ranks
    u = ranks[len(b):].sum() - len(c) * (len(c) + 1) / 2
    mu, sd = len(b) * len(c) / 2, np.sqrt(len(b) * len(c) * (len(pooled) + 1) / 12)
    z = (u - mu) / sd
    p = float(np.exp(-z * z / 2))
    return float(u), float(z), p, 'fail' if p < alpha else 'pass'
'''

SOLUTION = '''import math
import numpy as np

def average_ranks(x):
    """1-based ranks; tied values share the mean of the ranks they occupy."""
    order = np.argsort(x, kind='stable')
    xs = x[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])     # first index of each tie group
    sizes = np.diff(np.r_[starts, len(x)])
    avg = starts + (sizes + 1) / 2.0                             # mean of ranks starts+1 .. starts+size
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(avg, sizes)
    return ranks, sizes

def canary_judge(baseline, canary, alpha=0.01, min_samples=10):
    b, c = np.asarray(baseline, dtype=float), np.asarray(canary, dtype=float)
    nb, nc = len(b), len(c)
    if nb == 0 or nc == 0:
        return 0.0, 0.0, 1.0, 'inconclusive'
    N = nb + nc
    ranks, sizes = average_ranks(np.concatenate([b, c]))
    u = ranks[nb:].sum() - nc * (nc + 1) / 2.0
    mu = nb * nc / 2.0
    ties = float(np.sum(sizes ** 3 - sizes))
    var = nb * nc / 12.0 * ((N + 1) - ties / (N * (N - 1))) if N > 1 else 0.0
    if var <= 0:
        z = 0.0
    else:
        d = u - mu
        z = (d - 0.5 * np.sign(d)) / math.sqrt(var)              # continuity correction towards mu
    p = min(1.0, math.erfc(abs(z) / math.sqrt(2)))
    if nb < min_samples or nc < min_samples:
        verdict = 'inconclusive'
    elif p < alpha and u > mu:
        verdict = 'fail'
    else:
        verdict = 'pass'
    return float(u), float(z), float(p), verdict
'''

HINTS = [
    'For average ranks, sort once (`np.argsort(..., kind="stable")`), find where each run of equal values starts with `xs[1:] != xs[:-1]`, and give every member of a run of size $t$ starting at sorted position $s$ the rank $s + (t+1)/2$. The run sizes are also the $t$ values for the tie correction.',
    "$U$ is the canary's rank sum minus $n_c(n_c+1)/2$. Since ranks are multiples of 0.5, $U - \\mu$ is a multiple of 0.5 too, so the continuity correction never flips the sign of $z$.",
    'Use `math.erfc(abs(z) / math.sqrt(2))` for the two-sided p-value. Check the empty case first, compute $u, z, p$ normally even when a sample is below `min_samples`, and only then pick the verdict.',
]

_rng = np.random.default_rng(31)
_base = np.round(_rng.lognormal(3.0, 0.4, 300))
_same = np.round(_rng.lognormal(3.0, 0.4, 120))
_slow = np.round(_rng.lognormal(3.12, 0.4, 120))
_fast = np.round(_rng.lognormal(2.85, 0.4, 120))

TESTS = [
    (np.array([10.0, 11.0, 12.0, 13.0]), np.array([14.0, 15.0, 16.0]), 0.05, 3),
    (np.array([20.0, 21.0, 22.0, 20.0, 21.0, 22.0]), np.array([20.0, 22.0, 21.0, 22.0]), 0.05, 3),
    (np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0]), 0.05, 1),
    (np.array([1.0, 2.0, 3.0]), np.array([]), 0.05, 1),
    (np.arange(1.0, 31.0), np.arange(11.0, 41.0), 0.01, 10),
    (np.arange(11.0, 41.0), np.arange(1.0, 31.0), 0.01, 10),
    (np.arange(1.0, 31.0), np.arange(11.0, 41.0), 0.01, 50),
    (_base, _same),
    (_base, _slow),
    (_base, _fast),
]

EXPLANATION = r"""
## Why it works
$U$ counts, over all $n_b n_c$ cross pairs, how often the canary value beats the baseline value (ties count one half), so $U / (n_b n_c)$ estimates $P(\text{canary} > \text{baseline})$. Under $H_0$ that's $1/2$, and $U$ is approximately normal with mean $\mu$ and variance $\sigma^2$. Ranks don't care about the size of an outlier, so one 30-second timeout can't dominate the test the way it dominates a mean.

Ties shrink the variance: tied values can't be ordered, so there is less randomness in the rank sum than the no-ties formula assumes. The $\sum(t^3 - t)$ term removes exactly that amount; with integer millisecond latencies it matters (test 2). The verdict is one-sided by design: a canary that is significantly *faster* (test 10, also test 6) should not be rolled back.

## Complexity
$O(N \log N)$ for the sort; everything else is linear.

## Common bugs
- Ordinal ranks via `argsort(argsort(x))`: tied values get different ranks depending on input order, so $U$ changes when you shuffle the data.
- Forgetting the tie correction, or applying the continuity correction in the wrong direction (away from $\mu$).
- Failing a canary that is significantly better, because the verdict only looked at $p$.
- Running the judge on tiny samples right after rollout. `min_samples` (test 7) exists because the normal approximation and the business decision both need data.

## Interview follow-ups
- The judge compares p95 latency, not the whole distribution. How would you test a difference in a high quantile? (Bootstrap the quantile difference, or a quantile-specific test; rank tests answer a different question.)
- Your canary gets 1% of traffic but you evaluate 40 metrics every 5 minutes. What goes wrong and how do you fix it? (Repeated, multiple testing inflates false rollbacks: use sequential tests and correct across metrics, or require several consecutive failures.)
"""
