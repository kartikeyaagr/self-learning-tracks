import numpy as np

META = dict(title='Multiple testing: Bonferroni and Benjamini-Hochberg', tags=['Probability and Statistics'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:12.17'], related=['t-test-one-sample', 'b-test-significance'])

STATEMENT = r"""
Run 20 tests at $\alpha = 0.05$ when nothing is going on, and you expect one "significant" result by chance. Two standard corrections fix this for $m$ p-values $p_1, \dots, p_m$.

**Bonferroni** controls the chance of even one false positive. The adjusted p-value is $\tilde p_i = \min(1,\; m\,p_i)$.

**Benjamini-Hochberg (BH)** controls the expected fraction of false discoveries. Sort the p-values ascending as $p_{(1)} \le \dots \le p_{(m)}$. The adjusted value at rank $r$ is

$$q_{(r)} = \min\Big(1,\; \min_{j \ge r} \frac{m\,p_{(j)}}{j}\Big)$$

Then put each $q$ back in its original position.

Write `adjust(pvals, alpha=0.05)` that returns the tuple `(bonf, bonf_reject, bh, bh_reject)`:

- `bonf`, `bh`: float arrays of adjusted p-values, in the **original** order,
- `bonf_reject`, `bh_reject`: bool arrays, `True` where the adjusted p-value is `<= alpha`.

## Requirements
- `pvals`: 1-D float array of values in $[0, 1]$, possibly empty, possibly with ties. Every output has the same shape.
- Vectorised. `np.minimum.accumulate` on a reversed array gives the running minimum from the right.
"""

STARTER = '''import numpy as np

def adjust(pvals, alpha=0.05):
    """Return (bonferroni_adj, bonferroni_reject, bh_adj, bh_reject)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def adjust(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    bonf = np.minimum(1.0, m * p)
    order = np.argsort(p, kind='stable')
    scaled = m * p[order] / np.arange(1, m + 1)                  # m p_(j) / j
    q_sorted = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])   # running min from the right
    bh = np.empty(m)
    bh[order] = q_sorted                                        # scatter back to the original order
    return bonf, bonf <= alpha, bh, bh <= alpha
'''

HINTS = [
    'Bonferroni is one line. For BH, get the sort order with `np.argsort`, and work on the sorted p-values.',
    'Compute $m\\,p_{(j)}/j$ for every rank, then take the minimum over all later ranks: reverse, `np.minimum.accumulate`, reverse back. Cap at 1.',
    'To undo the sort, assign into an empty array with `bh[order] = q_sorted`. Tied p-values automatically end up with the same adjusted value.',
]

_r = np.random.default_rng(55)
TESTS = [
    (np.array([0.01, 0.04, 0.03, 0.20]),),
    (np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]),),
    (np.array([]),),
    (np.array([0.03]),),
    (np.array([0.02, 0.02, 0.02, 0.5]), 0.05),
    (np.array([0.9, 0.0, 1.0, 0.5]), 0.1),
    (np.concatenate([_r.uniform(0, 0.002, 5), _r.uniform(0, 1, 45)]),),
]

EXPLANATION = r"""
## Two different promises
- **Bonferroni** bounds the family-wise error rate, $P(\text{any false positive}) \le \alpha$, using only the union bound. It needs no independence, but with many tests it becomes very conservative: 10,000 features means each needs $p < 5 \times 10^{-6}$.
- **BH** bounds the false discovery rate, $E[\text{false rejections} / \text{rejections}] \le \alpha$, for independent (or positively dependent) tests. It allows a few mistakes among many discoveries, which is the right trade for screening.

In test 2, Bonferroni rejects 1 of 10 at 0.05, while BH rejects 2.

## Why the running minimum
The raw ratios $m p_{(j)}/j$ aren't monotone in $j$. BH's rule "reject everything up to the largest rank $k$ with $p_{(k)} \le k\alpha/m$" means a p-value inherits the verdict of any larger one that passes. Taking the minimum from the right encodes that, and makes the adjusted values monotone in $p$.

## Complexity
$O(m \log m)$ for the sort, $O(m)$ after it.

## Common bugs
- Returning the adjusted values in sorted order instead of the original order.
- Running the minimum from the left (`np.minimum.accumulate(scaled)`), which is wrong.
- Forgetting the cap at 1.

## Interview follow-ups
- You look at an A/B dashboard daily and stop the day it hits $p < 0.05$. What's wrong, and how do you fix it? (Optional stopping inflates false positives. Use a fixed horizon, alpha spending, or sequential tests.)
- With 50 metrics on one experiment, which correction would you use, and why? (BH for exploratory metrics, plus one pre-registered primary metric tested at full $\alpha$.)
"""
