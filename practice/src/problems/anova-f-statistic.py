import numpy as np

META = dict(title='One-way ANOVA F statistic', tags=['Probability and Statistics'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:12.01'], related=['t-test-one-sample', 'stratified-sampling'])

STATEMENT = r"""
One-way ANOVA asks whether $k$ groups share the same mean (for example, latency under $k$ different batch sizes). It splits the total variation around the grand mean $\bar y$ into a part **between** groups and a part **within** them:

$$SS_B = \sum_{g} n_g\,(\bar y_g - \bar y)^2, \qquad SS_W = \sum_{g} \sum_{i \in g} (y_i - \bar y_g)^2$$

With $N$ observations, the F statistic compares the two mean squares:

$$F = \frac{SS_B / (k - 1)}{SS_W / (N - k)}$$

and $\eta^2 = SS_B / (SS_B + SS_W)$ is the share of variance explained by the grouping.

Write `anova(values, groups)` that returns the tuple `(F, df_between, df_within, eta_sq)`, with $F$ and $\eta^2$ as floats and the degrees of freedom $k - 1$ and $N - k$ as ints.

## Requirements
- `values`: 1-D float array. `groups`: int array of the same length with **arbitrary** labels (not necessarily `0..k-1`, not necessarily sorted).
- There are at least 2 groups, $N > k$, and $SS_W > 0$.
- No Python loops over groups.
"""

STARTER = '''import numpy as np

def anova(values, groups):
    """Return (F, df_between, df_within, eta_squared)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def anova(values, groups):
    y = np.asarray(values, dtype=float)
    _, g = np.unique(groups, return_inverse=True)          # relabel groups as 0..k-1
    k, N = g.max() + 1, len(y)
    n = np.bincount(g)
    means = np.bincount(g, weights=y) / n
    ss_b = float(n @ (means - y.mean()) ** 2)
    ss_w = float(((y - means[g]) ** 2).sum())
    df_b, df_w = int(k - 1), int(N - k)
    return (ss_b / df_b) / (ss_w / df_w), df_b, df_w, ss_b / (ss_b + ss_w)
'''

HINTS = [
    '`np.unique(groups, return_inverse=True)` maps any labels to `0..k-1`, which `np.bincount` needs.',
    'Group sizes and sums come from `np.bincount(g)` and `np.bincount(g, weights=y)`. `means[g]` gives each observation its own group mean.',
    'Weight each group\'s squared deviation from the grand mean by its size $n_g$. The degrees of freedom are $k - 1$ and $N - k$.',
]

_r = np.random.default_rng(88)
_g = _r.integers(0, 4, size=80) * 10 + 7
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), np.array([0, 0, 1, 1, 2, 2])),
    (np.array([5.0, 6.0, 5.5, 5.0, 6.0, 5.5]), np.array([3, 3, 3, 9, 9, 9])),
    (np.array([10.0, 12.0, 11.0, 30.0, 31.0]), np.array([-1, -1, -1, 5, 5])),
    (np.array([2.0, 9.0, 4.0, 7.0, 3.0, 8.0, 5.0]), np.array([2, 1, 2, 1, 2, 1, 0])),
    (_r.normal(0.0, 1.0, size=80) + _g / 50, _g),
    (_r.normal(1e4, 1.0, size=30), np.repeat([1, 2, 3], 10)),
]

EXPLANATION = r"""
## Why it works
If all groups share one mean, both mean squares estimate the same noise variance $\sigma^2$: the within-group one directly, and the between-group one through how much group means scatter (each has variance $\sigma^2/n_g$). So $F \approx 1$ under $H_0$. Real differences inflate only the numerator. Under normality and equal variances, $F$ follows an $F_{k-1,\,N-k}$ distribution, which gives the p-value.

$SS_T = SS_B + SS_W$ always holds, since the cross term vanishes because deviations from each group mean sum to zero.

## Relation to the t-test
With $k = 2$, $F = t^2$ for the pooled two-sample t-test. ANOVA is also a linear regression on group indicators: $F$ is the regression's overall F test and $\eta^2$ is its $R^2$.

## Complexity
$O(N \log N)$ for `np.unique`, $O(N)$ otherwise.

## Common bugs
- Forgetting the $n_g$ weight in $SS_B$, which matters with unequal group sizes.
- Using $k$ and $N$ as the degrees of freedom instead of $k-1$ and $N-k$.
- `bincount` on raw labels like `[-1, 5]` (negative labels raise; large labels waste memory).

## Interview follow-ups
- ANOVA says "not all means are equal". How do you find which ones differ? (Pairwise tests with a correction such as Tukey's HSD or Bonferroni.)
- What if the groups have very different variances? (Welch's ANOVA, or the Kruskal-Wallis test on ranks.)
"""
