import numpy as np

META = dict(title='Two-sample Kolmogorov-Smirnov drift test', tags=['MLOps', 'Probability and Statistics'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:11.12'], related=['population-stability-index', 'percentiles-quantiles'])

STATEMENT = r"""
PSI needs bins. The two-sample **Kolmogorov-Smirnov** test doesn't: it compares the empirical CDFs of the reference and current samples directly,

$$D = \max_x \bigl|F_{\mathrm{ref}}(x) - F_{\mathrm{cur}}(x)\bigr|, \qquad F(x) = \frac{\#\{\text{values} \le x\}}{\text{sample size}}$$

Both ECDFs are step functions that only jump at observed values, so the maximum is attained at one of the pooled sample values.

Write `ks_drift(reference, current, alpha=0.05)` that returns the tuple `(D, location, p_value, drifted)`.

## Definitions
- `reference` has $n$ values and `current` has $m$ values (1-D float arrays, $n, m \ge 1$). Ties within and across samples are allowed.
- `D` (float): the statistic above, evaluated at every pooled value with the right-continuous ECDF ($\le x$).
- `location` (float): the **smallest** pooled value at which $D$ is attained. To find it reliably, compare the exact integer gaps $|c_{\mathrm{ref}}(x)\cdot m - c_{\mathrm{cur}}(x)\cdot n|$, where $c(x)$ counts values $\le x$, and set $D = \text{max gap} / (n m)$.
- `p_value` (float): the asymptotic Kolmogorov approximation. With $n_e = \frac{nm}{n+m}$ and $\lambda = \bigl(\sqrt{n_e} + 0.12 + 0.11/\sqrt{n_e}\bigr)\,D$,

$$p = 2\sum_{j=1}^{100} (-1)^{j-1} e^{-2 j^2 \lambda^2}$$

  clipped to $[0, 1]$. If $\lambda < 0.3$ the series converges too slowly; use $p = 1.0$ instead.
- `drifted` (bool): `p_value < alpha`.
"""

STARTER = '''import numpy as np

def ks_drift(reference, current, alpha=0.05):
    """Return (D, location, p_value, drifted)."""
    a, b = np.sort(reference), np.sort(current)
    grid = np.concatenate([a, b])
    Fa = np.searchsorted(a, grid, side='left') / len(a)     # is this the ECDF at x?
    Fb = np.searchsorted(b, grid, side='left') / len(b)
    gaps = np.abs(Fa - Fb)
    i = int(np.argmax(gaps))
    return float(gaps[i]), float(grid[i]), 1.0, False
'''

SOLUTION = '''import numpy as np

def ks_drift(reference, current, alpha=0.05):
    a = np.sort(np.asarray(reference, dtype=float))
    b = np.sort(np.asarray(current, dtype=float))
    n, m = len(a), len(b)
    grid = np.unique(np.concatenate([a, b]))                 # sorted, so argmax finds the smallest x
    ca = np.searchsorted(a, grid, side='right')              # counts <= x: right-continuous ECDF
    cb = np.searchsorted(b, grid, side='right')
    gaps = np.abs(ca * m - cb * n)                           # exact integers: ties are real ties
    i = int(np.argmax(gaps))
    D = gaps[i] / (n * m)

    ne = n * m / (n + m)
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * D
    if lam < 0.3:
        p = 1.0
    else:
        j = np.arange(1, 101)
        p = 2 * np.sum((-1.0) ** (j - 1) * np.exp(-2 * j ** 2 * lam ** 2))
        p = float(np.clip(p, 0.0, 1.0))
    return float(D), float(grid[i]), p, bool(p < alpha)
'''

HINTS = [
    'The ECDF at $x$ counts values **less than or equal to** $x$: on a sorted array that is `np.searchsorted(sorted_x, x, side="right")`.',
    'Evaluate both counts on `np.unique` of the pooled values. Because that grid is sorted, `np.argmax` of the gaps returns the smallest location. Compute the gaps as integers, `|ca * m - cb * n|`, so equal gaps compare equal.',
    'For the p-value, build $\\lambda$ from the effective size $n_e = nm/(n+m)$, sum 100 alternating terms with `j = np.arange(1, 101)`, clip to [0, 1], and short-circuit to 1.0 when $\\lambda < 0.3$.',
]

_rng = np.random.default_rng(3)
_ref = _rng.normal(0.0, 1.0, 400)

TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0, 6.0])),
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
    (np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    (np.array([5.0]), np.array([1.0, 2.0, 3.0, 4.0, 6.0, 7.0])),
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), 0.1),
    (np.arange(50, dtype=float), np.arange(50, dtype=float) + 30.0),
    (_ref, _rng.normal(0.0, 1.0, 300)),
    (_ref, _rng.normal(0.25, 1.0, 300)),
    (_ref, _rng.normal(0.0, 1.6, 300), 0.01),
]

EXPLANATION = r"""
## Why it works
Between two consecutive pooled values neither ECDF changes, so checking only the pooled values is exhaustive. Counting with `side='right'` gives the right-continuous ECDF; with ties across samples (test 3) the `side='left'` version measures the gap *just before* the jump and gets $D$ wrong.

Multiplying through by $nm$ turns $|c_a/n - c_b/m|$ into the integer $|c_a m - c_b n|$. Mathematically equal gaps such as $2/3 - 1/3$ and $1/3 - 0$ are not always equal in floating point, so the integer form is what makes "the smallest location" well defined.

Under $H_0$ (same distribution) $\sqrt{n_e}\,D$ converges to the Kolmogorov distribution, whose tail is the alternating series. The $0.12 + 0.11/\sqrt{n_e}$ term is a small-sample correction.

## Complexity
$O((n + m)\log(n + m))$ for the sorts and binary searches. KS is scale-free: it reacts to shifts in location (test 8) and in spread (test 9) without any binning choice.

## Common bugs
- `side='left'`, which evaluates the ECDF just below each value.
- Not using the pooled grid (only one sample's values), which can miss the maximum.
- Computing the series for tiny $\lambda$: 100 alternating terms of nearly 1 sum to about 0, giving $p \approx 0$ for identical samples.
- Running KS on millions of rows and treating every $p < 0.05$ as a real problem. At that size even negligible drift is significant; pair it with an effect size such as $D$ itself or PSI.

## Interview follow-ups
- You run KS on 300 features every hour. How do you control false alarms? (Bonferroni or Benjamini-Hochberg across features, plus a minimum $D$.)
- KS is least sensitive in the tails. Which test weights the tails more? (Anderson-Darling.)
"""
