import numpy as np

META = dict(title='Estimating pi three ways with Monte Carlo', tags=['Probability', 'Probability and Statistics'], diff='Hard',
            runtime='numpy', lessons=['ieng:00.04', 'mlmaths:12.01'], related=['standard-error-comparison', 'bootstrap-confidence-interval'])

STATEMENT = r"""
The quarter circle $x^2 + y^2 \le 1$ in the unit square has area $\pi/4$. Three Monte Carlo estimators of $\pi$, all with the same budget $n$, show how much the **design** of an estimator matters.

Draw all randomness with exactly these calls, in this order:

```
rng = np.random.default_rng(seed)
pts = rng.random((n, 2))   # hit-or-miss
u = rng.random(n)          # mean value
w = rng.random(n // 2)     # antithetic pairs
```

1. **Hit-or-miss.** $h_i = 4$ if $x_i^2 + y_i^2 \le 1$, else $0$, using the rows of `pts`.
2. **Mean value.** $\pi = 4\int_0^1 \sqrt{1 - u^2}\,du$, so use $g_i = 4\sqrt{1 - u_i^2}$.
3. **Antithetic.** Pair each $w_j$ with $1 - w_j$: $a_j = 2\big(\sqrt{1 - w_j^2} + \sqrt{1 - (1 - w_j)^2}\big)$, the average of the two mean-value terms.

For each, the estimate is the sample mean of its terms, and its standard error is the sample standard deviation of the terms (`ddof=1`) divided by the square root of **the number of terms** ($n$, $n$ and $n/2$).

Write `mc_pi(n, seed)` that returns `((est_hit, se_hit), (est_mean, se_mean), (est_anti, se_anti))`, a tuple of three `(estimate, standard error)` pairs of Python floats.

## Requirements
- `n` is an even int $\ge 4$. `seed` is an int.
- Vectorised: no Python loops over samples.
"""

STARTER = '''import numpy as np

def mc_pi(n, seed):
    """Return ((hit, se), (mean_value, se), (antithetic, se))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _summary(terms):
    return float(terms.mean()), float(terms.std(ddof=1) / np.sqrt(len(terms)))

def mc_pi(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    u = rng.random(n)
    w = rng.random(n // 2)
    f = lambda v: np.sqrt(1 - v ** 2)              # quarter-circle height at v
    hit = 4.0 * ((pts ** 2).sum(axis=1) <= 1)
    mean_value = 4.0 * f(u)
    anti = 2.0 * (f(w) + f(1 - w))                 # negatively correlated pair: one high, one low
    return _summary(hit), _summary(mean_value), _summary(anti)
'''

HINTS = [
    'Create the generator from `seed` inside the function and make the three `rng.random` calls in the order shown. Any extra or reordered call changes every number after it.',
    'Each estimator is "build a vector of terms, then report its mean and its standard error". Write that summary once as a helper.',
    'The antithetic estimator has only $n/2$ terms (each costs two function evaluations), so its standard error divides by $\\sqrt{n/2}$.',
]

TESTS = [
    (4, 0),
    (10, 1),
    (1000, 42),
    (20000, 7),
    (200000, 123),
    (6, 99),
]

EXPLANATION = r"""
## Same answer, different variance
All three are unbiased for $\pi$, so they differ only in variance per unit of work:

- **Hit-or-miss** terms are 0 or 4, a Bernoulli scaled by 4: the variance is $16 \cdot \frac\pi4(1 - \frac\pi4) \approx 2.70$.
- **Mean value** replaces the random 0/4 with the exact height of the curve at $u$, which removes the randomness in $y$: the variance drops to about $0.80$. This is Rao-Blackwellisation: conditioning on part of the randomness never increases variance.
- **Antithetic** pairs use the fact that $f$ is decreasing, so $f(w)$ and $f(1-w)$ are negatively correlated: when one is high, the other is low. Averaging them cancels much of the noise: the variance per pair is about $0.11$, against $0.80/2 = 0.40$ for the average of two independent mean-value terms.

Since the standard error is $\sigma/\sqrt{\text{terms}}$, reaching a fixed precision with hit-or-miss needs about 12 times larger $n$ than antithetic pairs here, and the mean-value estimator about 3.5 times more. Compare the standard errors in the large tests.

## Why fix the random calls
Monte Carlo code is only testable when it's reproducible. Seeding a local `default_rng` and fixing the draw order makes the output a deterministic function of `(n, seed)`, and keeps it independent of other code that uses randomness.

## Complexity
$O(n)$ time and memory.

## Common bugs
- Using `np.random.seed` and the global functions, which share state with everything else.
- Dividing the antithetic standard error by $\sqrt n$ instead of $\sqrt{n/2}$, which overstates its advantage.
- Treating $f(w)$ and $f(1-w)$ as $n$ independent terms. They aren't independent, so their naive standard error is wrong.

## Interview follow-ups
- What's the error rate of quasi-Monte Carlo (Sobol points) compared with $O(1/\sqrt n)$? (Close to $O(\log^d n / n)$ for smooth integrands.)
- Antithetic sampling helped here because $f$ is monotone. Give an integrand where it makes things worse. (A symmetric one such as $f(u) = (u - \tfrac12)^2$: then $f(w) = f(1-w)$ and each pair is a duplicate.)
"""
