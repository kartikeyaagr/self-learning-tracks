import numpy as np

META = dict(title='Log transform for skewed features', tags=['Feature Engineering'], diff='Easy', runtime='numpy',
            related=['skewness-and-kurtosis', 'feature-scaling', 'winsorize'])

STATEMENT = r"""
Heavy-tailed, non-negative features (prices, counts, durations) are often easier for models after a log transform. Write `log_transform(X, threshold)` for a 2-D float array `X` of shape `(n, d)`:

1. For each column compute the sample skewness $g = \frac{\frac1n\sum (x - \bar x)^3}{\left(\frac1n \sum (x-\bar x)^2\right)^{3/2}}$ (use 0 when the column is constant).
2. A column is transformed if $g >$ `threshold` **and** its minimum is $\ge 0$. Transformed columns become `np.log1p(x)`; others are unchanged.

Return a tuple `(X_new, transformed)` where `transformed` is a list of the transformed column indices in increasing order.
"""

STARTER = '''import numpy as np

def log_transform(X, threshold):
    """Return (X_new, transformed_column_indices)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def log_transform(X, threshold):
    X = np.asarray(X, dtype=float)
    out, cols = X.copy(), []
    for j in range(X.shape[1]):
        x = X[:, j]
        c = x - x.mean()
        m2 = (c ** 2).mean()
        g = 0.0 if m2 == 0 else (c ** 3).mean() / m2 ** 1.5
        if g > threshold and x.min() >= 0:
            out[:, j] = np.log1p(x)
            cols.append(j)
    return out, cols
'''

HINTS = [
    'Skewness is the third central moment divided by the variance to the power 3/2. Compute it per column.',
    'Use `np.log1p`, which is log(1 + x): it maps 0 to 0 and is accurate for small x.',
    'Only transform columns with no negative values; log1p of values below -1 is undefined.',
]

_r = np.random.default_rng(3)
TESTS = [
    (np.array([[1.0, 5.0], [2.0, 5.0], [100.0, 5.0]]), 0.5),
    (np.column_stack([_r.lognormal(0, 1, 30), _r.normal(0, 1, 30)]), 1.0),
    (np.column_stack([_r.exponential(2.0, 25), -_r.exponential(2.0, 25), _r.uniform(0, 1, 25)]), 0.75),
    (np.array([[0.0], [0.0], [0.0], [9.0]]), 0.0),
    (np.column_stack([_r.poisson(1.0, 40).astype(float), _r.poisson(30.0, 40).astype(float)]), 0.6),
]

EXPLANATION = r"""
## Why log helps
A log turns multiplicative differences into additive ones: going from 10 to 100 and from 100 to 1,000 become equal steps. Linear models and distance-based methods then see a more even spread, and a few huge values no longer dominate squared-error losses. Tree models are invariant to monotonic transforms, so they gain little.

## Why log1p
Counts often include zeros, and $\log 0$ is undefined. $\log(1 + x)$ handles zeros and behaves like $x$ for small values. For data with negative values, alternatives are the Yeo-Johnson transform or a signed log, $\mathrm{sign}(x)\log(1 + |x|)$.

## Fit on training data only
Which columns to transform is a decision learned from data; make it on the training split and apply the same list to validation and test.

## Interview follow-ups
- How do you report predictions in original units if the target itself was log-transformed? (Exponentiating the mean of a log prediction gives a median, not a mean.)
"""
