import numpy as np

META = dict(title='Equal-width and quantile binning', tags=['Feature Engineering'], diff='Easy', runtime='numpy',
            related=['percentiles-quantiles', 'population-stability-index', 'winsorize'])

STATEMENT = r"""
Write `bin_feature(train, test, k, strategy)` that learns bin edges on `train` and assigns bin indices $0, \dots, k-1$ to both arrays.

- `strategy="width"`: interior edges at equal steps between `train.min()` and `train.max()`: `np.linspace(min, max, k + 1)[1:-1]`.
- `strategy="quantile"`: interior edges at `np.quantile(train, [i / k for i in 1..k-1])` (default linear interpolation), then remove duplicate edges with `np.unique`.
- A value's bin is the number of interior edges that are $\le$ the value: `np.searchsorted(edges, x, side="right")`. Values outside the training range fall in the first or last bin automatically.

Return `(train_bins, test_bins, edges)` with integer arrays and the float edge array.
"""

STARTER = '''import numpy as np

def bin_feature(train, test, k, strategy):
    """Return (train_bins, test_bins, interior_edges)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def bin_feature(train, test, k, strategy):
    train, test = np.asarray(train, dtype=float), np.asarray(test, dtype=float)
    if strategy == "width":
        edges = np.linspace(train.min(), train.max(), k + 1)[1:-1]
    else:
        edges = np.unique(np.quantile(train, [i / k for i in range(1, k)]))
    f = lambda x: np.searchsorted(edges, x, side="right")
    return f(train), f(test), edges
'''

HINTS = [
    'Only the k - 1 interior edges matter; everything below the first edge is bin 0 and everything above the last is the top bin.',
    '`np.searchsorted(edges, x, side="right")` counts how many edges are <= x, which is exactly the bin index.',
    'With many repeated values, quantile edges can coincide; `np.unique` removes the duplicates (so there may be fewer than k bins).',
]

_r = np.random.default_rng(8)
TESTS = [
    (np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([-1.0, 2.5, 10.0]), 2, "width"),
    (np.array([1.0, 2.0, 3.0, 4.0, 100.0]), np.array([50.0]), 4, "quantile"),
    (np.array([1.0, 2.0, 3.0, 4.0, 100.0]), np.array([50.0]), 4, "width"),
    (np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0]), np.array([0.0, 1.5]), 3, "quantile"),
    (_r.exponential(1.0, 50), _r.exponential(1.0, 10), 5, "quantile"),
    (_r.normal(size=40), _r.normal(size=8) * 3, 6, "width"),
]

EXPLANATION = r"""
## Width versus quantile
Equal-width bins are easy to interpret but a skewed feature puts almost everything in the first bin (see the test with 100). Quantile bins put roughly the same number of training rows in each bin, which is what population stability checks and many tree-building libraries use.

## Why learn edges on train
Edges are model parameters. Re-computing them on test data would shift the meaning of each bin and hide drift.

## Ties
When a value equals an edge, `side="right"` puts it in the upper bin. Pick a rule and keep it the same in training and serving.

## Interview follow-ups
- When does binning help a linear model? (It lets it fit non-linear effects as a step function.)
- What is lost by binning for a tree model?
"""
