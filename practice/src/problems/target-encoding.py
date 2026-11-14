import numpy as np

META = dict(title='Out-of-fold target encoding with smoothing', tags=['Feature Engineering', 'Data Processing'], diff='Medium',
            runtime='numpy', related=['categorical-encoding', 'k-fold-cross-validation', 'frequency-encoding'])

STATEMENT = r"""
Target encoding replaces a category with the average target for that category. Done naively, each row's own label leaks into its feature. Implement the out-of-fold, smoothed version.

Write `target_encode(cats, y, folds, alpha)`:
- `cats`: array of category labels (strings), length $n$.
- `y`: array of 0/1 targets.
- `folds`: integer array; row $i$ belongs to fold `folds[i]`.
- `alpha`: smoothing strength (a float $\ge 0$).

For each row $i$ in fold $f$, use only rows **not** in fold $f$. With $m$ the mean of `y` over those rows, $n_c$ the number of those rows with category $c$ and $s_c$ the sum of their targets, the encoding is

$$\text{enc}_i = \frac{s_c + \alpha m}{n_c + \alpha}, \quad c = \text{cats}[i].$$

A category unseen outside the fold gets $m$ (this is the formula with $n_c = s_c = 0$, as long as $\alpha > 0$; if $\alpha = 0$ and $n_c = 0$, also return $m$). Return a float array of length $n$.
"""

STARTER = '''import numpy as np

def target_encode(cats, y, folds, alpha):
    """Out-of-fold smoothed target encoding. Returns a float array, one value per row."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def target_encode(cats, y, folds, alpha):
    cats, y, folds = np.asarray(cats), np.asarray(y, dtype=float), np.asarray(folds)
    out = np.empty(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = y[tr].mean()
        for i in np.where(te)[0]:
            sel = tr & (cats == cats[i])
            n, s = sel.sum(), y[sel].sum()
            out[i] = m if n + alpha == 0 else (s + alpha * m) / (n + alpha)
    return out
'''

HINTS = [
    'Loop over folds. For rows in fold f, compute the statistics from every other fold only.',
    'The smoothed value pulls a category with few rows towards the global mean m: alpha acts like alpha extra rows with target m.',
    'Watch the case alpha = 0 with a category that never appears outside the fold: avoid dividing by zero and return m.',
]

TESTS = [
    (np.array(['a', 'a', 'b', 'b']), np.array([1, 0, 1, 1]), np.array([0, 1, 0, 1]), 0.0),
    (np.array(['a', 'a', 'b', 'b']), np.array([1, 0, 1, 1]), np.array([0, 1, 0, 1]), 2.0),
    (np.array(['x', 'y', 'z', 'x', 'y', 'x']), np.array([1, 0, 0, 1, 1, 0]), np.array([0, 0, 1, 1, 2, 2]), 1.0),
    (np.array(['u', 'v', 'w']), np.array([1, 0, 1]), np.array([0, 1, 2]), 0.0),
    (np.array(['p'] * 5 + ['q'] * 5), np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1]), np.array([0, 1, 2, 3, 4] * 2), 3.0),
    (np.array(['c', 'c', 'c', 'd']), np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 0.5),
]

EXPLANATION = r"""
## Why out of fold
If a row's own label is included, a category seen once gets an encoding equal to its label, and the model learns to read the answer off the feature. Validation looks excellent and production does not. Computing each fold's encoding from the other folds mimics what will happen for genuinely new data.

## Why smoothing
A category seen twice with targets 1 and 1 has mean 1.0, which is not much evidence. The smoothed estimate is a weighted average of the category mean and the global mean, with weights $n_c$ and $\alpha$. It is the posterior mean under a Beta prior centred on $m$.

## At serving time
Fit the encoding on all training data (no folds) and apply it to new rows; unseen categories get the global mean.

## Complexity
This direct version is $O(n^2)$ in the worst case; a production version computes per-fold group sums with a single group-by.

## Interview follow-ups
- How does target encoding interact with tree models versus linear models?
- How would you extend it to a regression target or to multi-class targets?
"""
