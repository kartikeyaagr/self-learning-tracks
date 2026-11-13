import numpy as np

META = dict(title='Standardise features with broadcasting', tags=['Data Processing', 'Neural Networks'], diff='Easy', runtime='torch',
            lessons=['air-fanout-v1:02.03', 'air-fanout-v1:03.08'],
            related=['tensor-factory', 'per-channel-mean-and-std', 'batch-normalization'])

STATEMENT = r"""
A feature matrix has to be standardised before it reaches a model: centre each column, divide by its spread, and clip outliers so one bad sensor reading can't dominate.

Write `standardize(x, clip)`:

- `x`: float64 tensor `(N, D)`, one row per sample. $N \ge 1$.
- `clip`: positive float.

Return a tuple `(mean, std, z, row_norm)`:

1. `mean`: `(D,)`, the column means.
2. `std`: `(D,)`, the **population** standard deviation of each column (divide by $N$, not $N-1$).
3. `z`: `(N, D)`, $z_{ij} = (x_{ij} - \text{mean}_j) / \text{std}_j$, then clamped to `[-clip, clip]`. A column whose `std` is exactly 0 gets `z = 0` everywhere (not `nan`).
4. `row_norm`: `(N,)`, the L2 norm of each row of the clipped `z`.

Use tensor operations only, no Python loops.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def standardize(x, clip):
    """x: (N, D) float64. Return (mean (D,), std (D,), z (N, D), row_norm (N,))."""
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    z = (x - mean) / std
    return mean, std, z, z.norm(dim=1)
'''

SOLUTION = '''import torch

def standardize(x, clip):
    mean = x.mean(dim=0)                                    # (D,) reduce over the sample axis
    std = x.std(dim=0, correction=0)                        # population std; torch's default is N-1
    safe = torch.where(std > 0, std, torch.ones_like(std))  # avoid 0/0 in constant columns
    z = torch.where(std > 0, (x - mean) / safe, torch.zeros_like(x))   # (N, D) - (D,) broadcasts per column
    z = z.clamp(-clip, clip)
    return mean, std, z, torch.linalg.vector_norm(z, dim=1)
'''

HINTS = [
    'Reduce over `dim=0` to get one value per column. A `(D,)` tensor broadcasts against `(N, D)` from the right, so `x - mean` just works.',
    '`Tensor.std` divides by $N-1$ by default. Pass `correction=0` (or `unbiased=False` on older versions).',
    'Handle constant columns with `torch.where(std > 0, ..., 0)`, but divide by a "safe" std so no `nan` is ever produced. Then `clamp`, then `torch.linalg.vector_norm(z, dim=1)`.',
]

_r = np.random.default_rng(3)
TESTS = [
    (np.array([[1.0, 10.0], [3.0, 30.0]]), 3.0),
    (np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]]), 1.0),
    (np.array([[7.0, -1.0, 2.0]]), 2.0),
    (np.array([[0.0], [0.0], [0.0], [100.0]]), 1.5),
    (_r.normal(size=(6, 4)) * np.array([1.0, 10.0, 0.1, 3.0]) + 5.0, 2.0),
    (np.vstack([_r.normal(size=(9, 3)), [[50.0, -50.0, 0.0]]]), 2.5),
    (_r.uniform(-1, 1, size=(20, 5)), 10.0),
]

EXPLANATION = r"""
## Why it works
Reducing over `dim=0` collapses the sample axis and leaves one statistic per feature. Broadcasting aligns shapes from the right, so a `(D,)` mean subtracts from every row of an `(N, D)` matrix without `unsqueeze`. `torch.where` chooses per element, and `clamp` bounds the result.

## The two silent bugs
- **$N$ vs $N-1$.** `Tensor.std` defaults to the unbiased estimator (`correction=1`). For a single row that's $0/0 = $ `nan`; for small $N$ the scale is off by $\sqrt{N/(N-1)}$. NumPy's `np.std` defaults to the opposite, which is a classic porting bug. BatchNorm normalises with the population variance.
- **`where` doesn't stop `nan` from being computed.** `torch.where(std > 0, (x - mean) / std, 0)` returns the right forward values, but the division still produces `nan` in the masked branch, and in a backward pass `0 * nan = nan` leaks into the gradient. Dividing by a safe denominator avoids that.

## Complexity
$O(ND)$ time and memory; every step is one vectorised pass.

## Interview follow-ups
- Where should the mean and std come from at inference time? (The training set, stored with the model. Recomputing on each serving batch is training/serving skew.)
- How would you compute these statistics over a dataset too large for memory? (Running sums of $x$ and $x^2$, or Welford's update for better numerical behaviour.)
"""
