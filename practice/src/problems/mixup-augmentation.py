import numpy as np

META = dict(title='MixUp of a batch', tags=['Computer Vision', 'Training'], diff='Easy', runtime='numpy',
            lessons=[], related=['cutmix-augmentation', 'cross-entropy-logits'])

STATEMENT = r"""
**MixUp** trains on convex combinations of pairs of examples and of their labels. Given a mixing weight $\lambda \in [0, 1]$ and a permutation $\pi$ of the batch:

$$\tilde x_i = \lambda\, x_i + (1 - \lambda)\, x_{\pi(i)}, \qquad \tilde y_i = \lambda\, \mathbf e_{y_i} + (1 - \lambda)\, \mathbf e_{y_{\pi(i)}}$$

where $\mathbf e_k$ is the one-hot vector of class $k$. In practice $\lambda \sim \mathrm{Beta}(a, a)$ and $\pi$ is a random shuffle; here both are passed in so the result is deterministic.

Write `mixup(x, labels, num_classes, lam, perm)` and return `(mixed_x, mixed_y)`.

## Input
- `x`: array `(B, ...)` of any numeric dtype (images `(B, H, W, C)`, features `(B, D)`, ...).
- `labels`: int array `(B,)` in `0..num_classes-1`.
- `lam`: float in `[0, 1]`. `perm`: int array `(B,)`, a permutation of `0..B-1`.

## Output
- `mixed_x`: `float64` array with the shape of `x`.
- `mixed_y`: `float64` array `(B, num_classes)`. Each row sums to 1. When $y_i = y_{\pi(i)}$ the row is simply one-hot.
"""

STARTER = '''import numpy as np

def mixup(x, labels, num_classes, lam, perm):
    """Return (mixed_x float64 like x, mixed_y float64 (B, num_classes))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def mixup(x, labels, num_classes, lam, perm):
    x = np.asarray(x, dtype=np.float64)
    onehot = np.eye(num_classes)[np.asarray(labels)]          # (B, K)
    mixed_x = lam * x + (1 - lam) * x[perm]
    mixed_y = lam * onehot + (1 - lam) * onehot[perm]
    return mixed_x, mixed_y
'''

HINTS = [
    'Indexing the batch with `perm` gives the partner of every example at once: `x[perm]`.',
    'One-hot labels in one line: `np.eye(num_classes)[labels]`.',
    'Apply the same weighted sum to inputs and to one-hot labels, after casting `x` to float.',
]

_r = np.random.default_rng(32)
TESTS = [
    (np.array([[0.0, 10.0], [20.0, 30.0]]), np.array([0, 1]), 2, 0.75, np.array([1, 0])),
    (np.array([[1.0], [2.0], [3.0]]), np.array([2, 2, 0]), 3, 0.5, np.array([1, 2, 0])),
    (_r.integers(0, 256, size=(4, 3, 3, 3), dtype=np.uint8), np.array([0, 1, 2, 3]), 5, 0.3, np.array([3, 2, 1, 0])),
    (_r.normal(size=(5, 6)), _r.integers(0, 4, size=5), 4, 1.0, _r.permutation(5)),
    (_r.normal(size=(5, 6)), _r.integers(0, 4, size=5), 4, 0.0, _r.permutation(5)),
    (_r.normal(size=(6, 2, 2)), _r.integers(0, 3, size=6), 3, float(_r.beta(0.4, 0.4)), _r.permutation(6)),
]

EXPLANATION = r"""
## Why mixing helps
MixUp asks the model to behave **linearly** between training examples: halfway between a cat and a dog image, predict half cat and half dog. That regularises the decision boundaries, improves calibration, and reduces memorisation of noisy labels. With $\mathrm{Beta}(a, a)$ and small $a$ (0.2-0.4), most $\lambda$ values are close to 0 or 1, so most mixed images still look like one of the two.

## Training with soft targets
The loss is the cross-entropy against the soft label, which equals $\lambda\,\mathrm{CE}(\hat y, y_i) + (1 - \lambda)\,\mathrm{CE}(\hat y, y_{\pi(i)})$. Many implementations compute it that way and never build $\tilde y$.

## Complexity
$O(B \cdot \text{example size})$, one extra copy of the batch.

## Common bugs
- Mixing in `uint8`: `0.3 * x` on a uint8 array is fine, but `x + x[perm]` before scaling overflows.
- Using a different permutation for inputs and labels.
- Mixing a batch with itself in the same order (`perm = arange(B)`), which does nothing.

## Interview follow-ups
- How does MixUp interact with label smoothing? (Both soften targets; stacking them double-counts.)
- What's "manifold mixup"? (The same interpolation applied to hidden activations at a random layer.)
"""
