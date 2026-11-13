import numpy as np

META = dict(title='Multi-class soft Dice loss', tags=['Loss Functions', 'Computer Vision'], diff='Easy', runtime='numpy',
            lessons=[], related=['mean-iou-for-segmentation', 'sigmoid-focal-loss'])

STATEMENT = r"""
Segmentation models are often trained with the **Dice loss**, a differentiable version of the overlap between the predicted and true masks. It ignores the huge number of easy background pixels that dominate pixel-wise cross-entropy.

For class $c$, with predicted probabilities $p$ and one-hot targets $g$, summing over all images and pixels:

$$D_c = \frac{2\sum p_c\, g_c + s}{\sum p_c + \sum g_c + s}, \qquad \mathcal L = 1 - \frac{1}{C}\sum_{c=0}^{C-1} D_c$$

where $s$ is a smoothing constant.

Write `dice_loss(probs, target, smooth=1.0)`.

## Input
- `probs`: float array `(N, C, H, W)` of class probabilities (e.g. a softmax over axis 1).
- `target`: int array `(N, H, W)` of class labels in `0..C-1`.
- `smooth`: float `>= 0`. With `smooth > 0` a class absent from both prediction and target gets $D_c = 1$.

## Output
The loss as a Python float.
"""

STARTER = '''import numpy as np

def dice_loss(probs, target, smooth=1.0):
    """probs: (N, C, H, W); target: (N, H, W) ints. Return 1 - mean per-class soft Dice."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dice_loss(probs, target, smooth=1.0):
    p = np.asarray(probs, dtype=np.float64)
    C = p.shape[1]
    g = (np.asarray(target)[:, None] == np.arange(C)[None, :, None, None]).astype(np.float64)   # one-hot (N, C, H, W)
    axes = (0, 2, 3)                                   # pool images and pixels, keep classes
    inter = (p * g).sum(axes)
    denom = p.sum(axes) + g.sum(axes)
    dice = (2 * inter + smooth) / (denom + smooth)
    return float(1 - dice.mean())
'''

HINTS = [
    'Turn the label map into a one-hot array with the same shape as `probs`, so both are `(N, C, H, W)`.',
    'One-hot by broadcasting: `target[:, None] == np.arange(C)[None, :, None, None]`.',
    'Sum `p * g`, `p` and `g` over axes `(0, 2, 3)` to get per-class totals, form the Dice ratio per class, and return `1 - mean`.',
]

_r = np.random.default_rng(29)
def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
_t = np.array([[[0, 1], [1, 1]]])
TESTS = [
    (np.stack([(_t == 0), (_t == 1)], axis=1).astype(float), _t),
    (np.full((1, 2, 2, 2), 0.5), _t, 0.0),
    (np.full((1, 3, 2, 2), 1 / 3), _t),
    (_softmax(_r.normal(size=(2, 4, 5, 5))), _r.integers(0, 4, size=(2, 5, 5))),
    (_softmax(_r.normal(size=(3, 2, 4, 6)) * 3), _r.integers(0, 2, size=(3, 4, 6)), 0.0),
    (_softmax(_r.normal(size=(1, 5, 3, 3))), np.zeros((1, 3, 3), dtype=int), 1e-6),
]

EXPLANATION = r"""
## Why Dice handles imbalance
If a tumour covers 0.1% of the pixels, predicting "background everywhere" gives 99.9% pixel accuracy and a small cross-entropy, but the tumour class has $D_c \approx 0$, so the Dice loss stays high. Each class contributes equally to the mean regardless of how many pixels it covers.

## Relation to IoU
For hard masks, Dice $= 2|A \cap B| / (|A| + |B|)$ and IoU $= |A \cap B| / |A \cup B|$ satisfy $D = 2J/(1 + J)$: they rank predictions identically. Dice is the one usually used as a loss because its soft version has a simpler gradient.

## The smoothing term
$s$ avoids $0/0$ for classes absent from both masks and stabilises gradients for small structures. A large $s$ biases the loss toward predicting nothing for rare classes, so it is typically 1 or smaller.

## Complexity
$O(N C H W)$ time and memory for the one-hot array.

## Common bugs
- Computing Dice per image and averaging, when the statement pools the batch (both variants exist; know which you need).
- Including the class axis in the sum, which collapses all classes into one ratio.
- Feeding logits instead of probabilities.

## Interview follow-ups
- Why do people often train with `cross_entropy + dice` rather than either alone?
- What is the Tversky loss and when would you weight false negatives more than false positives?
"""
