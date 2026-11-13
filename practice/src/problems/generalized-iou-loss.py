import numpy as np

META = dict(title='Generalized IoU loss', tags=['Loss Functions', 'Object Detection'], diff='Medium', runtime='numpy',
            lessons=[], related=['iou-matrix', 'complete-iou-loss', 'box-encode-and-decode'])

STATEMENT = r"""
Using $1 - \mathrm{IoU}$ as a box-regression loss has a flaw: if the predicted and true boxes don't overlap, IoU is 0 no matter how far apart they are, so the gradient is zero. **Generalized IoU** adds a penalty based on the smallest box $E$ that encloses both:

$$\mathrm{GIoU} = \mathrm{IoU} - \frac{|E| - |A \cup B|}{|E|}, \qquad \mathcal L_{\mathrm{GIoU}} = 1 - \mathrm{GIoU}$$

GIoU lies in $(-1, 1]$, so the loss lies in $[0, 2)$, and it keeps growing as non-overlapping boxes move apart.

Write `giou_loss(pred, target, reduction="none")`.

## Input
- `pred`, `target`: float arrays `(N, 4)` of **matched** pairs, `[x1, y1, x2, y2]` with `x1 < x2` and `y1 < y2` (positive area), `N >= 1`. Pair `k` compares `pred[k]` with `target[k]`; this is not a pairwise matrix.
- `reduction`: `"none"` returns the `(N,)` array of losses; `"mean"` and `"sum"` return a Python float.

Areas are `(x2 - x1) * (y2 - y1)`.
"""

STARTER = '''import numpy as np

def giou_loss(pred, target, reduction="none"):
    """pred, target: (N, 4) matched boxes. Return 1 - GIoU per pair, or its mean/sum."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def giou_loss(pred, target, reduction="none"):
    a, b = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    wh = np.clip(np.minimum(a[:, 2:], b[:, 2:]) - np.maximum(a[:, :2], b[:, :2]), 0, None)
    inter = wh[:, 0] * wh[:, 1]
    union = area(a) + area(b) - inter
    # Enclosing box: min of the top-lefts, max of the bottom-rights (always non-empty).
    ewh = np.maximum(a[:, 2:], b[:, 2:]) - np.minimum(a[:, :2], b[:, :2])
    enclose = ewh[:, 0] * ewh[:, 1]
    loss = 1 - (inter / union - (enclose - union) / enclose)
    if reduction == "none":
        return loss
    return float(loss.sum() if reduction == "sum" else loss.mean())
'''

HINTS = [
    'You need three areas per pair: intersection, union and the enclosing box.',
    'The intersection uses max of the top-lefts and min of the bottom-rights (clipped at 0). The enclosing box swaps them: min of the top-lefts, max of the bottom-rights.',
    '`loss = 1 - (inter / union - (enclose - union) / enclose)`, all as `(N,)` arrays. Reduce at the end.',
]

_r = np.random.default_rng(30)
_xy = _r.uniform(0, 10, (6, 2)); _p = np.hstack([_xy, _xy + _r.uniform(0.5, 5, (6, 2))])
_xy = _r.uniform(0, 10, (6, 2)); _g = np.hstack([_xy, _xy + _r.uniform(0.5, 5, (6, 2))])
TESTS = [
    (np.array([[0.0, 0.0, 2.0, 2.0]]), np.array([[1.0, 1.0, 3.0, 3.0]])),
    (np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]), np.array([[2.0, 0.0, 3.0, 1.0], [5.0, 0.0, 6.0, 1.0]])),
    (np.array([[1.0, 1.0, 4.0, 5.0]]), np.array([[1.0, 1.0, 4.0, 5.0]])),
    (np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([[4.0, 4.0, 6.0, 6.0]])),
    (_p, _g),
    (_p, _g, "mean"),
    (_p, _g, "sum"),
]

EXPLANATION = r"""
## Why the enclosing box helps
When boxes overlap, $|E|$ is close to $|A \cup B|$ and GIoU is close to IoU. When they don't, IoU is 0 but $(|E| - |A\cup B|)/|E|$ grows with the empty space between them, so the gradient pulls the prediction toward the target. Test 2 shows two non-overlapping pairs: both have IoU 0, but the farther pair has the larger loss.

## Properties
- GIoU $\le$ IoU, with equality exactly when the enclosing box equals the union, for example when one box contains the other (test 4, where $E = A$).
- It's scale-invariant, unlike an L1 loss on coordinates, so small and large objects contribute comparably.

## Complexity
$O(N)$ for matched pairs. A pairwise version (for matching cost matrices, as in DETR) broadcasts to $(N, M)$ just like a pairwise IoU.

## Common bugs
- Clipping the enclosing box's width at zero (it can't be negative anyway) but forgetting to clip the **intersection**.
- Using the union instead of $|E|$ in the penalty's denominator.
- Computing a pairwise `(N, N)` matrix when the inputs are matched pairs.

## Interview follow-ups
- GIoU converges slowly when one box sits inside the other, because the penalty is then zero. How do DIoU and CIoU fix that?
- Where does GIoU appear besides the loss? (In DETR's Hungarian matching cost, together with class probability and L1 distance.)
"""
