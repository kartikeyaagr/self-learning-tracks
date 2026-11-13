import numpy as np

META = dict(title='Distance-IoU and Complete-IoU losses', tags=['Loss Functions', 'Object Detection'], diff='Medium', runtime='numpy',
            lessons=[], related=['generalized-iou-loss', 'iou-matrix'])

STATEMENT = r"""
GIoU stops helping once one box contains the other. **DIoU** penalises the distance between the two box centres directly, normalised by the diagonal of the smallest enclosing box. **CIoU** adds a term for the mismatch in aspect ratio.

For a predicted box $A$ (width $w$, height $h$) and target $B$ ($w^g$, $h^g$):

$$\mathcal L_{\mathrm{DIoU}} = 1 - \mathrm{IoU} + \frac{\rho^2}{c^2}$$
$$\mathcal L_{\mathrm{CIoU}} = \mathcal L_{\mathrm{DIoU}} + \alpha v, \qquad v = \frac{4}{\pi^2}\left(\arctan\frac{w^g}{h^g} - \arctan\frac{w}{h}\right)^2, \qquad \alpha = \frac{v}{(1 - \mathrm{IoU}) + v}$$

- $\rho$ is the Euclidean distance between the two box centres.
- $c$ is the diagonal length of the smallest box enclosing both.
- If $(1 - \mathrm{IoU}) + v = 0$ (identical boxes), use $\alpha = 0$.

Write `diou_ciou_loss(pred, target)` and return the tuple `(diou, ciou)`.

## Input / output
- `pred`, `target`: float arrays `(N, 4)` of matched pairs `[x1, y1, x2, y2]`, all with positive width and height. Pair `k` compares `pred[k]` with `target[k]`.
- `diou`, `ciou`: `float64` arrays of shape `(N,)`.
"""

STARTER = '''import numpy as np

def diou_ciou_loss(pred, target):
    """pred, target: (N, 4) matched boxes. Return (diou_loss (N,), ciou_loss (N,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def diou_ciou_loss(pred, target):
    a, b = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    wa, ha = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
    wb, hb = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]

    iwh = np.clip(np.minimum(a[:, 2:], b[:, 2:]) - np.maximum(a[:, :2], b[:, :2]), 0, None)
    inter = iwh[:, 0] * iwh[:, 1]
    iou = inter / (wa * ha + wb * hb - inter)

    centre = lambda x: (x[:, :2] + x[:, 2:]) / 2
    rho2 = ((centre(a) - centre(b)) ** 2).sum(axis=1)
    ewh = np.maximum(a[:, 2:], b[:, 2:]) - np.minimum(a[:, :2], b[:, :2])
    c2 = (ewh ** 2).sum(axis=1)                       # squared enclosing diagonal, > 0 for positive boxes
    diou = 1 - iou + rho2 / c2

    v = 4 / np.pi**2 * (np.arctan(wb / hb) - np.arctan(wa / ha)) ** 2
    den = (1 - iou) + v
    alpha = np.divide(v, den, out=np.zeros_like(v), where=den > 0)
    return diou, diou + alpha * v
'''

HINTS = [
    'Reuse the IoU computation for matched pairs, then add terms. Every quantity is a `(N,)` array.',
    'Centres are `(x1 + x2) / 2` and `(y1 + y2) / 2`. The enclosing box is min of the top-lefts to max of the bottom-rights; its squared diagonal is `width**2 + height**2`.',
    'For CIoU, `v = 4/pi**2 * (arctan(wg/hg) - arctan(w/h))**2` and `alpha = v / ((1 - iou) + v)`, using `np.divide(..., where=den > 0)` to return 0 for identical boxes.',
]

_r = np.random.default_rng(31)
_xy = _r.uniform(0, 10, (6, 2)); _p = np.hstack([_xy, _xy + _r.uniform(0.5, 5, (6, 2))])
_xy = _r.uniform(0, 10, (6, 2)); _g = np.hstack([_xy, _xy + _r.uniform(0.5, 5, (6, 2))])
TESTS = [
    (np.array([[0.0, 0.0, 2.0, 2.0]]), np.array([[1.0, 1.0, 3.0, 3.0]])),
    (np.array([[0.0, 0.0, 4.0, 2.0]]), np.array([[1.0, 0.0, 3.0, 2.0]])),
    (np.array([[2.0, 2.0, 5.0, 6.0]]), np.array([[2.0, 2.0, 5.0, 6.0]])),
    (np.array([[0.0, 0.0, 10.0, 10.0], [4.0, 4.0, 6.0, 6.0]]), np.array([[3.0, 3.0, 5.0, 5.0], [0.0, 0.0, 10.0, 10.0]])),
    (np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([[3.0, 4.0, 4.0, 5.0]])),
    (_p, _g),
]

EXPLANATION = r"""
## Three geometric factors
A good box regression loss should care about overlap, centre distance and shape. IoU covers overlap. $\rho^2 / c^2 \in [0, 1)$ measures centre distance relative to the size of the pair, so it still produces a gradient when one box contains the other (test 4, where GIoU's penalty is zero). $v$ compares aspect ratios through $\arctan$, which keeps it bounded in $[0, 1)$.

## Why alpha looks like that
$\alpha = v / ((1 - \mathrm{IoU}) + v)$ gives the aspect-ratio term little weight while the boxes barely overlap and more weight once IoU is high and only the shape is left to fix. In training it is treated as a constant (no gradient through $\alpha$). Test 2 has concentric boxes with the same height: DIoU only sees the IoU, CIoU adds the shape penalty.

## Complexity
$O(N)$.

## Common bugs
- Using the enclosing box's **area** instead of its squared diagonal for $c^2$.
- Swapping $w/h$ for $h/w$ in one of the two arctangents (the square hides the sign, but not a mixed-up ratio).
- $0/0$ for identical boxes in $\alpha$ (test 3).

## Interview follow-ups
- DIoU is also used in NMS: suppress a box only if $\mathrm{IoU} - \rho^2/c^2$ exceeds the threshold. Why does that help with crowded scenes?
- Rank IoU, GIoU, DIoU and CIoU by how quickly they pull a far-away prediction toward its target.
"""
