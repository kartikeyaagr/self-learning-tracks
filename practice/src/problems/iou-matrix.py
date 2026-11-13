import numpy as np

META = dict(title='IoU between two sets of boxes', tags=['Object Detection', 'Computer Vision'], diff='Easy', runtime='numpy',
            lessons=[], related=['precision-recall-at-k'])

STATEMENT = r"""
Intersection over union scores how well two axis-aligned boxes overlap:

$$\mathrm{IoU}(a, b) = \frac{\mathrm{area}(a \cap b)}{\mathrm{area}(a) + \mathrm{area}(b) - \mathrm{area}(a \cap b)}$$

Boxes are `[x1, y1, x2, y2]`, with `x1 <= x2` and `y1 <= y2` (corners, continuous coordinates, so area is `(x2-x1)*(y2-y1)`).

Write `iou_matrix(A, B)` that returns the `(n, m)` matrix of IoUs between every box in `A` `(n, 4)` and every box in `B` `(m, 4)`.

## Requirements
- Vectorised with broadcasting, no Python loops.
- Boxes that don't overlap have IoU `0`. If the union area is 0 (two degenerate boxes), the IoU is `0`.
"""

STARTER = '''import numpy as np

def iou_matrix(A, B):
    """A: (n, 4), B: (m, 4) boxes as [x1, y1, x2, y2]. Return (n, m) IoU."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def iou_matrix(A, B):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    lt = np.maximum(A[:, None, :2], B[None, :, :2])          # (n, m, 2) top-left of the overlap
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])          # (n, m, 2) bottom-right
    wh = np.clip(rb - lt, 0, None)                           # negative = no overlap
    inter = wh[..., 0] * wh[..., 1]
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    union = area(A)[:, None] + area(B)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
'''

HINTS = [
    'The overlap of two boxes is a box too: its top-left is the elementwise **max** of the two top-lefts, and its bottom-right is the **min** of the two bottom-rights.',
    'Index with `A[:, None, :2]` and `B[None, :, :2]` so the max and min broadcast to `(n, m, 2)`.',
    'Clip negative widths and heights to 0 before multiplying. Otherwise two negative sides multiply into a positive "overlap".',
]

_r = np.random.default_rng(4)
xy = _r.uniform(0, 10, size=(6, 2)); wh = _r.uniform(0.5, 4, size=(6, 2))
rand = np.hstack([xy, xy + wh])
TESTS = [
    (np.array([[0.0, 0.0, 2.0, 2.0]]), np.array([[1.0, 1.0, 3.0, 3.0]])),
    (np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([[2.0, 2.0, 3.0, 3.0], [0.0, 0.0, 1.0, 1.0]])),
    (np.array([[0.0, 0.0, 4.0, 4.0]]), np.array([[1.0, 1.0, 2.0, 2.0]])),
    (np.array([[0.0, 0.0, 2.0, 2.0]]), np.array([[3.0, -1.0, 5.0, 0.5]])),
    (rand[:3], rand[3:]),
    (np.array([[1.0, 1.0, 1.0, 1.0]]), np.array([[1.0, 1.0, 1.0, 1.0]])),
    (rand, rand),
]

EXPLANATION = r"""
## Max of starts, min of ends
On each axis the overlap interval is $[\max(a_1, b_1), \min(a_2, b_2)]$, and it's empty when that length is negative. Clipping at 0 handles every non-overlapping arrangement at once, including boxes that overlap on one axis but not the other (test 4).

## Where IoU is used
- **Matching** predictions to ground truth in training (anchor assignment) and in evaluation: a detection counts as a true positive if IoU ≥ 0.5 (or 0.5:0.95 for COCO mAP).
- **Non-max suppression**: drop a box if its IoU with a higher-scoring box exceeds a threshold.

## Pixel convention
Some codebases treat coordinates as inclusive pixel indices and use `x2 - x1 + 1`. That shifts small-box IoUs a lot, so check which convention a dataset uses.

## Interview follow-ups
- Implement greedy NMS using this function.
- Why use GIoU or DIoU as a loss? (Plain IoU has zero gradient when boxes don't overlap.)
"""
