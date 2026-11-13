import numpy as np

META = dict(title='Greedy non-maximum suppression, class-aware', tags=['Object Detection', 'Computer Vision'], diff='Medium',
            runtime='numpy', lessons=[], related=['iou-matrix', 'precision-recall-curve-and-average-precision'])

STATEMENT = r"""
A detector fires many overlapping boxes around each object. Greedy NMS keeps the most confident box, deletes every remaining box that overlaps it too much, and repeats.

Write `nms(boxes, scores, iou_thresh, score_thresh=0.0, classes=None)` that returns the **list of kept indices** (Python ints) in the order they were kept.

## Inputs
- `boxes`: float `(n, 4)` as `[x1, y1, x2, y2]` with `x1 <= x2`, `y1 <= y2`; area is `(x2 - x1) * (y2 - y1)`. IoU is intersection over union, and 0 when the union is 0.
- `scores`: float `(n,)`. `classes`: optional int `(n,)`.

## Procedure
1. Discard every box with `score < score_thresh`.
2. Order the rest by score, highest first; **equal scores keep their original index order** (lower index first).
3. Walk that order. A box is kept unless it was suppressed; keeping box $i$ suppresses every later box $j$ with $\mathrm{IoU}(i, j) > $ `iou_thresh` (strictly greater) and, when `classes` is given, `classes[j] == classes[i]` (boxes of different classes never suppress each other).

## Output
A list of ints, possibly empty. `n` may be 0.
"""

STARTER = '''import numpy as np

def nms(boxes, scores, iou_thresh, score_thresh=0.0, classes=None):
    """Return the kept indices (list of int), in the order they are kept."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def nms(boxes, scores, iou_thresh, score_thresh=0.0, classes=None):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    cls = np.zeros(len(scores), dtype=int) if classes is None else np.asarray(classes)
    order = [int(i) for i in np.argsort(-scores, kind='stable') if scores[i] >= score_thresh]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    alive = np.ones(len(scores), dtype=bool)
    keep = []
    for pos, i in enumerate(order):
        if not alive[i]:
            continue
        keep.append(i)
        rest = np.array(order[pos + 1:], dtype=int)
        if rest.size == 0:
            break
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.prod(np.clip(rb - lt, 0, None), axis=1)
        union = area[i] + area[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        alive[rest[(iou > iou_thresh) & (cls[rest] == cls[i])]] = False
    return keep
'''

HINTS = [
    '`np.argsort(-scores, kind="stable")` sorts by descending score and keeps equal scores in index order. Filter out low scores while building the order.',
    'Keep a boolean `alive` array. Walk the order; skip dead boxes; for each kept box compute its IoU against every box after it in the order (one vectorised IoU of a box against many).',
    'Kill the boxes with `iou > iou_thresh` and the same class. With `classes=None` treat every box as one class. Return plain Python ints.',
]

_r = np.random.default_rng(127)
_c = _r.uniform(0, 20, size=(12, 2)); _wh = _r.uniform(2, 8, size=(12, 2))
_B = np.hstack([_c, _c + _wh])
_B = np.vstack([_B, _B[:6] + _r.uniform(-1, 1, size=(6, 4)).clip(-0.9, 0.9) * [1, 1, 0, 0]])
_S = _r.random(len(_B))
TESTS = [
    (np.array([[0.0, 0.0, 2.0, 2.0], [0.2, 0.0, 2.2, 2.0], [5.0, 5.0, 6.0, 6.0]]), np.array([0.9, 0.8, 0.7]), 0.5),
    (np.array([[0.0, 0.0, 2.0, 2.0], [0.2, 0.0, 2.2, 2.0], [5.0, 5.0, 6.0, 6.0]]), np.array([0.9, 0.8, 0.7]), 0.5, 0.75),
    (np.array([[0.0, 0.0, 3.0, 1.0], [1.0, 0.0, 4.0, 1.0]]), np.array([0.5, 0.6]), 0.5),
    (np.array([[0.0, 0.0, 3.0, 1.0], [1.0, 0.0, 4.0, 1.0]]), np.array([0.5, 0.6]), 0.49),
    (np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]), np.array([0.5, 0.5, 0.5]), 0.3, 0.0, np.array([1, 0, 1])),
    (np.zeros((0, 4)), np.zeros(0), 0.5),
    (_B, _S, 0.3),
    (_B, _S, 0.3, 0.2, _r.integers(0, 3, size=len(_B))),
    (np.array([[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 6.0, 6.0]]), np.array([0.9, 0.8, 0.7]), 0.2),
]

EXPLANATION = r"""
## Why greedy works well enough
Detectors produce dense, redundant predictions; NMS turns "many boxes per object" into "one box per object" under the assumption that two boxes of the same class with high overlap describe the same thing. It's greedy because the highest-scoring box is trusted most, and once kept it is never revisited.

## The chain case
In the last test, box 0 suppresses box 1 but **not** box 2 (they overlap less), and box 2 is not suppressed by box 1 because box 1 was already removed. A suppressed box suppresses nothing; that's why the loop checks `alive` before acting.

## Ties and thresholds
Sorting with a stable key keeps equal-score boxes in index order (test 5), and the strict `>` means an IoU of exactly the threshold survives (test 3 has IoU exactly 0.5; test 4 lowers the threshold just below it).

## Complexity
$O(n \log n)$ for the sort plus $O(n^2)$ IoUs in the worst case (few suppressions). Real implementations cap the number of candidates (top-k before NMS) and run per class in a batch by offsetting boxes of each class far apart, so one class-agnostic NMS does all classes at once.

## Common bugs
- Letting suppressed boxes suppress others.
- `>=` vs `>` on the IoU threshold, or forgetting the score threshold.
- Running class-agnostic NMS when classes are given: a person and a bicycle overlap a lot.

## Interview follow-ups
- What does Soft-NMS change, and why does it help crowded scenes?
- How would you implement batched (per-class) NMS with a single call to a class-agnostic routine?
"""
