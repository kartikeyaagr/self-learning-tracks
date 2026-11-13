import numpy as np

META = dict(title='Soft-NMS with Gaussian and linear decay', tags=['Object Detection'], diff='Medium', runtime='numpy',
            lessons=[], related=['iou-matrix', 'anchor-box-generation'])

STATEMENT = r"""
Greedy NMS deletes every box that overlaps a higher-scoring one too much, which also deletes real objects standing close together. **Soft-NMS** instead lowers the scores of overlapping boxes in proportion to the overlap, and only drops a box once its score falls below a small threshold.

Write `soft_nms(boxes, scores, method="gaussian", sigma=0.5, iou_threshold=0.3, score_threshold=0.001)` and return `(keep, kept_scores)`.

## Algorithm
Maintain a working copy of the scores and a candidate set, initially every box whose score is `>= score_threshold`. Repeat while candidates remain:

1. Select the candidate with the **highest current score**; ties go to the **smallest index**. Append its index to `keep` and its current score to `kept_scores`, and remove it from the candidates.
2. For every remaining candidate `j`, let $u$ = IoU(selected, `j`) and rescale its score:
   - `"gaussian"`: $s_j \leftarrow s_j \cdot \exp(-u^2 / \sigma)$
   - `"linear"`: $s_j \leftarrow s_j \cdot (1 - u)$ if $u > $ `iou_threshold`, otherwise unchanged.
3. Remove every candidate whose score is now `< score_threshold`.

IoU uses `[x1, y1, x2, y2]` boxes with area `(x2 - x1) * (y2 - y1)`; all boxes have positive area.

## Input / output
- `boxes`: float array `(N, 4)`; `scores`: float array `(N,)`, `N >= 0`.
- `keep`: Python list of ints, in selection order. `kept_scores`: `float64` array of the same length (the decayed score at the moment of selection).
"""

STARTER = '''import numpy as np

def soft_nms(boxes, scores, method="gaussian", sigma=0.5, iou_threshold=0.3, score_threshold=0.001):
    """Return (keep: list of indices in selection order, kept_scores: float64 array)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _iou_one(box, others):
    lt = np.maximum(box[:2], others[:, :2])
    rb = np.minimum(box[2:], others[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(others) - inter)

def soft_nms(boxes, scores, method="gaussian", sigma=0.5, iou_threshold=0.3, score_threshold=0.001):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float64).copy()
    cand = np.flatnonzero(s >= score_threshold)          # ascending indices
    keep, kept = [], []
    while cand.size:
        best = cand[np.argmax(s[cand])]                  # argmax returns the first max -> smallest index
        keep.append(int(best)); kept.append(s[best])
        cand = cand[cand != best]
        if not cand.size:
            break
        u = _iou_one(boxes[best], boxes[cand])
        if method == "gaussian":
            s[cand] *= np.exp(-u**2 / sigma)
        else:
            s[cand] *= np.where(u > iou_threshold, 1 - u, 1.0)
        cand = cand[s[cand] >= score_threshold]
    return keep, np.array(kept, dtype=np.float64)
'''

HINTS = [
    'This is a loop of at most `N` rounds; each round selects one box and rescales the others. Keep the candidate set as an array of indices.',
    'Because the candidate indices stay sorted ascending, `cand[np.argmax(s[cand])]` already gives the tie rule for free.',
    'Each round: record `best`, drop it from `cand`, compute IoU of `best` against `boxes[cand]`, multiply `s[cand]` by the decay, then keep only `cand[s[cand] >= score_threshold]`.',
]

_r = np.random.default_rng(25)
_xy = _r.uniform(0, 20, (8, 2)); _wh = _r.uniform(4, 12, (8, 2))
_boxes = np.hstack([_xy, _xy + _wh])
_scores = _r.uniform(0.05, 1.0, 8)
B2 = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
TESTS = [
    (B2, np.array([0.9, 0.8, 0.7])),
    (B2, np.array([0.9, 0.8, 0.7]), "linear"),
    (np.array([[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 7.0, 7.0], [0.0, 0.0, 2.0, 2.0]]), np.array([0.5, 0.5, 0.5])),
    (B2, np.array([0.9, 0.0005, 0.7])),
    (np.zeros((0, 4)), np.zeros(0)),
    (_boxes, _scores),
    (_boxes, _scores, "linear", 0.5, 0.3, 0.2),
    (_boxes, _scores, "gaussian", 0.1, 0.3, 0.05),
]

EXPLANATION = r"""
## Why decay instead of delete
Greedy NMS is the special case "multiply by 0 if $u > $ threshold". Soft-NMS keeps a heavily overlapped box alive with a reduced score, so two people standing shoulder to shoulder both survive, while a near-duplicate ($u \approx 1$) is pushed almost to zero. Because AP is computed from a ranking, a lowered score is far less harmful than a deletion when the box was actually correct.

## Gaussian vs linear
The linear rule is discontinuous at the threshold; the Gaussian $e^{-u^2/\sigma}$ is smooth and has no threshold at all, so `iou_threshold` is unused for it. A smaller $\sigma$ suppresses more aggressively.

## Complexity
$O(N^2)$: up to $N$ rounds, each an $O(N)$ IoU against the remaining candidates. Greedy NMS has the same worst case but often finishes sooner because whole clusters vanish at once.

## Common bugs
- Sorting once at the start: after decay the order changes, so the maximum must be re-selected each round.
- Recording the original score instead of the decayed one.
- Forgetting the initial threshold filter (test 4) or the empty input (test 5).
- Decaying the already-kept boxes.

## Interview follow-ups
- Make it class-aware: run it per class without a loop over classes. (Offset each class's boxes by `class_id * (max_coord + 1)` so boxes of different classes never overlap.)
- How does Weighted Boxes Fusion differ? (It averages the coordinates of a cluster instead of picking one box.)
"""
