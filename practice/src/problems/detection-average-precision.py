import numpy as np

META = dict(title='Average precision for one detection class', tags=['Object Detection', 'Metrics & Evaluation'], diff='Medium', runtime='numpy',
            related=['iou-matrix', 'precision-recall-curve-and-average-precision'])

STATEMENT = r"""
Write `detection_ap(dets, gts, iou_thr)` for one class across several images.

- `dets`: list of `[image_id, score, x1, y1, x2, y2]`. `gts`: list of `[image_id, x1, y1, x2, y2]`. IoU uses continuous coordinates (area = width × height).
- Sort detections by score, highest first (stable, so ties keep input order).
- Each detection looks at the ground-truth boxes **in its own image** and takes the one with the highest IoU. If that IoU is at least `iou_thr` and the box is not yet matched, it is a true positive and the box becomes matched. Otherwise it is a false positive (a duplicate of a matched box counts as a false positive).
- Precision and recall after each detection: $P = TP/(TP+FP)$, $R = TP/N_{gt}$.
- AP (all-point interpolation): replace each precision by the maximum precision at any later point, then sum $\Delta R \times P$ over the points where recall increases.

Return a dict with `ap`, `precision`, `recall` (lists in sorted order). With no ground truth, `ap` is 0.0.
"""

STARTER = '''import numpy as np

def detection_ap(dets, gts, iou_thr):
    """All-point interpolated AP for one class, matching detections to ground truth by IoU."""
    d = sorted(dets, key=lambda r: -r[1])
    tp = np.ones(len(d))
    P = np.cumsum(tp) / np.arange(1, len(d) + 1)
    R = np.cumsum(tp) / max(len(gts), 1)
    return {"ap": float(P.mean()) if len(d) else 0.0, "precision": P.tolist(), "recall": R.tolist()}
'''

SOLUTION = '''import numpy as np

def _iou(b, G):
    x1 = np.maximum(b[0], G[:, 0]); y1 = np.maximum(b[1], G[:, 1])
    x2 = np.minimum(b[2], G[:, 2]); y2 = np.minimum(b[3], G[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1])
    ag = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1])
    return inter / (a + ag - inter)

def detection_ap(dets, gts, iou_thr):
    order = sorted(range(len(dets)), key=lambda i: -dets[i][1])
    by_img = {}
    for g in gts:
        by_img.setdefault(g[0], []).append(g[1:])
    by_img = {k: np.asarray(v, float) for k, v in by_img.items()}
    used = {k: np.zeros(len(v), bool) for k, v in by_img.items()}
    tp = []
    for i in order:
        img, box = dets[i][0], np.asarray(dets[i][2:], float)
        if img not in by_img:
            tp.append(0); continue
        ious = _iou(box, by_img[img])
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not used[img][j]:
            used[img][j] = True; tp.append(1)
        else:
            tp.append(0)
    tp = np.asarray(tp, float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    n = len(gts)
    P = ctp / np.maximum(ctp + cfp, 1e-12) if len(tp) else np.zeros(0)
    R = ctp / n if n else np.zeros(len(tp))
    if n == 0 or len(tp) == 0:
        return {"ap": 0.0, "precision": P.tolist(), "recall": R.tolist()}
    env = np.maximum.accumulate(P[::-1])[::-1]
    prev = np.r_[0.0, R[:-1]]
    ap = float(((R - prev) * env).sum())
    return {"ap": ap, "precision": P.tolist(), "recall": R.tolist()}
'''

HINTS = [
    'Group ground-truth boxes by image first; a detection can only match boxes in its own image.',
    'The envelope is a reversed running maximum: np.maximum.accumulate on the reversed precision array.',
    'Recall only increases at true positives, so false positives contribute nothing to the sum except by lowering later precision.',
]

TESTS = [
    ([[0, 0.9, 0, 0, 10, 10]], [[0, 0, 0, 10, 10]], 0.5),
    ([[0, 0.9, 0, 0, 10, 10], [0, 0.8, 1, 1, 10, 10]], [[0, 0, 0, 10, 10]], 0.5),
    ([[0, 0.9, 0, 0, 10, 10], [0, 0.95, 20, 20, 30, 30], [1, 0.7, 0, 0, 5, 5]], [[0, 0, 0, 10, 10], [1, 0, 0, 5, 5], [1, 10, 10, 20, 20]], 0.5),
    ([[0, 0.9, 0, 0, 10, 10], [0, 0.6, 2, 2, 12, 12]], [[0, 0, 0, 10, 10], [0, 2, 2, 12, 12]], 0.5),
    ([[0, 0.9, 3, 0, 13, 10], [0, 0.6, 2.5, 0, 12.5, 10]], [[0, 0, 0, 10, 10], [0, 4, 0, 14, 10]], 0.5),
    ([[2, 0.5, 0, 0, 1, 1]], [[0, 0, 0, 1, 1]], 0.5),
]

EXPLANATION = r"""
## How detectors are scored
A detection is only right if it lands on an object (IoU above a threshold) that no higher-scoring detection already claimed. Ranking all detections by confidence and tracing precision against recall gives a curve; AP is the area under its upper envelope. Mean AP (mAP) averages over classes, and COCO also averages over IoU thresholds from 0.5 to 0.95.

## Two subtleties the tests show
- **Duplicates are false positives.** In the second test the second box overlaps the same object; it cannot be counted again, which is why detectors need NMS.
- **Greedy best-IoU matching.** In the fifth test the first detection takes the right-hand object. The second detection's best overlap is that same object, so it is a false positive, even though it overlaps the left-hand object with IoU 0.6. COCO's evaluator matches each detection to the best *unmatched* box instead and would count it. Small rule differences like this are why AP from different toolkits does not always agree.

## Interview follow-ups
- Why does AP at IoU 0.5 reward a detector that localises loosely, and what does averaging over 0.5 to 0.95 change?
"""
