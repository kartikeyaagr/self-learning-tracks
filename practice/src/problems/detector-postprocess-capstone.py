import numpy as np

META = dict(title='Capstone: from raw detector outputs to mAP', tags=['Object Detection', 'Metrics & Evaluation'], diff='Hard', runtime='numpy',
            related=['box-encode-and-decode', 'non-maximum-suppression', 'detection-average-precision', 'iou-matrix'])

STATEMENT = r"""
Turn a detector's raw outputs into final detections and score them.

Write `detect_and_score(images, n_classes, score_thr, nms_iou, max_det, ap_iou)`. Each image is a dict with `anchors (A, 4)` as `x1, y1, x2, y2`, `deltas (A, 4)`, `scores (A, n_classes)`, `size` `[W, H]`, `gt_boxes` and `gt_labels`.

1. **Decode** each anchor: with centre $(a_x, a_y)$ and size $(a_w, a_h)$, the box has centre $(a_x + d_x a_w,\ a_y + d_y a_h)$ and size $(a_w e^{d_w},\ a_h e^{d_h})$. Clip corners to $[0, W] \times [0, H]$.
2. **Per class**, keep anchors with score strictly above `score_thr`, then run greedy NMS: sort by score (stable), keep the top box, drop the remaining boxes with IoU strictly above `nms_iou` against it, repeat.
3. Pool the kept detections of all classes in the image, sort by score (stable, class order then anchor order as the pre-sort order), keep the top `max_det`.
4. **Score**: for each class, compute AP over all images with the matching rules and all-point interpolation from the AP step, at IoU `ap_iou`. Classes with no ground truth are skipped.

Return a dict with `ap` (list per class, `None` for skipped classes), `map` (mean over non-skipped classes) and `n_dets` (detections kept per image).
"""

STARTER = '''import numpy as np

def detect_and_score(images, n_classes, score_thr, nms_iou, max_det, ap_iou):
    """Decode, filter, NMS and cap detections per image, then compute per-class AP and mAP."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _iou1(b, G):
    x1 = np.maximum(b[0], G[:, 0]); y1 = np.maximum(b[1], G[:, 1])
    x2 = np.minimum(b[2], G[:, 2]); y2 = np.minimum(b[3], G[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1]); ag = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1])
    return inter / np.maximum(a + ag - inter, 1e-12)

def _decode(anc, d, W, H):
    aw, ah = anc[:, 2] - anc[:, 0], anc[:, 3] - anc[:, 1]
    ax, ay = anc[:, 0] + aw / 2, anc[:, 1] + ah / 2
    cx, cy = ax + d[:, 0] * aw, ay + d[:, 1] * ah
    w, h = aw * np.exp(d[:, 2]), ah * np.exp(d[:, 3])
    B = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    B[:, [0, 2]] = B[:, [0, 2]].clip(0, W); B[:, [1, 3]] = B[:, [1, 3]].clip(0, H)
    return B

def _nms(B, s, thr):
    order = list(np.argsort(-s, kind="stable"))
    keep = []
    while order:
        i = order.pop(0); keep.append(i)
        if order:
            ious = _iou1(B[i], B[order])
            order = [j for j, v in zip(order, ious) if v <= thr]
    return keep

def _ap(dets, gts, thr):
    order = sorted(range(len(dets)), key=lambda i: -dets[i][1])
    by = {}
    for g in gts:
        by.setdefault(g[0], []).append(g[1:])
    by = {k: np.asarray(v, float) for k, v in by.items()}
    used = {k: np.zeros(len(v), bool) for k, v in by.items()}
    tp = []
    for i in order:
        img, box = dets[i][0], np.asarray(dets[i][2:], float)
        if img not in by:
            tp.append(0); continue
        ious = _iou1(box, by[img]); j = int(np.argmax(ious))
        if ious[j] >= thr and not used[img][j]:
            used[img][j] = True; tp.append(1)
        else:
            tp.append(0)
    if not tp:
        return 0.0
    tp = np.asarray(tp, float)
    ctp = np.cumsum(tp); P = ctp / np.arange(1, len(tp) + 1); R = ctp / len(gts)
    env = np.maximum.accumulate(P[::-1])[::-1]
    return float(((R - np.r_[0.0, R[:-1]]) * env).sum())

def detect_and_score(images, n_classes, score_thr, nms_iou, max_det, ap_iou):
    dets = [[] for _ in range(n_classes)]
    gts = [[] for _ in range(n_classes)]
    n_dets = []
    for k, im in enumerate(images):
        W, H = im["size"]
        B = _decode(np.asarray(im["anchors"], float), np.asarray(im["deltas"], float), W, H)
        S = np.asarray(im["scores"], float)
        pool = []
        for c in range(n_classes):
            idx = np.flatnonzero(S[:, c] > score_thr)
            for j in _nms(B[idx], S[idx, c], nms_iou):
                pool.append((float(S[idx[j], c]), c, B[idx[j]]))
        pool = sorted(pool, key=lambda t: -t[0])[:max_det]
        n_dets.append(len(pool))
        for s, c, b in pool:
            dets[c].append([k, s] + b.tolist())
        for b, c in zip(im["gt_boxes"], im["gt_labels"]):
            gts[c].append([k] + list(b))
    ap = [None if not gts[c] else _ap(dets[c], gts[c], ap_iou) for c in range(n_classes)]
    vals = [a for a in ap if a is not None]
    return {"ap": ap, "map": float(np.mean(vals)) if vals else 0.0, "n_dets": n_dets}
'''

HINTS = [
    'Decode first, then clip; clipping anchors before decoding gives different boxes.',
    'NMS is per class: a cat box should never suppress an overlapping dog box.',
    'Run the score threshold before NMS so the expensive step sees fewer boxes.',
    'A max_det cap that is too small lowers recall and therefore AP on crowded images.',
]

def _image(seed, objs, W, H, n_classes, noise, n_bg):
    g = np.random.default_rng(seed)
    anchors, deltas, scores = [], [], []
    for (x1, y1, x2, y2), c in objs:
        for _ in range(3):
            j = g.normal(size=4) * noise
            a = np.array([x1, y1, x2, y2], float) + g.normal(size=4) * 3
            anchors.append(a)
            aw, ah = a[2] - a[0], a[3] - a[1]
            tx = ((x1 + x2) / 2 - (a[0] + a[2]) / 2) / aw; ty = ((y1 + y2) / 2 - (a[1] + a[3]) / 2) / ah
            deltas.append([tx + j[0], ty + j[1], np.log((x2 - x1) / aw) + j[2], np.log((y2 - y1) / ah) + j[3]])
            s = g.uniform(0, 0.2, n_classes); s[c] = g.uniform(0.5, 0.95); scores.append(s)
    for _ in range(n_bg):
        x, y = g.uniform(0, W - 20), g.uniform(0, H - 20)
        anchors.append([x, y, x + 20, y + 20]); deltas.append(g.normal(size=4) * 0.1)
        scores.append(g.uniform(0, 0.6, n_classes))
    return {"anchors": np.round(anchors, 4).tolist(), "deltas": np.round(deltas, 4).tolist(), "scores": np.round(scores, 4).tolist(),
            "size": [W, H], "gt_boxes": [list(map(float, b)) for b, _ in objs], "gt_labels": [c for _, c in objs]}

_I1 = _image(1, [((10, 10, 50, 60), 0), ((60, 20, 100, 70), 1)], 128, 96, 3, 0.02, 4)
_I2 = _image(2, [((5, 5, 40, 40), 1), ((30, 30, 80, 90), 0), ((90, 10, 120, 50), 0)], 128, 96, 3, 0.05, 6)
_I3 = _image(3, [((20, 20, 60, 60), 2), ((25, 25, 65, 65), 2)], 100, 100, 3, 0.02, 2)

TESTS = [
    ([_I1], 3, 0.3, 0.5, 100, 0.5),
    ([_I1, _I2], 3, 0.3, 0.5, 100, 0.5),
    ([_I1, _I2, _I3], 3, 0.3, 0.5, 100, 0.5),
    ([_I1, _I2, _I3], 3, 0.3, 0.5, 2, 0.5),
    ([_I1, _I2, _I3], 3, 0.3, 0.9, 100, 0.5),
    ([_I1, _I2, _I3], 3, 0.3, 0.5, 100, 0.9),
]

EXPLANATION = r"""
## The back half of a detector
Everything after the network's forward pass is here: decode the predicted offsets against anchors, clip, threshold scores, suppress duplicates per class, cap the number of detections, and evaluate. Mistakes in this part change mAP as much as changes to the network.

## What the tests show
- **Crowded objects and NMS.** The third image holds two heavily overlapping objects of the same class. At an NMS threshold of 0.5 one of them is suppressed, which costs recall.
- **A loose NMS threshold.** At 0.9 (the fifth test) duplicates survive and the detection counts rise, yet AP does not fall: every duplicate here scores below all the true positives, so the precision envelope is unchanged. Both objects of the crowded pair are now found and class 2 rises to 1.0. AP tolerates low-scoring duplicates; it is hurt only when a duplicate outranks another object's true positive.
- **The detection cap.** With `max_det = 2` (the fourth test), images with three objects cannot reach full recall.
- **A strict IoU for scoring.** At an AP threshold of 0.9 (the last test) small localisation errors turn true positives into false positives.

## Interview follow-ups
- Soft-NMS lowers the scores of overlapping boxes instead of deleting them. Which test would it help, and what could it hurt?
"""
