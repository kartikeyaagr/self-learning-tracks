import numpy as np

META = dict(title='Region proposal network targets: label anchors and encode boxes', tags=['Object Detection', 'Computer Vision'],
            diff='Medium', runtime='numpy', lessons=['dlcore:04.03'],
            related=['anchor-box-generation', 'iou-matrix', 'box-encode-and-decode'])

STATEMENT = r"""
To train Faster R-CNN's region proposal network, every anchor needs an objectness label and, if it is positive, a regression target towards its ground-truth box.

Write `rpn_targets(anchors, gt, pos_thr, neg_thr)` returning `(labels, targets)`.

## IoU
Boxes are `(x1, y1, x2, y2)`, width $x_2 - x_1$, height $y_2 - y_1$. Compute the IoU matrix between `anchors` `(A, 4)` and `gt` `(G, 4)`.

## Labels `(A,)` ints
1. Start at $-1$ (ignored).
2. $0$ (background) where the anchor's best IoU over all ground truths is $<$ `neg_thr`.
3. $1$ where its best IoU is $\ge$ `pos_thr`.
4. Also $1$ for every anchor whose IoU with some ground truth equals that ground truth's highest IoU over all anchors, provided it is $> 0$ (so every object gets at least one positive, ties included). This step overrides step 2.

## Targets `(A, 4)`
For positive anchors, encode the ground truth with the highest IoU for that anchor (lowest index on ties): with centres $x_a, y_a$, sizes $w_a, h_a$ and likewise for the ground truth,
$t = \big((x - x_a)/w_a,\ (y - y_a)/h_a,\ \log(w/w_a),\ \log(h/h_a)\big)$. Rows for non-positive anchors are 0.
"""

STARTER = '''import numpy as np

def rpn_targets(anchors, gt, pos_thr, neg_thr):
    """Return (labels (A,), targets (A, 4))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda u: (u[:, 2] - u[:, 0]) * (u[:, 3] - u[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)

def _cwh(u):
    w, h = u[:, 2] - u[:, 0], u[:, 3] - u[:, 1]
    return u[:, 0] + w / 2, u[:, 1] + h / 2, w, h

def rpn_targets(anchors, gt, pos_thr, neg_thr):
    a, g = np.asarray(anchors, dtype=float), np.asarray(gt, dtype=float)
    iou = _iou(a, g)
    best, arg = iou.max(1), iou.argmax(1)
    labels = -np.ones(len(a), dtype=int)
    labels[best < neg_thr] = 0
    labels[best >= pos_thr] = 1
    gbest = iou.max(0)
    labels[((iou == gbest[None, :]) & (gbest[None, :] > 0)).any(1)] = 1
    ax, ay, aw, ah = _cwh(a)
    gx, gy, gw, gh = _cwh(g[arg])
    t = np.stack([(gx - ax) / aw, (gy - ay) / ah, np.log(gw / aw), np.log(gh / ah)], 1)
    t[labels != 1] = 0.0
    return labels, t
'''

HINTS = [
    'Compute the full $A\\times G$ IoU matrix once; its row maxima give the threshold labels and its column maxima find each object\'s best anchors.',
    'Apply the rules in order: background, then threshold positives, then the "best anchor for each object" positives, which can override a background label.',
    'Encode against the ground truth chosen by the row argmax for every anchor, then zero the rows that are not positive.',
]

_r = np.random.default_rng(2106)
def _anchors(n, img=64):
    c = _r.random((n, 2)) * img
    s = np.exp(_r.uniform(np.log(6), np.log(30), (n, 2)))
    return np.hstack([c - s / 2, c + s / 2])
def _gt_near(anchors, idx, jitter=3.0):
    return anchors[idx] + _r.normal(size=(len(idx), 4)) * jitter
_a1 = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30], [0, 0, 4, 4]], dtype=float)
_a2, _a3 = _anchors(30), _anchors(60)
_ag = np.array([[cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2] for cy in range(4, 64, 8) for cx in range(4, 64, 8) for s in (16, 24)], dtype=float)
TESTS = [
    (_a1, np.array([[1, 1, 11, 11]], dtype=float), 0.7, 0.3),
    (_a1, np.array([[2, 2, 12, 12], [22, 19, 33, 29]], dtype=float), 0.7, 0.3),
    (_a2, _gt_near(_a2, [0, 5, 9]), 0.7, 0.3),
    (_a3, _gt_near(_a3, [1, 2, 30, 44], jitter=6.0), 0.6, 0.4),
    (_a3, _gt_near(_a3, [7], jitter=1.0), 0.5, 0.5),
    (_ag, np.array([[10, 12, 30, 34], [40, 8, 58, 20], [20, 40, 52, 60]], dtype=float), 0.7, 0.3),
]

EXPLANATION = r"""
## What the RPN learns
At every feature-map position the RPN scores $k$ anchors (several scales and aspect ratios) for "object or not" and regresses an offset for each. These labels and targets are the supervision. The paper samples 256 anchors per image (up to half positive) for each loss computation; that sampling step is random and left out here.

## Why the extra positive rule
With a high threshold like 0.7, a small or oddly shaped object may have no anchor above it. Making each object's best-matching anchor positive guarantees it contributes to training. Anchors with IoU between the thresholds are ignored, so the classifier isn't forced to make a call on ambiguous boxes.

## The encoding
Offsets are measured in units of the anchor's size and sizes in log space, so the regression targets are roughly scale-invariant and a single head works across anchor sizes.

## Common bugs
- Letting the background rule run after the extra-positive rule.
- Picking only one anchor per object when several tie at the maximum.
- Mixing box conventions: here width is $x_2 - x_1$, while some older code used $x_2 - x_1 + 1$ for pixel-inclusive boxes.

## Interview follow-ups
- How do anchor-free detectors such as FCOS assign targets instead?
- Why does a two-stage detector need a second round of matching for the RoI head?
"""
