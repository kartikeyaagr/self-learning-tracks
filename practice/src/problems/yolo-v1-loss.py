import numpy as np

META = dict(title='The YOLO v1 loss: responsible boxes, square-root sizes and weighted terms', tags=['Object Detection', 'Loss Functions'],
            diff='Hard', runtime='numpy', lessons=['dlcore:04.02', 'dlcore:04.03'],
            related=['iou-matrix', 'box-encode-and-decode', 'non-maximum-suppression', 'generalized-iou-loss'])

STATEMENT = r"""
YOLO v1 divides the image into an $S\times S$ grid. Each cell predicts $B$ boxes (each with a confidence) and one set of class probabilities, and the whole thing is trained with a single sum-squared loss. Implement it for one image.

Write `yolo_loss(pred, objects, B, lambda_coord, lambda_noobj)` returning `(total, coord, obj, noobj, cls)` as floats.

## Prediction layout
`pred` `(S, S, 5B + C)`. For cell $(i, j)$ (row $i$, column $j$) and box $b$: entries $5b..5b+4$ are $(x, y, w, h, c)$, where $x, y \in [0, 1]$ are offsets within the cell, $w, h$ are fractions of the image size, and $c$ is the confidence. The last $C$ entries are class probabilities.

## Targets
`objects` is a list of $(c_x, c_y, w, h, k)$ in image-normalised coordinates with class $k$. An object belongs to cell $i = \lfloor c_yS\rfloor$, $j = \lfloor c_xS\rfloor$, with cell offsets $c_xS - j$, $c_yS - i$. At most one object per cell.

## Responsible box
In an object's cell, convert each predicted box to image coordinates (centre $((j + x)/S, (i + y)/S)$, size $w, h$) and compute its IoU with the object. The box with the highest IoU is responsible (ties go to the lower index).

## Loss
- `coord` $= \lambda_{coord}\sum_{\text{resp}}\big[(x - \hat x)^2 + (y - \hat y)^2 + (\sqrt w - \sqrt{\hat w})^2 + (\sqrt h - \sqrt{\hat h})^2\big]$
- `obj` $= \sum_{\text{resp}} (c - \text{IoU})^2$, with that box's IoU as the target.
- `noobj` $= \lambda_{noobj}\sum_{\text{all other boxes}} c^2$ (including the non-responsible boxes in object cells).
- `cls` $= \sum_{\text{object cells}}\sum_k (p_k - [k = \hat k])^2$.
- `total` is the sum of the four.
"""

STARTER = '''import numpy as np

def yolo_loss(pred, objects, B, lambda_coord, lambda_noobj):
    """Return (total, coord, obj, noobj, cls)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)

def yolo_loss(pred, objects, B, lambda_coord, lambda_noobj):
    pred = np.asarray(pred, dtype=float)
    S = pred.shape[0]
    C = pred.shape[2] - 5 * B
    resp = np.zeros((S, S, B), dtype=bool)
    coord = obj = cls = 0.0
    for cx, cy, w, h, k in objects:
        i, j = int(np.floor(cy * S)), int(np.floor(cx * S))
        tx, ty = cx * S - j, cy * S - i
        cell = pred[i, j]
        ious = [_iou(((j + cell[5 * b]) / S, (i + cell[5 * b + 1]) / S, cell[5 * b + 2], cell[5 * b + 3]), (cx, cy, w, h))
                for b in range(B)]
        b = int(np.argmax(ious))
        resp[i, j, b] = True
        x, y, pw, ph, c = cell[5 * b:5 * b + 5]
        coord += (x - tx) ** 2 + (y - ty) ** 2 + (np.sqrt(pw) - np.sqrt(w)) ** 2 + (np.sqrt(ph) - np.sqrt(h)) ** 2
        obj += (c - ious[b]) ** 2
        cls += ((cell[5 * B:] - np.eye(C)[int(k)]) ** 2).sum()
    confs = pred[:, :, 4:5 * B:5]
    noobj = lambda_noobj * float((confs[~resp] ** 2).sum())
    coord *= lambda_coord
    return float(coord + obj + noobj + cls), float(coord), float(obj), noobj, float(cls)
'''

HINTS = [
    'The prediction\'s $x, y$ are relative to its cell, but IoU needs image coordinates: add the cell index and divide by $S$. Width and height are already image fractions.',
    'Only the responsible box gets coordinate and confidence targets. Every other box, in every cell, is pushed towards confidence 0 with weight $\\lambda_{noobj}$.',
    'Confidences sit at offsets $4, 9, 14, \\dots$ of each cell: `pred[:, :, 4:5*B:5]` collects them all, then mask out the responsible ones.',
]

_r = np.random.default_rng(2105)
def _pred(S, B, C):
    p = 1 / (1 + np.exp(-_r.normal(size=(S, S, 5 * B + C))))
    p[..., 5 * B:] /= p[..., 5 * B:].sum(-1, keepdims=True)
    return p
def _objs(S, n, C):
    cells = _r.choice(S * S, n, replace=False)
    out = []
    for c in cells:
        i, j = divmod(int(c), S)
        out.append(((j + _r.random()) / S, (i + _r.random()) / S, 0.1 + 0.5 * _r.random(), 0.1 + 0.5 * _r.random(), int(_r.integers(C))))
    return out
TESTS = [
    (_pred(2, 1, 2), _objs(2, 1, 2), 1, 5.0, 0.5),
    (_pred(3, 2, 3), _objs(3, 2, 3), 2, 5.0, 0.5),
    (_pred(4, 2, 5), _objs(4, 5, 5), 2, 5.0, 0.5),
    (_pred(7, 2, 4), _objs(7, 6, 4), 2, 5.0, 0.5),
    (_pred(3, 3, 2), _objs(3, 4, 2), 3, 1.0, 1.0),
]

EXPLANATION = r"""
## One network, one pass
YOLO v1 framed detection as a single regression from image to grid of boxes and class probabilities, with no proposals or sliding windows, which made it real-time (45 frames per second in the paper) at the cost of some accuracy, especially on small or clustered objects: each cell can only predict one class.

## Why the weights
Most cells contain no object, so their confidence errors would swamp everything else: $\lambda_{noobj} = 0.5$ down-weights them and $\lambda_{coord} = 5$ up-weights localisation. Regressing $\sqrt w, \sqrt h$ makes the same absolute error count more for a small box than for a large one.

## Responsibility
Assigning each object to the predictor whose current box overlaps it best lets the $B$ predictors in a cell specialise (for example, by aspect ratio) as training goes on.

## Common bugs
- Computing IoU with cell-relative centres.
- Leaving the non-responsible box in an object cell out of the no-object term.
- Using 1 as the confidence target; the paper's target is the IoU.

## Interview follow-ups
- What did anchors (YOLO v2) and multi-scale heads (v3) fix?
- Why do later detectors replace the squared-error class term with cross-entropy or focal loss?
"""
