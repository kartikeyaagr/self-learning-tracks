import numpy as np

META = dict(title='Mean IoU from a confusion matrix', tags=['Computer Vision', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            lessons=[], related=['dice-loss', 'iou-matrix', 'top-k-classification-accuracy'])

STATEMENT = r"""
The standard semantic-segmentation metric is **mean intersection over union**. For each class $c$, count pixels predicted as $c$ and labelled $c$ (intersection) against pixels predicted **or** labelled $c$ (union), then average over classes.

Write `mean_iou(pred, gt, num_classes, ignore_index=255)` and return `(per_class, miou)`.

## Input
- `pred`: int array of any shape (e.g. `(H, W)` or `(N, H, W)`) with values in `0..num_classes-1`.
- `gt`: int array of the same shape, values in `0..num_classes-1` or equal to `ignore_index`.
- Pixels where `gt == ignore_index` are excluded from every count, whatever `pred` says there.

## Output
- `per_class`: `float64` array `(num_classes,)`. Entry `c` is $\frac{|P_c \cap G_c|}{|P_c \cup G_c|}$, or `nan` if the union is empty (the class appears in neither the prediction nor the ground truth of the counted pixels).
- `miou`: Python float, the mean of the non-`nan` entries; `nan` if every entry is `nan`.

## Example
`pred = [0, 0, 1, 1]`, `gt = [0, 1, 1, 255]` (with the ignored last pixel): class 0 has intersection 1 and union 2, class 1 has intersection 1 and union 2, so `per_class = [0.5, 0.5]` and `miou = 0.5`.
"""

STARTER = '''import numpy as np

def mean_iou(pred, gt, num_classes, ignore_index=255):
    """Return (per-class IoU with nan for absent classes, mean over present classes)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def mean_iou(pred, gt, num_classes, ignore_index=255):
    pred, gt = np.asarray(pred).ravel(), np.asarray(gt).ravel()
    keep = gt != ignore_index
    K = num_classes
    # Confusion matrix in one bincount: row = true class, column = predicted class.
    conf = np.bincount(gt[keep] * K + pred[keep], minlength=K * K).reshape(K, K)
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - inter
    per_class = np.divide(inter, union, out=np.full(K, np.nan), where=union > 0)
    present = ~np.isnan(per_class)
    miou = float(per_class[present].mean()) if present.any() else float('nan')
    return per_class, miou
'''

HINTS = [
    'Everything follows from the `K x K` confusion matrix of (true class, predicted class) counts over the non-ignored pixels.',
    'Build it with one `np.bincount` on the combined index `gt * K + pred`, with `minlength=K*K`, then reshape. Intersection is the diagonal; union is row sum plus column sum minus the diagonal.',
    'Use `np.divide(inter, union, out=np.full(K, np.nan), where=union > 0)` so absent classes become `nan`, then average only the finite entries.',
]

_r = np.random.default_rng(39)
_gt = _r.integers(0, 4, size=(2, 6, 6)); _gt[_r.uniform(size=_gt.shape) < 0.1] = 255
TESTS = [
    (np.array([0, 0, 1, 1]), np.array([0, 1, 1, 255]), 2),
    (np.array([[0, 1], [1, 1]]), np.array([[0, 1], [1, 1]]), 3),
    (np.array([[2, 2], [2, 2]]), np.array([[0, 0], [1, 1]]), 3),
    (np.array([1, 2]), np.array([255, 255]), 3),
    (_r.integers(0, 4, size=(2, 6, 6)), _gt, 4),
    (_r.integers(0, 5, size=(8, 8)), _r.integers(0, 5, size=(8, 8)), 5, -1),
    (np.array([0, 1, 2, 3]), np.array([0, 1, 2, -1]), 4, -1),
]

EXPLANATION = r"""
## One bincount
Each counted pixel maps to a unique cell `gt * K + pred` of a flattened `K x K` matrix, so a single `np.bincount` builds the whole confusion matrix in $O(\text{pixels})$. The column sum is $|P_c|$, the row sum is $|G_c|$, and their sum minus the diagonal is the union.

## Absent classes
A class missing from both prediction and ground truth has no meaningful IoU: counting it as 1 inflates the score, counting it as 0 punishes the model for something it got right. Reporting `nan` and averaging the rest is the usual convention. On a real benchmark you accumulate the confusion matrix over the **whole dataset** first and compute IoUs once; averaging per-image mIoUs gives a different, noisier number.

## Complexity
$O(\text{pixels} + K^2)$.

## Common bugs
- Not masking `pred` with the same ignore mask as `gt`.
- Leaving `ignore_index` pixels in, which crashes `bincount` (or worse, silently creates a huge array).
- Using `np.mean` instead of averaging only present classes, which returns `nan` whenever one class is absent.

## Interview follow-ups
- Compute frequency-weighted IoU and pixel accuracy from the same confusion matrix.
- Why might mIoU and per-pixel accuracy disagree about which of two models is better? (Accuracy is dominated by large classes like road and sky.)
"""
