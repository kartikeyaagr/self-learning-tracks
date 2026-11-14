import numpy as np

META = dict(title='Choose a threshold under a precision constraint', tags=['Metrics & Evaluation', 'Classification'], diff='Medium', runtime='numpy',
            related=['precision-recall-curve-and-average-precision', 'area-under-roc-curve-via-trapezoidal-rule', 'expected-calibration-error'])

STATEMENT = r"""
A fraud team will only auto-block if precision is at least `min_precision`. Among thresholds that meet it, they want the highest recall. Write `pick_threshold(scores, y, min_precision)`.

- Candidate thresholds are the distinct values of `scores`. A row is predicted positive if `score >= threshold`.
- For each candidate compute precision and recall on (`scores`, `y`).
- Among candidates with precision $\ge$ `min_precision`, choose the one with the largest recall; on ties, the **largest** threshold (fewer flags for the same recall).
- Return `(threshold, precision, recall)` as floats, or `None` if no threshold qualifies.

Aim for $O(n \log n)$: sort once and use cumulative sums rather than recomputing counts for each threshold.
"""

STARTER = '''import numpy as np

def pick_threshold(scores, y, min_precision):
    """Return (threshold, precision, recall) or None."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pick_threshold(scores, y, min_precision):
    s, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=float)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]   # last index of each distinct score
    P = y.sum()
    best = None
    for i in last:
        prec = tp[i] / (tp[i] + fp[i])
        rec = tp[i] / P if P else 0.0
        if prec >= min_precision and (best is None or rec > best[2]):
            best = (float(s[i]), float(prec), float(rec))
    return best
'''

HINTS = [
    'Sort scores in decreasing order. Lowering the threshold to the k-th score flags the top k rows, so cumulative sums of y give TP and of 1 - y give FP.',
    'With tied scores, only the last position of each tie group is a valid cut; the threshold flags all of the group or none of it.',
    'Walking from high to low thresholds, recall never decreases, so keep the first threshold that reaches each new recall level: that is the largest one.',
]

_r = np.random.default_rng(31)
_y = (_r.random(60) < 0.3).astype(int)
_s = np.round(_y * 0.4 + _r.random(60) * 0.7, 2)
TESTS = [
    (np.array([0.9, 0.8, 0.7, 0.6, 0.5]), np.array([1, 1, 0, 1, 0]), 0.75),
    (np.array([0.9, 0.8, 0.7, 0.6, 0.5]), np.array([1, 1, 0, 1, 0]), 1.0),
    (np.array([0.5, 0.5, 0.5, 0.2]), np.array([1, 0, 1, 1]), 0.6),
    (np.array([0.3, 0.2]), np.array([0, 0]), 0.5),
    (_s, _y, 0.8),
    (_s, _y, 0.5),
]

EXPLANATION = r"""
## Why a constraint instead of F1
Business rules are usually asymmetric: "at most one in ten blocked transactions may be legitimate" is a precision constraint, and within it you want to catch as much fraud as possible. Maximising F1 picks some balance that nobody asked for.

## Ties and monotonicity
Precision is not monotonic in the threshold, so you must check every candidate; recall is monotonic, so the largest-recall candidate is the lowest qualifying threshold, and on equal recall the higher threshold flags fewer rows.

## Pick on validation, check on test
The threshold is a fitted parameter. Choose it on validation data, then report precision and recall on a separate test set, since the chosen point is optimistically biased. With few positives, precision at a high threshold is noisy; consider a confidence interval.

## Interview follow-ups
- How would you set thresholds per segment (merchant, region) without overfitting?
"""
