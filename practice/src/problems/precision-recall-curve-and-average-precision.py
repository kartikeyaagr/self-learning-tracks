import numpy as np

META = dict(title='Precision-recall curve and average precision', tags=['Metrics & Evaluation', 'Classification'], diff='Medium', runtime='numpy',
            lessons=['sdsd:09.09'], related=['auc-roc', 'precision-recall-at-k'])

STATEMENT = r"""
A precision-recall curve shows the trade-off a classifier offers as its decision threshold moves. Average precision (AP) summarises it as a step-wise area.

Write `pr_curve(y_true, scores)` that returns `(precision, recall, thresholds, ap)`.

## Definitions
- `thresholds`: the **distinct** values of `scores`, in **descending** order.
- At threshold $t$, predict positive for every row with score $\ge t$. Then $\text{precision}_t = TP/(TP + FP)$ and $\text{recall}_t = TP / n_+$, where $n_+$ is the number of positive labels.
- `precision[k]` and `recall[k]` belong to `thresholds[k]`.
- $$\mathrm{AP} = \sum_k (R_k - R_{k-1})\,P_k, \qquad R_{-1} = 0.$$

## Input / output
- `y_true`: int array `(n,)` of 0/1 with at least one 1; `scores`: float array `(n,)`.
- Return three float arrays of the same length (one entry per distinct score) and `ap` as a Python `float`.
- Vectorise with a sort and cumulative sums: no loop over thresholds.
"""

STARTER = '''import numpy as np

def pr_curve(y_true, scores):
    """Return (precision, recall, thresholds descending, average precision)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pr_curve(y_true, scores):
    y, s = np.asarray(y_true, dtype=float), np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind='stable')
    s, y = s[order], y[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    # the last row of each run of equal scores closes that threshold: everything >= t is included
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp, fp = tp[last], fp[last]
    precision, recall = tp / (tp + fp), tp / y.sum()
    ap = np.sum(np.diff(recall, prepend=0.0) * precision)
    return precision, recall, s[last], float(ap)
'''

HINTS = [
    'Sort by score descending. Then "predict positive for the top $j$ rows" is just a prefix, and TP and FP for every prefix come from two cumulative sums.',
    'Tied scores must enter together, because a threshold can\'t split them. Keep only the **last** index of each run of equal sorted scores.',
    'AP: `np.sum(np.diff(recall, prepend=0) * precision)`. Recall only increases at thresholds that add a positive, so steps that add only negatives contribute 0.',
]

_r = np.random.default_rng(127)
_y = (_r.uniform(size=80) < 0.3).astype(int)
TESTS = [
    (np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])),
    (np.array([0, 1, 1]), np.array([0.5, 0.5, 0.2])),
    (np.array([1, 1, 0, 0]), np.array([4.0, 3.0, 2.0, 1.0])),
    (np.array([0, 0, 1, 1]), np.array([4.0, 3.0, 2.0, 1.0])),
    (np.array([1, 0, 1]), np.array([0.3, 0.3, 0.3])),
    (np.array([1]), np.array([-2.0])),
    (_y, _y + _r.normal(size=80)),
    (_y, np.round(_r.uniform(size=80) + 0.5 * _y, 1)),
]

EXPLANATION = r"""
## Why prefixes
Thresholding at $t$ selects exactly the rows whose score is at least $t$, which after a descending sort is a prefix. So one sort and two `cumsum`s give TP and FP for every threshold in $O(n\log n)$. Ties are the subtle part: a run of equal scores is all-in or all-out, so only the end of each run is a valid operating point (test 2 and test 5).

## Why this AP formula
AP is the area under the step function that holds precision $P_k$ over the recall interval $(R_{k-1}, R_k]$. It's the precision averaged over the positives, each positive weighted by when it's recovered: with no ties, AP equals the mean of precision@rank over the ranks of the positives. It doesn't interpolate linearly between points, which would be over-optimistic in PR space.

## Complexity
$O(n\log n)$ for the sort; everything else is linear.

## Common bugs
- Emitting one point per row instead of per distinct score.
- Using `>` instead of `>=`, which shifts every threshold by one row.
- Trapezoidal integration (`np.trapz`) of the PR curve: that's a different, optimistic quantity.
- Starting the recall at the first point instead of 0, which drops the first step of the area.

## Interview follow-ups
- Why is PR-AUC more informative than ROC-AUC under heavy class imbalance? (False positives are measured against predicted positives, not against the huge negative pool.)
- What's the expected AP of a random classifier? (The positive rate $n_+/n$, unlike ROC-AUC's fixed 0.5.)
"""
