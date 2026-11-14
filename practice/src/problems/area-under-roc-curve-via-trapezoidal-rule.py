import numpy as np

META = dict(title='ROC curve and AUC with tied scores', tags=['Metrics & Evaluation', 'Classification'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:09.39'],
            related=['expected-value-via-numerical-integration', 'precision-recall-at-k'])

STATEMENT = r"""
A ROC curve plots the true-positive rate against the false-positive rate as the decision threshold sweeps from $+\infty$ down to $-\infty$. The area under it (AUC) is an integral, and with finitely many points the trapezoidal rule computes it exactly.

Write `roc_auc(scores, labels)` that returns `(fpr, tpr, auc)`.

## Building the curve
1. Let $P$ and $N$ be the numbers of positive and negative labels.
2. Take the **distinct** score values in decreasing order: $s_1 > s_2 > \dots > s_m$.
3. For threshold $s_i$, predict positive when `score >= s_i`. Count true positives $TP_i$ and false positives $FP_i$.
4. Prepend the point for "nothing predicted positive". So `fpr = [0, FP_1/N, ..., FP_m/N]` and `tpr = [0, TP_1/P, ..., TP_m/P]`, each of length $m + 1$, ending at `(1, 1)`.
5. `auc` = the trapezoidal-rule area under the polyline, as a Python `float`.

Samples with **tied** scores enter the curve together, so a tie between a positive and a negative becomes a diagonal segment and counts as half a correctly ranked pair.

## Inputs
- `scores`: float array `(n,)`. `labels`: int array `(n,)` of 0/1 with at least one of each.
- Don't use `sklearn` or `np.trapezoid`. Vectorise (sorting is fine, no loop over thresholds).
"""

STARTER = '''import numpy as np

def roc_auc(scores, labels):
    """Return (fpr, tpr, auc) with one ROC point per distinct score plus the (0, 0) start."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def roc_auc(scores, labels):
    s, y = np.asarray(scores, dtype=float), np.asarray(labels)
    order = np.argsort(-s, kind='stable')
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]   # last index of each tied group
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp                                         # everyone above the cut who isn't positive
    tpr = np.r_[0, tp] / tp[-1]
    fpr = np.r_[0, fp] / fp[-1]
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)          # trapezoids
    return fpr, tpr, float(auc)
'''

HINTS = [
    'Sort by score, descending. Then the samples predicted positive at each threshold are a prefix of the sorted array, and a cumulative sum of the labels counts the true positives.',
    'With ties, only keep the cumulative counts at the **last** element of each run of equal scores: `np.flatnonzero(np.diff(sorted_scores) != 0)` gives those positions, plus the final index.',
    'False positives at a cut are `(index + 1) - TP`. Prepend 0 to both counts, divide by $P$ and $N$, and sum `diff(fpr) * (tpr[1:] + tpr[:-1]) / 2`.',
]

_r = np.random.default_rng(12)
_y = (_r.random(40) < 0.4).astype(int)
_y[:2] = [0, 1]
TESTS = [
    (np.array([0.9, 0.8, 0.3, 0.1]), np.array([1, 1, 0, 0])),
    (np.array([0.9, 0.6, 0.6, 0.2]), np.array([1, 0, 1, 0])),
    (np.array([0.1, 0.2, 0.3]), np.array([1, 0, 0])),
    (np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 0, 1, 0])),
    (np.array([3.0, -1.0, 2.0, 2.0, 0.0, -1.0]), np.array([0, 1, 1, 0, 1, 0])),
    (np.array([1.0, 0.0]), np.array([0, 1])),
    (np.round(_y * 0.8 + _r.normal(size=40), 1), _y),
    (_r.normal(size=25), np.r_[np.ones(10, dtype=int), np.zeros(15, dtype=int)]),
]

EXPLANATION = r"""
## Why the trapezoid rule is exact here
The ROC "curve" built from finitely many thresholds is a polyline, and the trapezoid rule integrates straight segments exactly. Horizontal segments (a negative crosses the threshold) add $\Delta\text{FPR}\cdot\text{TPR}$; tied groups produce diagonal segments whose trapezoid adds half of the rectangle.

## What AUC means
AUC equals the probability that a random positive scores higher than a random negative, counting ties as $\tfrac12$. That's the Mann-Whitney $U$ statistic divided by $PN$, which gives an $O(n\log n)$ rank-based formula and a sanity check for this implementation.

## Complexity
One sort, $O(n \log n)$; the rest is linear.

## Common bugs
- Not grouping ties: each tied sample becomes its own step, and the answer depends on the sort order (test 4 would give anything from 0 to 1 instead of 0.5).
- Starting the curve at the first threshold instead of $(0, 0)$, which drops the first trapezoid.
- An unstable sort plus no tie grouping: results change between runs and libraries.

## Interview follow-ups
- Why can a model with AUC 0.95 still be useless on a dataset with 0.1% positives? What curve would you look at instead?
- Compute the AUC in $O(n\log n)$ from ranks alone, without building the curve.
"""
