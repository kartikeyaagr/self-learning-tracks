import numpy as np

META = dict(title='ROC AUC by ranking', tags=['Metrics & Evaluation', 'Classification'], diff='Medium', runtime='numpy',
            lessons=['sdsd:09.09'], related=['precision-recall-curve-and-average-precision', 'log-loss-binary-cross-entropy'])

STATEMENT = r"""
The area under the ROC curve equals the probability that a randomly chosen positive gets a higher score than a randomly chosen negative, with ties counting one half:

$$\mathrm{AUC} = \frac{1}{n_+ n_-}\sum_{i:\,y_i=1}\ \sum_{j:\,y_j=0}\Big([s_i > s_j] + \tfrac12[s_i = s_j]\Big).$$

Write `roc_auc(y_true, scores)` that computes it **without** forming all $n_+ n_-$ pairs.

## Rules
- `y_true`: int array `(n,)` of 0/1; `scores`: float array `(n,)`; $n \ge 1$.
- Equal scores must count as half, exactly as in the formula.
- If either class is absent, return `float('nan')`.
- Return a Python `float`. Aim for $O(n\log n)$.
"""

STARTER = '''import numpy as np

def roc_auc(y_true, scores):
    """P(score of a random positive > score of a random negative), ties = 1/2. nan if a class is missing."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def roc_auc(y_true, scores):
    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # 1-based ranks, with tied scores sharing the average of the ranks they occupy
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    # Mann-Whitney U: rank sum of positives minus its minimum possible value
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
'''

HINTS = [
    'Sort everything once. A positive at rank $r$ (1-based, ascending) beats every lower-ranked point, but some of those are other positives.',
    'Sum the ranks of the positives and subtract $n_+(n_+ + 1)/2$, the rank sum they would have if they were all at the bottom. What remains counts positive-over-negative pairs. Divide by $n_+ n_-$.',
    'For ties, give every member of a tied group the average of the ranks it spans: with `np.unique(s, return_inverse=True, return_counts=True)`, that\'s `cumsum(counts) - (counts - 1) / 2` indexed by `inv`.',
]

_r = np.random.default_rng(113)
_y = _r.integers(0, 2, size=200)
TESTS = [
    (np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])),
    (np.array([1, 0, 1, 0]), np.array([0.9, 0.9, 0.1, 0.5])),
    (np.array([0, 1]), np.array([0.3, 0.3])),
    (np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9])),
    (np.array([1, 1, 0, 0]), np.array([-5.0, -3.0, 2.0, 7.0])),
    (np.array([0, 1, 0, 1, 1, 0]), np.array([3.0, 3.0, 1.0, 2.0, 3.0, 2.0])),
    (_y, _y * 0.8 + _r.normal(size=200)),
    (_y, np.round(_r.uniform(size=200) + 0.3 * _y, 1)),
]

EXPLANATION = r"""
## Why the rank formula equals the pair count
Sort all scores ascending. A positive at rank $r$ is above $r - 1$ points. Summing over positives counts every (positive, lower point) pair; the positive-positive pairs among them number exactly $\binom{n_+}{2}$, and adding the $n_+$ self terms gives the $n_+(n_++1)/2$ correction. What's left is the number of positive-negative pairs ordered correctly. That's the Mann-Whitney $U$ statistic, so AUC is $U/(n_+n_-)$. Average ranks make a tied positive-negative pair contribute exactly one half.

## Why it equals the area
Sweeping a threshold from high to low traces the ROC curve: each negative passed moves right by $1/n_-$, and the height there is the fraction of positives already above it. Summing those heights is the pair count. With ties, the curve goes diagonally through a tied group, and the trapezoid it adds is the "half" credit.

## Complexity
$O(n\log n)$ for the sort inside `np.unique`, versus $O(n_+ n_-)$ for the pairwise definition.

## Common bugs
- Using `argsort().argsort()` for ranks, which breaks ties arbitrarily instead of averaging (test 6 has a three-way tie).
- Dividing by $n^2$ or $n_+ + n_-$.
- Crashing (division by zero) on a single-class input instead of returning `nan`.

## Interview follow-ups
- Why can AUC look great on a heavily imbalanced problem while precision is terrible? When is PR-AUC the better metric?
- AUC is invariant to any monotone transform of the scores. What does that tell you about whether AUC measures calibration?
"""
