import numpy as np

META = dict(title='Per-group thresholds for equal true positive rates', tags=['Classification', 'Metrics & Evaluation'], diff='Medium', runtime='numpy')

STATEMENT = r"""
A classifier flags examples with `score >= t`. Write `equal_tpr_thresholds(scores, labels, groups, target_tpr)` that picks one threshold per group so each group's true positive rate is at least `target_tpr`, flagging as few examples as possible.

For each group (sorted by name):

- Let `pos` be that group's positive scores, sorted descending, and `m = ceil(target_tpr * len(pos))` (with a tiny tolerance: use `ceil(target_tpr * len(pos) - 1e-9)`).
- If `m == 0`, the threshold is `+inf`. Otherwise the threshold is the `m`-th highest positive score `pos[m - 1]`.
- Report `[threshold, tpr, fpr]` at that threshold, where FPR is over the group's negatives (0 if it has none). A group with no positives gets `[inf, 0.0, 0.0]`.

Return a dict from group name to that list.
"""

STARTER = '''import numpy as np

def equal_tpr_thresholds(scores, labels, groups, target_tpr):
    """One threshold per group meeting target_tpr."""
    t = float(np.quantile(np.asarray(scores)[np.asarray(labels) == 1], 1 - target_tpr))
    return {str(g): [t, target_tpr, 0.0] for g in sorted(set(groups))}
'''

SOLUTION = '''import math
import numpy as np

def equal_tpr_thresholds(scores, labels, groups, target_tpr):
    s, y, g = np.asarray(scores, float), np.asarray(labels), np.asarray(groups)
    out = {}
    for name in sorted(set(g.tolist())):
        m_g = g == name
        sg, yg = s[m_g], y[m_g]
        pos = np.sort(sg[yg == 1])[::-1]
        neg = sg[yg == 0]
        if len(pos) == 0:
            out[str(name)] = [float("inf"), 0.0, 0.0]
            continue
        m = math.ceil(target_tpr * len(pos) - 1e-9)
        t = float("inf") if m == 0 else float(pos[m - 1])
        tpr = float((pos >= t).mean())
        fpr = float((neg >= t).mean()) if len(neg) else 0.0
        out[str(name)] = [t, tpr, fpr]
    return out
'''

HINTS = [
    'The highest threshold that still catches m positives is exactly the m-th highest positive score.',
    'Ties above the threshold can push the TPR above the target; that is fine and unavoidable.',
    "Compute FPR on the group's own negatives, since the point is to see how the cost differs by group.",
]

_r = np.random.default_rng(21)
_g = _r.choice(["A", "B"], 600, p=[0.7, 0.3])
_y = (_r.random(600) < 0.3).astype(int)
_s = 0.3 + 0.35 * _y - 0.1 * (_g == "B") + 0.15 * _r.normal(size=600)
TESTS = [
    ([0.9, 0.8, 0.4, 0.3, 0.7, 0.2], [1, 1, 0, 1, 0, 0], ["a", "a", "a", "a", "b", "b"], 0.6),
    (_s, _y, _g, 0.8),
    (_s, _y, _g, 0.95),
    ([0.5, 0.6], [1, 1], ["x", "x"], 0.0),
    ([0.5, 0.6, 0.7], [0, 0, 1], ["x", "y", "y"], 0.5),
]

EXPLANATION = r"""
## What equal opportunity means
With one global threshold, a model whose scores are shifted for one group catches fewer of that group's true positives. Equalising TPR across groups (equal opportunity, Hardt et al. 2016) gives each group the same chance of being correctly flagged.

## The cost
Different thresholds usually give different false positive rates, and lower total accuracy than one global threshold. Fairness criteria trade against each other: in general you cannot equalise TPR, FPR and calibration at once when base rates differ.

## Product reality
Per-group thresholds need the group attribute at decision time, which may be unavailable or legally restricted. In many settings the better fix is upstream: data coverage, features, or a model that is calibrated within each group.

## Interview follow-ups
- Which metric would you equalise for a content moderation system, and why?
"""
