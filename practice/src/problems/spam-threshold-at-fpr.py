import numpy as np

META = dict(title='Spam filter threshold at a false positive budget', tags=['Classification', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            related=['threshold-at-precision', 'auc-roc'])

STATEMENT = r"""
A spam filter must not send more than a fixed share of legitimate mail to the spam folder. Write `threshold_at_fpr(scores, labels, max_fpr)` where label 1 is spam and a message is flagged when `score >= t`.

- Consider as candidate thresholds every distinct score, plus `+inf` (flag nothing).
- Pick the **smallest** candidate whose false positive rate (flagged ham / all ham) is at most `max_fpr`.
- Return `(t, fpr, recall)` where recall is flagged spam / all spam. If there is no ham, FPR is 0; if there is no spam, recall is 0.
"""

STARTER = '''import numpy as np

def threshold_at_fpr(scores, labels, max_fpr):
    """Return (threshold, fpr, recall)."""
    t = float(np.quantile(scores, 1 - max_fpr))
    return t, max_fpr, 0.0
'''

SOLUTION = '''import numpy as np

def threshold_at_fpr(scores, labels, max_fpr):
    s, y = np.asarray(scores, float), np.asarray(labels)
    n_ham, n_spam = int((y == 0).sum()), int((y == 1).sum())
    for t in list(np.unique(s)) + [np.inf]:
        flag = s >= t
        fpr = (flag & (y == 0)).sum() / n_ham if n_ham else 0.0
        if fpr <= max_fpr:
            rec = (flag & (y == 1)).sum() / n_spam if n_spam else 0.0
            return float(t), float(fpr), float(rec)
'''

HINTS = [
    'FPR only falls as the threshold rises, so scan thresholds upward and stop at the first that fits.',
    'Ties matter: every message with the same score is flagged together, so a quantile can overshoot the budget.',
]

_r = np.random.default_rng(3)
_y = (_r.random(400) < 0.4).astype(int)
_s = np.clip(0.35 + 0.3 * _y + 0.2 * _r.normal(size=400), 0, 1)
TESTS = [
    ([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1], 0.0),
    ([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1], 0.5),
    (_s, _y, 0.01),
    (_s, _y, 0.1),
    ([0.9, 0.9, 0.9], [0, 1, 1], 0.2),
    ([0.2, 0.7], [1, 1], 0.0),
]

EXPLANATION = r"""
## Asymmetric costs
A spam message in the inbox is an annoyance; a job offer in the spam folder can cost someone a great deal. Spam systems therefore fix a false positive budget on legitimate mail and maximise recall within it, rather than optimising accuracy or F1.

## Why the smallest threshold
Among thresholds that meet the budget, the lowest one flags the most spam. With tied scores you cannot split a tie, which is why the achieved FPR can sit well below the budget.

## In production
The budget is usually set per user segment and re-checked after every model update, because a recalibrated model shifts the score distribution and an old threshold no longer gives the same FPR.

## Interview follow-ups
- How do you estimate FPR when users rarely report mail wrongly marked as spam?
"""
