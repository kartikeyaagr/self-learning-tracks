import numpy as np

META = dict(title='Calibration monitoring: ECE and MCE', tags=['MLOps', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:10.05'], related=['log-loss-binary-cross-entropy', 'delayed-label-monitoring'])

STATEMENT = r"""
A fraud model that says "0.8" should be right about 80% of the time it says so. Downstream thresholds and expected-cost calculations rely on that, and calibration often breaks after retraining or when traffic shifts, even when accuracy holds.

Bin predictions by confidence into $B$ equal-width bins. For each **non-empty** bin $b$ with $n_b$ predictions, let $\mathrm{conf}_b$ be the mean predicted probability and $\mathrm{acc}_b$ the fraction of positive labels. Then

$$\mathrm{ECE} = \sum_b \frac{n_b}{N}\,\bigl|\mathrm{acc}_b - \mathrm{conf}_b\bigr|, \qquad \mathrm{MCE} = \max_b \bigl|\mathrm{acc}_b - \mathrm{conf}_b\bigr|$$

Write `calibration_errors(probs, labels, n_bins=10)` that returns the tuple `(ece, mce)` as Python floats.

## Conventions
- `probs`: 1-D float array of predicted probabilities of the positive class, each in $[0, 1]$. `labels`: 1-D int array of 0/1, same length $N \ge 1$.
- A probability $p$ goes to bin $\lfloor p \cdot B \rfloor$, except $p = 1$, which goes to the last bin $B-1$.
- Empty bins are skipped: they add nothing to ECE and are not considered for MCE.
"""

STARTER = '''import numpy as np

def calibration_errors(probs, labels, n_bins=10):
    """Return (ece, mce) for binary probabilities and 0/1 labels."""
    p, y = np.asarray(probs, dtype=float), np.asarray(labels, dtype=float)
    gap = abs(y.mean() - p.mean())                # one global bin hides miscalibration
    return float(gap), float(gap)
'''

SOLUTION = '''import numpy as np

def calibration_errors(probs, labels, n_bins=10):
    p, y = np.asarray(probs, dtype=float), np.asarray(labels, dtype=float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)        # p == 1.0 joins the last bin
    n = np.bincount(b, minlength=n_bins)
    sum_p = np.bincount(b, weights=p, minlength=n_bins)
    sum_y = np.bincount(b, weights=y, minlength=n_bins)
    used = n > 0                                                # skip empty bins entirely
    gaps = np.abs(sum_y[used] - sum_p[used]) / n[used]          # |acc_b - conf_b|
    ece = np.sum(n[used] / len(p) * gaps)
    return float(ece), float(gaps.max())
'''

HINTS = [
    'Compute each prediction\'s bin index with `(p * n_bins).astype(int)` and clip it with `np.minimum(..., n_bins - 1)` so that `p = 1.0` is not out of range.',
    '`np.bincount(b, weights=p, minlength=n_bins)` gives per-bin sums of `p` in one call. Do the same for the labels and for plain counts.',
    'Keep only bins with a non-zero count. The per-bin gap is `|sum_y - sum_p| / n`. ECE weights the gaps by `n / N`, and MCE is their maximum.',
]

_rng = np.random.default_rng(11)
_p = _rng.uniform(0, 1, 500)
_y_cal = (_rng.uniform(0, 1, 500) < _p).astype(int)                # calibrated by construction
_p_over = np.clip(0.5 + 1.6 * (_p - 0.5), 0.0, 1.0)              # same labels, overconfident scores

TESTS = [
    (np.array([0.05, 0.15, 0.85, 0.95]), np.array([0, 0, 1, 1]), 10),
    (np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1]), np.array([1, 1, 0, 0, 0, 1]), 5),
    (np.array([1.0, 1.0, 0.0]), np.array([1, 0, 0]), 4),
    (np.array([0.55]), np.array([1]), 10),
    (np.array([0.25, 0.75, 0.5, 0.125]), np.array([0, 1, 1, 0]), 2),
    (np.array([0.62, 0.64, 0.66, 0.68]), np.array([1, 1, 1, 1]), 1),
    (_p, _y_cal, 10),
    (_p_over, _y_cal, 15),
]

EXPLANATION = r"""
## Why it works
Within a bin, every prediction claims roughly the same probability, so the bin's empirical positive rate estimates what that confidence level *should* have been. ECE averages the gaps weighted by how much traffic each bin gets, which is what matters for aggregate decisions. MCE reports the worst bin, which matters when one confidence range drives an action (for example, auto-blocking above 0.9).

The two random tests use the same labels. Test 7 draws labels with $P(y=1) = p$, so it's calibrated up to sampling noise. Test 8 stretches the scores away from 0.5, which keeps the ranking (same AUC) but raises ECE: ranking metrics can't see calibration drift.

## Complexity
$O(N + B)$ with `bincount`, no sorting.

## Common bugs
- `p = 1.0` gives index `B`, one past the end (test 3).
- Using `np.linspace(0, 1, B+1)` with `np.digitize`: decimal boundaries such as 0.3 aren't exact in binary, so values sitting on a boundary can land in a different bin than $\lfloor pB \rfloor$ gives.
- Dividing by the number of bins instead of weighting by $n_b / N$, which over-weights nearly empty bins.
- Including empty bins in MCE as a gap of 0 or `nan`.

## Interview follow-ups
- ECE with equal-width bins is noisy when most predictions sit near 0. What changes with equal-mass (quantile) bins, and what's the trade-off?
- ECE alerts after a retrain. How do you fix calibration without retraining the model? (Fit temperature scaling, Platt scaling or isotonic regression on a recent labelled holdout.)
"""
