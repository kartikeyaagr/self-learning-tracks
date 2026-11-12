import numpy as np

META = dict(title='Weighted binary log loss', tags=['Loss Functions', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:01.11'], related=['cross-entropy-logits', 'logistic-regression', 'sigmoid-stable'])

STATEMENT = r"""
Log loss (binary cross-entropy) scores predicted probabilities $p_i = P(y_i = 1)$ against labels $y_i \in \{0, 1\}$:

$$\ell_i = -\big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big], \qquad \text{loss} = \frac{\sum_i w_i \ell_i}{\sum_i w_i}.$$

Write `log_loss(y_true, p, sample_weight=None, eps=1e-15)` that returns the weighted mean.

## Rules
- First clip every probability into $[\varepsilon, 1 - \varepsilon]$, so a confident wrong prediction gives a large but finite loss.
- If `sample_weight` is `None`, every weight is 1.
- `y_true`: int array `(n,)` of 0/1; `p`: float array `(n,)` in $[0, 1]$; `sample_weight`: non-negative float array `(n,)` with a positive sum, or `None`. $n \ge 1$.
- Natural log. Return a Python `float`.
"""

STARTER = '''import numpy as np

def log_loss(y_true, p, sample_weight=None, eps=1e-15):
    """Weighted mean binary cross-entropy with probabilities clipped to [eps, 1 - eps]."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def log_loss(y_true, p, sample_weight=None, eps=1e-15):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)          # log(0) would be -inf
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.sum(w * losses) / np.sum(w))
'''

HINTS = [
    'Clip first, then take logs. `np.clip(p, eps, 1 - eps)`.',
    'Each sample contributes $-\\log p_i$ if $y_i = 1$ and $-\\log(1 - p_i)$ if $y_i = 0$. The formula with $y_i$ and $1 - y_i$ selects the right term without branching.',
    'Replace a missing weight vector with ones, then divide the weighted sum by the sum of weights (not by $n$).',
]

_r = np.random.default_rng(109)
_y = _r.integers(0, 2, size=50)
TESTS = [
    (np.array([1, 0, 1]), np.array([0.9, 0.2, 0.6])),
    (np.array([1, 0]), np.array([0.5, 0.5])),
    (np.array([1, 0]), np.array([0.0, 1.0])),
    (np.array([1, 1, 0]), np.array([1.0, 0.8, 0.0])),
    (np.array([1, 0, 1, 0]), np.array([0.7, 0.4, 0.2, 0.1]), np.array([1.0, 2.0, 0.5, 0.0])),
    (np.array([0]), np.array([0.3]), None, 0.01),
    (_y, np.clip(0.5 + 0.4 * (2 * _y - 1) + 0.2 * _r.normal(size=50), 0.0, 1.0), _r.uniform(0.1, 3.0, size=50)),
]

EXPLANATION = r"""
## Why log loss
It's the negative log-likelihood of the labels under a Bernoulli model with probabilities $p_i$, so minimising it is maximum likelihood. It's a **proper scoring rule**: the expected loss is minimised only by predicting the true probability, so it rewards calibration, not just ranking. A constant prediction of 0.5 scores $\ln 2 \approx 0.693$ (test 2), a useful baseline.

## Why clip
A prediction of exactly 0 for a positive example has infinite loss, and one such row would make the mean infinite. Clipping at $10^{-15}$ caps each term at about $34.5$ (test 3). It's a convention, so state it in reports.

## Weights
Weighted log loss is used for class imbalance and importance sampling. Dividing by $\sum w_i$ keeps the scale comparable to the unweighted loss; a zero weight (test 5) removes a row entirely.

## Common bugs
- Taking `log` before clipping, which produces `-inf` and a runtime warning.
- Averaging over $n$ when weights are given.
- Computing it from probabilities when you have logits: use the stable logits form, $\mathrm{softplus}(z) - yz$.

## Interview follow-ups
- Your model's AUC improved but log loss got worse. What happened? (Ranking improved but calibration degraded; for example, over-confident probabilities.)
- How do you calibrate a classifier after training? (Platt scaling or isotonic regression on a held-out set.)
"""
