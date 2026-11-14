import numpy as np

META = dict(title='Recalibrate probabilities after negative downsampling', tags=['Classification', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            related=['expected-calibration-error', 'log-loss-binary-cross-entropy', 'stratified-sampling'])

STATEMENT = r"""
A click model was trained on data where every positive was kept but only a fraction $w$ of negatives. Its predictions $p'$ are therefore too high. Write `recalibrate(p_prime, w)` returning the corrected probabilities

$$p = \frac{p'}{p' + (1 - p')/w}.$$

Also write the inverse in the same function: if `w` is given as a negative number, treat it as $|w|$ and map true probabilities to downsampled-scale probabilities, $p' = \frac{p}{p + w(1-p)}$.

Return a float array. Assume $0 < |w| \le 1$ and inputs in $[0, 1]$.
"""

STARTER = '''import numpy as np

def recalibrate(p_prime, w):
    """w > 0: downsampled -> true probability. w < 0: true -> downsampled, with rate |w|."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def recalibrate(p_prime, w):
    p = np.asarray(p_prime, dtype=float)
    if w > 0:
        return p / (p + (1 - p) / w)
    w = -w
    return p / (p + w * (1 - p))
'''

HINTS = [
    'Think in odds: downsampling negatives by w multiplies the odds of a positive by 1/w.',
    'So the true odds are w times the model\'s odds: p / (1 - p) = w * p\' / (1 - p\'). Solve for p.',
    'The inverse divides the odds by w instead.',
]

TESTS = [
    (np.array([0.5]), 0.1),
    (np.array([0.0, 0.2, 0.5, 0.9, 1.0]), 0.05),
    (np.array([0.3, 0.7]), 1.0),
    (np.array([0.01, 0.02]), -0.1),
    (np.array([0.1, 0.4, 0.8]), 0.25),
]

EXPLANATION = r"""
## Derivation
Keeping a fraction $w$ of negatives leaves the positive count unchanged and scales the negative count by $w$, so the odds of a positive become $\frac{P(+)}{w\,P(-)}$, that is, the true odds divided by $w$. A model trained on that data learns those inflated odds. Multiplying its odds by $w$ undoes it:
$\frac{p}{1-p} = w \frac{p'}{1-p'} \Rightarrow p = \frac{p'}{p' + (1-p')/w}$.

## Why it matters
Ranking is unaffected, because the map is monotonic, so AUC does not change. But anything that uses the probability itself, such as an ad auction (bid times predicted CTR), expected-loss decisions or calibration dashboards, will be wrong by a large factor if the correction is skipped.

## Common bugs
- Using the wrong $w$ after a pipeline change (the sampling rate differs by region or day).
- Correcting twice, once in training and once in serving.

## Interview follow-ups
- How would you check calibration after the correction? (Reliability buckets on unsampled data.)
"""
