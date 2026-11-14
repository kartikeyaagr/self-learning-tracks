import numpy as np

META = dict(title='Walk-forward validation splits', tags=['Time Series', 'Data Processing'], diff='Easy', runtime='python',
            related=['time-split-with-gap'])

STATEMENT = r"""
Write `walk_forward(n, initial, horizon, step, gap, window)` returning folds for a series of length `n`.

- Fold $f$ has training end $e_f = $ `initial` $+ f \cdot$ `step` (exclusive).
- Training starts at 0 if `window` is `None` (expanding window), otherwise at $\max(0, e_f - \text{window})$ (sliding window).
- The test block is $[e_f + \text{gap},\ e_f + \text{gap} + \text{horizon})$.
- Generate folds while the test block fits inside the series.

Return a list of `[train_start, train_end, test_start, test_end]`.
"""

STARTER = '''def walk_forward(n, initial, horizon, step, gap, window):
    """Expanding- or sliding-window folds that always test on the future."""
    folds, e = [], initial
    while e < n:
        folds.append([0, e, e, e + horizon])
        e += step
    return folds
'''

SOLUTION = '''def walk_forward(n, initial, horizon, step, gap, window):
    folds, e = [], initial
    while e + gap + horizon <= n:
        start = 0 if window is None else max(0, e - window)
        folds.append([start, e, e + gap, e + gap + horizon])
        e += step
    return folds
'''

HINTS = [
    'Check that the whole test block fits before adding a fold.',
    'A gap mirrors deployment: if forecasts are made at 9 am for the next day, the hours between are not available to the model.',
]

TESTS = [
    (10, 4, 2, 2, 0, None),
    (10, 4, 2, 2, 1, None),
    (10, 4, 2, 2, 0, 3),
    (100, 48, 24, 24, 0, None),
    (5, 4, 2, 1, 0, None),
    (24 * 10, 24 * 7, 24, 24, 12, 24 * 5),
]

EXPLANATION = r"""
## Why not a random split
A random split lets the model train on Thursday and test on Wednesday. Lag features then leak the answer, and the error estimate comes out far too optimistic. Walk-forward validation mimics deployment: train on the past, forecast the next block, move forward, repeat.

## Expanding versus sliding
An expanding window uses all history and suits stable patterns. A sliding window forgets old data, which helps when demand drifts (new buildings, new tariffs) at the cost of fewer training rows.

## Interview follow-ups
- Hyperparameters are tuned on the same folds used to report the error. What does that do to the reported number, and how would you avoid it?
"""
