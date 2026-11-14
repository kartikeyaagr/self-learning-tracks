import numpy as np

META = dict(title='Recursive multi-step forecast', tags=['Time Series'], diff='Medium', runtime='numpy',
            related=['lag-features'])

STATEMENT = r"""
A one-step model can forecast many steps ahead by feeding its own predictions back in as lags.

The model is linear in lagged values: $\hat y_t = c + \sum_i w_i\, y_{t - \ell_i}$ for lags $\ell_i$ in `lags`.

Write `recursive_forecast(history, lags, weights, intercept, h)`:
- Predict step $T$ from the history, append the prediction, predict step $T+1$, and so on for `h` steps.
- Whenever a needed lag falls inside the forecast period, use the earlier prediction.
- The history is always at least `max(lags)` long.

Return the `h` predictions as a list.
"""

STARTER = '''import numpy as np

def recursive_forecast(history, lags, weights, intercept, h):
    """Multi-step forecast by feeding predictions back as lag inputs."""
    y = list(map(float, history))
    x = [y[-l] for l in lags]
    p = intercept + float(np.dot(weights, x))
    return [p] * h
'''

SOLUTION = '''import numpy as np

def recursive_forecast(history, lags, weights, intercept, h):
    y = list(map(float, history))
    out = []
    for _ in range(h):
        p = intercept + float(sum(w * y[-l] for w, l in zip(weights, lags)))
        y.append(p); out.append(p)
    return out
'''

HINTS = [
    'Keep one growing list of values; lag l is always element -l of it.',
    'With a lag of 24 and h up to 24, no prediction is ever used as an input; errors start to feed back only after that.',
]

_t = np.arange(24 * 5)
_y = 50 + 10 * np.sin(2 * np.pi * _t / 24)

TESTS = [
    ([1.0, 2.0], [1], [1.0], 1.0, 3),
    ([1.0, 2.0, 3.0], [1, 2], [0.5, 0.5], 0.0, 4),
    (_y, [24], [1.0], 0.0, 30),
    (_y, [1, 24], [0.3, 0.6], 5.0, 48),
    ([10.0, 0.0, 10.0, 0.0], [2], [0.9], 0.0, 6),
]

EXPLANATION = r"""
## Recursive versus direct
The recursive strategy trains one model and applies it repeatedly. Its errors compound: a wrong forecast for 10 am becomes an input for 11 am. The direct strategy trains a separate model for each horizon (one for "1 hour ahead", one for "24 hours ahead") and avoids feedback, at the cost of more models and less data sharing. Many production systems mix the two.

## The seasonal lag
With only a 24-hour lag, predictions for the next day use only observed data, so no error feeds back until the horizon passes 24 steps. That is one reason seasonal lags are so effective for day-ahead demand forecasting.

## Interview follow-ups
- How would you produce prediction intervals for a recursive forecast?
"""
