import numpy as np

META = dict(title='Mean absolute scaled error', tags=['Time Series', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            related=['seasonal-naive-forecast'])

STATEMENT = r"""
MASE divides a forecast's mean absolute error by the in-sample error of the seasonal naive forecast on the training data.

Write `mase(y_true, y_pred, y_train, season)`:
$$\text{MASE} = \frac{\frac{1}{h}\sum_j |y_j - \hat y_j|}{\frac{1}{T-m}\sum_{t=m}^{T-1} |y_t - y_{t-m}|}.$$

The training series is always longer than `season` and its scale is never zero in the tests. Return a float.
"""

STARTER = '''import numpy as np

def mase(y_true, y_pred, y_train, season):
    """Forecast MAE scaled by the in-sample seasonal naive MAE."""
    err = np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
    return float(err / np.mean(np.abs(np.asarray(y_train))))
'''

SOLUTION = '''import numpy as np

def mase(y_true, y_pred, y_train, season):
    y_train = np.asarray(y_train, float)
    err = np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)))
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return float(err / scale)
'''

HINTS = [
    'The scale comes from the training data only, so it is the same for every model compared on that series.',
    'MASE below 1 means the forecast beat what the seasonal naive method achieved one season ahead in sample.',
]

_t = np.arange(24 * 14)
_y = 50 + 10 * np.sin(2 * np.pi * _t / 24) + np.random.default_rng(1).normal(0, 2, len(_t))

TESTS = [
    ([3.0, 4.0], [3.0, 4.0], [1.0, 2.0, 3.0], 1),
    ([3.0, 5.0], [4.0, 4.0], [1.0, 2.0, 4.0], 1),
    (_y[-24:], _y[-48:-24], _y[:-24], 24),
    (_y[-24:], np.full(24, _y[:-24].mean()), _y[:-24], 24),
    (_y[-24:], _y[-48:-24], _y[:-24], 1),
]

EXPLANATION = r"""
## Why scale the error
Mean absolute error is in megawatts, so it cannot be averaged across a small town and a large city. Percentage errors (MAPE) blow up near zero demand and punish over- and under-forecasts unequally. MASE (Hyndman and Koehler, 2006) is unit-free, defined whenever the training series is not constant, and has a natural reference point at 1.

## The tests
Predicting the mean for every hour (the fourth test) ignores the daily cycle and scores far above 1. The third and fifth tests score the same forecast against two scales. On this series a one-hour change (noise plus the daily swing) is larger than a one-day change (noise only), so the `season = 1` scale is larger and the same forecast looks better: 0.69 instead of 0.93. On a smoother series it goes the other way. Always report which season the scale uses.

## Interview follow-ups
- Why can MASE above 1 still be a good result when forecasting many steps ahead?
"""
