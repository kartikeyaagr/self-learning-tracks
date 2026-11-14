import numpy as np

META = dict(title='Seasonal naive forecast', tags=['Time Series'], diff='Easy', runtime='numpy')

STATEMENT = r"""
The seasonal naive forecast repeats the last observed season.

Write `seasonal_naive(y, season, h)`: for a history `y` of length $T \ge$ `season`, forecast the next `h` values as
$$\hat y_{T+j} = y_{T - m + (j \bmod m)}, \quad j = 0, \dots, h-1,$$
with $m$ = `season`. Return a list of floats.
"""

STARTER = '''import numpy as np

def seasonal_naive(y, season, h):
    """Repeat the last full season of the history h steps ahead."""
    y = np.asarray(y, float)
    return [float(y[-season])] * h
'''

SOLUTION = '''import numpy as np

def seasonal_naive(y, season, h):
    y = np.asarray(y, float)
    last = y[len(y) - season:]
    return [float(last[j % season]) for j in range(h)]
'''

HINTS = [
    'Take the last season as a block and tile it forward.',
    'With season = 1 this is the plain naive forecast: repeat the last value.',
]

_t = np.arange(24 * 7)
_load = 50 + 10 * np.sin(2 * np.pi * _t / 24) + np.random.default_rng(0).normal(0, 1, len(_t))

TESTS = [
    ([1.0, 2.0, 3.0, 4.0], 2, 3),
    ([1.0, 2.0, 3.0, 4.0], 1, 2),
    ([5.0, 6.0, 7.0], 3, 7),
    (_load, 24, 30),
    (_load, 168, 5),
]

EXPLANATION = r"""
## Why start here
Electricity demand has strong daily and weekly cycles. Repeating yesterday at the same hour already captures most of the pattern, and it costs nothing. Any model you build has to beat this, under the same evaluation, or it is not worth running.

## Choosing the season
For hourly load, `season = 24` repeats yesterday and `season = 168` repeats the same hour last week, which also captures weekday and weekend differences. The weekly version is usually the stronger baseline for demand.

## Interview follow-ups
- A public holiday falls on a Wednesday. What does the weekly seasonal naive forecast do on the Thursday after, and how would you fix it?
"""
