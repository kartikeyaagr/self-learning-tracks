import numpy as np

META = dict(title='Remove seasonality, then difference', tags=['Time Series', 'Data Processing'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:01.14'], related=['angle-features', 'boolean-masking', 'concat-and-correlate'])

STATEMENT = r"""
A daily metric often repeats a weekly pattern. A simple way to study what is left: estimate the average value at each **phase** of the cycle, subtract that repeating profile, and then look at day-to-day changes.

Write `deseasonalize_diff(x, period)` that returns `(profile, d)`.

## Input
- `x`: 1-D float array of length `n`, with `n >= period` and `n >= 2`.
- `period`: Python int `>= 1`.

## Steps
1. **Profile**: `profile[p]` is the mean of all `x[t]` with `t % period == p`, for `p = 0 .. period-1`. A cycle that is cut off at the end still contributes its samples.
2. **Seasonal component**: repeat `profile` end to end and cut it to length `n`, so position `t` holds `profile[t % period]`.
3. **Residual**: `r = x - seasonal`.
4. **Difference**: `d[t] = r[t+1] - r[t]` for `t = 0 .. n-2`.

## Output
`profile`: `float64` array `(period,)`. `d`: `float64` array `(n - 1,)`. No Python loops.

## Example
`x = [1, 5, 2, 6, 3]`, `period = 2`: the phase-0 values are `[1, 2, 3]` and the phase-1 values `[5, 6]`, so `profile = [2, 5.5]`. The seasonal component is `[2, 5.5, 2, 5.5, 2]`, the residual `[-1, -0.5, 0, 0.5, 1]` and `d = [0.5, 0.5, 0.5, 0.5]`.
"""

STARTER = '''import numpy as np

def deseasonalize_diff(x, period):
    """Return (profile (period,), d (n-1,)): phase means, then np.diff of x minus the tiled profile."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def deseasonalize_diff(x, period):
    x = np.asarray(x, dtype=float)
    n = len(x)
    phase = np.arange(n) % period
    # Per-phase sums divided by per-phase counts; bincount handles the incomplete last cycle.
    profile = np.bincount(phase, weights=x, minlength=period) / np.bincount(phase, minlength=period)
    seasonal = np.tile(profile, -(-n // period))[:n]          # repeat ceil(n / period) times, then trim
    return profile, np.diff(x - seasonal)
'''

HINTS = [
    'The phase of sample `t` is `t % period`. Every step is easy once you have `phase = np.arange(n) % period`.',
    'Per-phase means: `np.bincount(phase, weights=x)` sums the values in each phase and `np.bincount(phase)` counts them. (Padding to a multiple of `period` with NaN, reshaping to `(-1, period)` and using `np.nanmean(axis=0)` also works.)',
    '`np.tile(profile, k)` repeats the whole array `k` times; use `k = ceil(n / period)` and slice `[:n]`. Then `np.diff(x - seasonal)` gives the lag-1 differences.',
]

_r = np.random.default_rng(89)
_t = np.arange(45)
_weekly = 10 + 3 * np.sin(2 * np.pi * _t / 7) + 0.05 * _t + _r.normal(0, 0.3, size=45)
TESTS = [
    (np.array([1.0, 5.0, 2.0, 6.0, 3.0]), 2),
    (np.array([4.0, 4.0, 4.0]), 1),
    (np.array([1.0, 2.0, 3.0]), 3),
    (np.array([0.0, 10.0, 0.0, 10.0]), 2),
    (np.array([-5.0, 1e9, 2.0, -1e9, 7.0]), 4),
    (_weekly, 7),
    (_r.normal(size=20), 6),
]

EXPLANATION = r"""
## Why it works
Grouping by phase turns "average Mondays, average Tuesdays, ..." into one call: `np.bincount(phase, weights=x)` adds every value into the slot of its phase, and dividing by the unweighted counts gives the means, including for phases that appear one time fewer because the last cycle is cut off. `np.tile` repeats the profile as a block (`[a, b, a, b, ...]`), which is the seasonal component once trimmed to `n`. `np.diff` subtracts each value from the next, removing any constant offset or slow drift that survived.

Compare `np.repeat(profile, k)`, which repeats each **element** (`[a, a, b, b]`) and would be wrong here.

## Complexity
$O(n)$ time and memory.

## Common bugs
- `x.reshape(-1, period).mean(axis=0)` fails unless `n` is a multiple of `period`, and silently dropping the tail changes the means.
- Mixing up `np.tile` and `np.repeat`.
- Differencing before removing the seasonal profile. That is a different (also valid) transform, but the profile then no longer matches.
- `np.diff` returns `n - 1` values; aligning them with the wrong timestamps is a classic off-by-one in feature pipelines.

## Interview follow-ups
- What does a seasonal difference `x[t] - x[t - period]` do, and how is it written with slicing? (`x[period:] - x[:-period]`. It removes a fixed seasonal pattern without estimating it.)
- Why might you prefer the per-phase **median** to the mean here? (A single holiday or outage week distorts the mean profile, and the median is robust to it.)
"""
