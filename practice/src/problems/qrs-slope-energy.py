import numpy as np

META = dict(title='Slope energy for QRS detection', tags=['Time Series', 'Feature Engineering'], diff='Easy', runtime='numpy',
            related=['baseline-wander-removal'])

STATEMENT = r"""
QRS complexes are the steepest part of an ECG. Turn that into a detection signal, in the spirit of the Pan-Tompkins method.

Write `slope_energy(x, fs, win_s)`:
1. Difference: $d_n = x_n - x_{n-1}$, with $d_0 = 0$.
2. Square: $s_n = d_n^2$.
3. Trailing moving average over $w$ = `int(win_s * fs)` samples: $e_n$ is the mean of $s_{n-w+1}, \dots, s_n$, using only the available samples near the start (so $e_0 = s_0$, $e_1 = (s_0+s_1)/2$, …).

Return $e$ as a list.
"""

STARTER = '''import numpy as np

def slope_energy(x, fs, win_s):
    """Squared first difference, smoothed by a trailing moving average."""
    d = np.diff(np.asarray(x, float))
    return (d ** 2).tolist()
'''

SOLUTION = '''import numpy as np

def slope_energy(x, fs, win_s):
    x = np.asarray(x, float)
    d = np.r_[0.0, np.diff(x)]
    s = d ** 2
    w = max(int(win_s * fs), 1)
    c = np.r_[0.0, np.cumsum(s)]
    n = np.arange(1, len(s) + 1)
    lo = np.maximum(n - w, 0)
    return ((c[n] - c[lo]) / (n - lo)).tolist()
'''

HINTS = [
    'A cumulative sum turns every window mean into one subtraction.',
    'Squaring makes the signal positive and emphasises large slopes over small ones.',
    'The window should be about the width of a QRS complex (around 0.15 s); too short and one beat gives several peaks, too long and it merges with the T wave.',
]

def _ecg(seed, fs, secs):
    g = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    x = 0.02 * g.normal(size=len(t))
    for b in np.arange(0.4, secs, 0.8):
        x += np.exp(-((t - b) / 0.012) ** 2) + 0.3 * np.exp(-((t - b - 0.25) / 0.05) ** 2)
    return x

TESTS = [
    ([0.0, 1.0, 1.0, 3.0], 1, 2),
    ([0.0, 1.0, 1.0, 3.0], 1, 1),
    ([5.0], 10, 0.5),
    (_ecg(1, 100, 3), 100, 0.15),
    (_ecg(2, 250, 2), 250, 0.08),
]

EXPLANATION = r"""
## Why slope, not height
Tall T waves, artefacts and baseline drift can all reach the height of an R wave. Very few things are as steep. Differencing picks out the fast up- and down-strokes of the QRS, squaring makes them dominant, and the moving average merges the several slope peaks of one complex into a single hump per beat.

## The delay
The trailing window shifts each hump later than the R peak by roughly half the window. A detector that finds humps in this signal must look back in the filtered ECG to place the R peak exactly, which the capstone does.

## Interview follow-ups
- Pan and Tompkins band-pass filter the ECG (roughly 5 to 15 Hz) before differencing. What does that remove that differencing alone would amplify?
"""
