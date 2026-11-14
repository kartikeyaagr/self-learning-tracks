import numpy as np

META = dict(title='Remove ECG baseline wander', tags=['Time Series', 'Data Processing'], diff='Easy', runtime='numpy')

STATEMENT = r"""
Breathing and electrode movement add a slow drift to an ECG. A wide running median follows the drift but ignores the narrow heartbeat spikes.

Write `remove_baseline(x, fs, win_s)`:
- Window length $w$ = `int(win_s * fs)`, plus 1 if that is even, so the window is centred.
- Pad the signal by $(w-1)/2$ samples at each end, repeating the edge values.
- The baseline at sample $n$ is the median of the window centred on $n$.
- Return `x - baseline` as a list.
"""

STARTER = '''import numpy as np

def remove_baseline(x, fs, win_s):
    """Subtract a centred running median that tracks slow baseline drift."""
    x = np.asarray(x, float)
    w = int(win_s * fs)
    base = np.convolve(x, np.ones(w) / w, mode="same")
    return (x - base).tolist()
'''

SOLUTION = '''import numpy as np

def remove_baseline(x, fs, win_s):
    x = np.asarray(x, float)
    w = int(win_s * fs)
    if w % 2 == 0:
        w += 1
    r = (w - 1) // 2
    xp = np.pad(x, r, mode="edge")
    base = np.median(np.lib.stride_tricks.sliding_window_view(xp, w), axis=1)
    return (x - base).tolist()
'''

HINTS = [
    'np.lib.stride_tricks.sliding_window_view gives every window at once without copying.',
    'A median, unlike a mean, is not pulled up by the tall narrow QRS spike, so the beat survives the subtraction intact.',
    'The window should be longer than a QRS complex (about 0.1 s) and shorter than a breath (several seconds); 0.6 s is a common choice.',
]

def _ecg(seed, fs, secs, drift):
    g = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    x = drift * np.sin(2 * np.pi * 0.3 * t) + 0.02 * g.normal(size=len(t))
    for b in np.arange(0.4, secs, 0.8):
        x += np.exp(-((t - b) / 0.012) ** 2) - 0.2 * np.exp(-((t - b - 0.03) / 0.01) ** 2)
    return x

TESTS = [
    ([0.0, 0.0, 5.0, 0.0, 0.0], 1, 3),
    ([1.0, 2.0, 3.0, 4.0, 5.0], 1, 2),
    (_ecg(1, 100, 4, 0.5), 100, 0.6),
    (_ecg(2, 250, 3, 1.0), 250, 0.2),
    (np.full(6, 3.0), 10, 0.3),
]

EXPLANATION = r"""
## Why it matters
Beat detection thresholds and beat-shape features both assume a flat baseline. A drift of half a millivolt, small next to a normal R wave, can push the low parts of a beat above a threshold or make every beat look different from its neighbours.

## Median versus mean
A running mean spreads each QRS spike across the whole window, so subtracting it leaves a dent around every beat. A running median of a window much wider than the QRS ignores the spike almost completely. Clinical pipelines often use two medians in a row (about 200 ms, then 600 ms) to remove the P and T waves from the baseline estimate as well, or a high-pass filter around 0.5 Hz.

## Interview follow-ups
- Why can a high-pass filter distort the ST segment, and why does that matter clinically?
"""
