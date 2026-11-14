import numpy as np

META = dict(title='Cut a window around every beat', tags=['Time Series', 'Data Processing'], diff='Easy', runtime='numpy')

STATEMENT = r"""
A beat classifier sees a fixed-length window of signal around each R peak.

Write `beat_windows(x, peaks, before, after)`:
- For each peak index $p$ (in order), the window is `x[p - before : p + after]`.
- Skip peaks whose window would run off either end of the signal.
- Also return the RR interval before each kept beat in samples: the distance to the **previous peak in the input list** (kept or not), or `None` for the first peak in the list.

Return a dict with `windows` (list of lists), `kept` (the kept peak indices) and `rr_prev` (one entry per kept peak).
"""

STARTER = '''import numpy as np

def beat_windows(x, peaks, before, after):
    """Fixed-length windows around R peaks, plus the preceding RR interval."""
    x = np.asarray(x, float)
    w = [x[max(0, p - before):p + after].tolist() for p in peaks]
    return {"windows": w, "kept": list(peaks), "rr_prev": [None] * len(peaks)}
'''

SOLUTION = '''import numpy as np

def beat_windows(x, peaks, before, after):
    x = np.asarray(x, float)
    W, kept, rr = [], [], []
    for k, p in enumerate(peaks):
        if p - before < 0 or p + after > len(x):
            continue
        W.append(x[p - before:p + after].tolist())
        kept.append(int(p))
        rr.append(None if k == 0 else int(p - peaks[k - 1]))
    return {"windows": W, "kept": kept, "rr_prev": rr}
'''

HINTS = [
    'Do not pad short windows: a classifier trained on padded edges learns that padding means something.',
    'Compute RR from the full peak list so that skipping the first beat does not change the second beat’s RR interval.',
    'Premature beats are recognised as much by their short RR interval as by their shape, which is why it is returned alongside the window.',
]

_x = np.sin(np.arange(40) / 3.0)

TESTS = [
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 4, 8], 1, 2),
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 5, 9], 2, 1),
    (_x, [5, 15, 22, 35], 4, 6),
    (_x, [], 2, 2),
    (_x, [3, 10], 3, 30),
]

EXPLANATION = r"""
## One example per heartbeat
Arrhythmia datasets such as the MIT-BIH database label individual beats, so a common set-up classifies a window of roughly 0.25 s before to 0.45 s after each R peak. The window holds the P wave, the QRS complex and the start of the T wave, which together show most beat types.

## Context beyond the window
Some beats differ from normal ones mainly in timing: a premature atrial beat can look almost normal but arrives early. Feeding the previous RR interval (and often the next one, or a local average) alongside the window gives the classifier that timing.

## A leakage trap
Beats from the same patient are highly correlated. Splitting train and test sets by beat instead of by patient inflates accuracy dramatically; the standard inter-patient split keeps whole records apart.

## Interview follow-ups
- Why does the inter-patient split give much lower accuracy for some beat classes than for others?
"""
