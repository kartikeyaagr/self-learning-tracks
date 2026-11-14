import numpy as np

META = dict(title='Frame energy in decibels', tags=['Data Processing', 'Time Series'], diff='Easy', runtime='numpy')

STATEMENT = r"""
Write `frame_energy_db(x, frame_len, hop, eps=1e-10)` for a mono signal `x`.

- Frame `i` covers samples `i*hop` to `i*hop + frame_len - 1`. Only full frames count, so there are $\lfloor (N - L)/h \rfloor + 1$ frames, or none if the signal is shorter than one frame.
- The energy of a frame is the mean of its squared samples.
- Return a list of $10\log_{10}(\text{energy} + \varepsilon)$, one per frame.
"""

STARTER = '''import numpy as np

def frame_energy_db(x, frame_len, hop, eps=1e-10):
    """Mean-square energy of each full frame, in decibels."""
    x = np.asarray(x, float)
    return [float(10 * np.log10(np.sum(x[i:i + frame_len] ** 2) + eps)) for i in range(0, len(x), hop)]
'''

SOLUTION = '''import numpy as np

def frame_energy_db(x, frame_len, hop, eps=1e-10):
    x = np.asarray(x, float)
    if len(x) < frame_len:
        return []
    n = (len(x) - frame_len) // hop + 1
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    e = (x[idx] ** 2).mean(1)
    return (10 * np.log10(e + eps)).tolist()
'''

HINTS = [
    'Build an index matrix of shape (n_frames, frame_len) and gather all frames at once.',
    'Use the mean, not the sum, so the value does not depend on the frame length.',
    'Decibels use 10 log10 for power (squared amplitude); 20 log10 is for amplitude.',
]

_r = np.random.default_rng(0)
_sig = np.r_[0.01 * _r.normal(size=400), np.sin(np.arange(400) * 0.3), 0.01 * _r.normal(size=400)]

TESTS = [
    ([1.0, 1.0, 1.0, 1.0], 2, 2),
    ([0.0, 0.0, 2.0, 2.0, 0.0], 2, 1),
    ([1.0, 2.0], 4, 1),
    (_sig, 160, 80),
    (np.zeros(10), 5, 5),
]

EXPLANATION = r"""
## Framing
Audio is analysed in short overlapping frames, typically 25 ms long with a 10 ms hop. Short enough that speech is roughly steady inside a frame, long enough to hold a few pitch periods.

## Why decibels
Loudness perception is roughly logarithmic, and a speech signal spans a large dynamic range. On a dB scale, a fixed threshold like "6 dB above the noise floor" means the same thing for a quiet and a loud recording.

## The epsilon
Digital silence is exactly zero, and $\log 0$ is $-\infty$. The epsilon sets a floor (here $-100$ dB) so that silence becomes a very small finite number instead.

## Interview follow-ups
- Why does a pure energy detector fail on music, typing noise or a crowded café?
"""
