import numpy as np

META = dict(title='Build a mel filterbank', tags=['Data Processing', 'Feature Engineering'], diff='Medium', runtime='numpy')

STATEMENT = r"""
Write `mel_filterbank(n_mels, n_fft, sr, fmin, fmax)` returning a matrix of shape `(n_mels, n_fft // 2 + 1)`.

- Use the HTK mel scale: $m(f) = 2595 \log_{10}(1 + f/700)$ and its inverse.
- Place `n_mels + 2` points equally spaced in mel between $m(f_{min})$ and $m(f_{max})$, and convert them back to Hz: $f_0 < f_1 < \dots < f_{n+1}$.
- Filter `k` (0-based) is a triangle that rises from 0 at $f_k$ to 1 at $f_{k+1}$ and falls back to 0 at $f_{k+2}$.
- Evaluate each triangle at the FFT bin frequencies $j \cdot sr / n_{fft}$ for $j = 0, \dots, n_{fft}/2$. No area normalisation.
"""

STARTER = '''import numpy as np

def mel_filterbank(n_mels, n_fft, sr, fmin, fmax):
    """Triangular filters equally spaced on the mel scale."""
    edges = np.linspace(fmin, fmax, n_mels + 2)
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    lo, c, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0, np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)))
'''

SOLUTION = '''import numpy as np

def mel_filterbank(n_mels, n_fft, sr, fmin, fmax):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (np.asarray(m, float) / 2595.0) - 1.0)
    edges = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    lo, c, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0.0, np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)))
'''

HINTS = [
    'Space the points evenly in mel, not in Hz; then low filters are narrow and high filters are wide.',
    'Each triangle is the minimum of a rising and a falling line, clipped below at zero.',
    'With few FFT bins and many mel bands, the lowest filters can fall between bins and come out all zero. That is a real failure mode, not a bug in your code.',
]

TESTS = [
    (3, 8, 8000, 0, 4000),
    (4, 16, 16000, 0, 8000),
    (10, 64, 16000, 20, 8000),
    (2, 4, 1000, 0, 500),
    (40, 32, 16000, 0, 8000),
]

EXPLANATION = r"""
## Why mel
Pitch perception is close to linear below about 1 kHz and close to logarithmic above it. The mel scale matches this, so a filterbank with equal spacing in mel gives fine resolution where speech carries the most information (vowel formants) and coarse resolution at high frequencies.

## The empty-filter trap
The last test asks for 40 bands from only 17 FFT bins. The lowest filters are narrower than the bin spacing and catch no bin at all, so their output is always zero and their log is the floor value. Real front ends avoid this by using a long enough FFT (400 samples at 16 kHz is standard for 80 bands).

## Variants
Libraries differ: librosa defaults to the Slaney mel scale and normalises each filter to equal area; HTK and many speech toolkits use the formula here with peak height 1. Mixing them silently changes model inputs, which is a common reason a model trained in one toolkit fails in another.

## Interview follow-ups
- Why do modern speech models still use mel features instead of learning the filterbank from raw audio?
"""
