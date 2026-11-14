import numpy as np

META = dict(title='Log-mel spectrogram', tags=['Data Processing', 'Feature Engineering'], diff='Medium', runtime='numpy',
            related=['mel-filterbank'])

STATEMENT = r"""
Write `log_mel(x, n_fft, hop, fb)` that turns a signal into log-mel features of shape `(n_frames, n_mels)`.

1. Cut full frames of length `n_fft` every `hop` samples (no padding).
2. Multiply each frame by a periodic Hann window, $w[n] = 0.5 - 0.5\cos(2\pi n / N)$ with $N$ = `n_fft`.
3. Power spectrum: $|\mathrm{rfft}(\text{frame})|^2$, length `n_fft // 2 + 1`.
4. Apply the filterbank `fb (n_mels, n_fft // 2 + 1)` and take $\ln(\cdot + 10^{-10})$.

Return an array. The signal is always at least one frame long.
"""

STARTER = '''import numpy as np

def log_mel(x, n_fft, hop, fb):
    """Framed, windowed power spectrum through a mel filterbank, then log."""
    x = np.asarray(x, float)
    n = (len(x) - n_fft) // hop + 1
    frames = np.stack([x[i * hop:i * hop + n_fft] for i in range(n)])
    return np.log(np.abs(np.fft.rfft(frames, axis=1)) @ np.asarray(fb).T + 1e-10)
'''

SOLUTION = '''import numpy as np

def log_mel(x, n_fft, hop, fb):
    x, fb = np.asarray(x, float), np.asarray(fb, float)
    n = (len(x) - n_fft) // hop + 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    P = np.abs(np.fft.rfft(x[idx] * w, axis=1)) ** 2
    return np.log(P @ fb.T + 1e-10)
'''

HINTS = [
    'The window tapers each frame to zero at the edges, which stops the cut from smearing energy across all frequencies.',
    'Use power (magnitude squared), not magnitude, before the filterbank.',
    'Multiply by fb transposed: (frames, bins) @ (bins, mels).',
]

def _fb(n_mels, n_fft, sr):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (np.asarray(m, float) / 2595.0) - 1.0)
    e = mel2hz(np.linspace(0, hz2mel(sr / 2), n_mels + 2))
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    return np.maximum(0.0, np.minimum((f - e[:-2, None]) / (e[1:-1, None] - e[:-2, None]), (e[2:, None] - f) / (e[2:, None] - e[1:-1, None])))

_t = np.arange(1600) / 16000

TESTS = [
    (np.sin(2 * np.pi * 1000 * _t), 64, 32, _fb(8, 64, 16000)),
    (np.sin(2 * np.pi * 300 * _t) + 0.5 * np.sin(2 * np.pi * 3000 * _t), 128, 64, _fb(16, 128, 16000)),
    (np.zeros(256), 64, 64, _fb(4, 64, 16000)),
    (np.ones(16), 16, 8, _fb(4, 16, 16000)),
    (np.random.default_rng(2).normal(size=512), 32, 16, _fb(6, 32, 16000)),
]

EXPLANATION = r"""
## The standard speech front end
Speech recognisers such as Whisper and most voice-activity networks take exactly this input: 25 ms windows, 10 ms hop, 80 mel bands, log compression. The network then learns from a picture with time on one axis and perceptual frequency on the other.

## Why each step
- **Window:** cutting a frame with sharp edges adds energy at every frequency. The Hann taper removes most of that leakage.
- **Power:** energy adds across a filter's bins, so filters should sum power, not amplitude.
- **Log:** compresses the huge dynamic range and turns a fixed gain (a louder microphone) into a constant offset, which normalisation can remove.

## Interview follow-ups
- Why do many models normalise log-mel features per utterance or with global statistics, and what does each choice do for a streaming system?
"""
