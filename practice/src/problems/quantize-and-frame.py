import numpy as np

META = dict(title='Mu-law codes in overlapping frames', tags=['Quantization', 'Time Series'], diff='Hard', runtime='numpy',
            lessons=['ieng:11.01'], related=['int8-symmetric-quantize', 'reshaping-arrays', 'tile-and-diff'])

STATEMENT = r"""
Audio models such as WaveNet predict quantised samples, and almost every audio pipeline processes sound in overlapping windows. Combine both steps.

Write `mulaw_frames(x, bits, frame, hop)` that compands and quantises a waveform with **mu-law**, then cuts the integer codes into overlapping frames.

## Input
- `x`: 1-D float array of length `n >= 1` (values are nominally in `[-1, 1]`).
- `bits` (`>= 1`), `frame` (`>= 1`), `hop` (`>= 1`): Python ints.

## Step 1: quantise
With $\mu = 2^{\text{bits}} - 1$:
1. Clip `x` to $[-1, 1]$.
2. Compand: $F(x) = \operatorname{sign}(x)\,\dfrac{\ln(1 + \mu\lvert x\rvert)}{\ln(1 + \mu)}$, which still lies in $[-1, 1]$.
3. Code: $\text{code} = \operatorname{rint}\!\left(\dfrac{F(x) + 1}{2}\,\mu\right)$, an integer in $[0, \mu]$. `rint` rounds halves to the **nearest even** integer, as `np.rint` and `np.round` do.

## Step 2: frame
- The number of frames is $m = 1$ if $n \le \text{frame}$, otherwise $m = 1 + \left\lceil \dfrac{n - \text{frame}}{\text{hop}} \right\rceil$.
- Pad `x` at the end with zeros (**before** quantising, so padding becomes the code of 0.0) to length $(m-1)\cdot\text{hop} + \text{frame}$.
- Frame `i` holds the codes of samples `i*hop` to `i*hop + frame - 1`.

## Output
An `int64` array of shape `(m, frame)`. No Python loop over frames or samples.
"""

STARTER = '''import numpy as np

def mulaw_frames(x, bits, frame, hop):
    """Mu-law quantise x to `bits` bits, then return int64 frames of shape (m, frame) with the given hop."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def mulaw_frames(x, bits, frame, hop):
    x = np.asarray(x, dtype=float)
    n, mu = len(x), 2 ** bits - 1
    m = 1 if n <= frame else 1 + -(-(n - frame) // hop)       # integer ceil division
    x = np.concatenate([x, np.zeros((m - 1) * hop + frame - n)])
    x = np.clip(x, -1.0, 1.0)
    F = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    codes = np.rint((F + 1) / 2 * mu).astype(np.int64)
    # Row i, column j reads sample i*hop + j: one fancy-index gather builds every frame.
    idx = np.arange(m)[:, None] * hop + np.arange(frame)[None, :]
    return codes[idx]
'''

HINTS = [
    'Do the two parts separately. Quantising is purely element-wise (clip, log, round). Framing only needs an index matrix.',
    'Compute the frame count with integer arithmetic (`-(-a // b)` is ceil division), pad with `np.concatenate([x, np.zeros(...)])`, and only then quantise, so the padding becomes the code of 0.0. `np.log1p(t)` computes $\\ln(1+t)$ accurately for small $t$.',
    'Build `idx = np.arange(m)[:, None] * hop + np.arange(frame)` (shape `(m, frame)`) and return `codes[idx]`. Note that for `bits = 1`, the code of 0.0 is `rint(0.5) = 0`.',
]

_r = np.random.default_rng(83)
TESTS = [
    (np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), 8, 4, 2),
    (np.array([0.0, 0.1, -0.1]), 3, 2, 1),
    (np.array([0.3]), 8, 4, 4),
    (np.array([2.0, -3.0, 0.0, 1e-9]), 4, 2, 2),
    (np.array([0.0, 0.0, 1.0, -1.0]), 1, 3, 1),
    (np.array([0.2, -0.7, 0.9, -0.05, 0.6, -0.3, 0.01]), 2, 3, 5),
    (np.sin(np.linspace(0, 6 * np.pi, 50)) * 0.8, 8, 16, 8),
    (_r.uniform(-1.2, 1.2, size=37), 6, 10, 3),
]

EXPLANATION = r"""
## Why it works
**Mu-law companding.** Human hearing and most audio signals spend most of their time at low amplitudes. Applying $F$ before uniform quantisation spends more code levels near zero and fewer near $\pm1$, so 8 bits of mu-law sound about as good as roughly 13 bits of uniform (linear) quantisation. $F$ maps $[-1, 1]$ onto itself, and the affine map $(F+1)/2\cdot\mu$ turns it into $[0, \mu]$, which `rint` snaps to integers.

**Framing.** The sample in frame $i$ at offset $j$ is $i\cdot\text{hop} + j$. Broadcasting a column of frame starts against a row of offsets builds the whole `(m, frame)` index matrix, and one gather produces every frame. Frames overlap when `hop < frame`, and samples are skipped when `hop > frame`. The ceiling in $m$ guarantees the last real sample lands in some frame.

## Complexity
$O(n)$ for quantising, $O(m \cdot \text{frame})$ for the gathered frames. `np.lib.stride_tricks.sliding_window_view(codes, frame)[::hop]` would give a zero-copy view of the same frames.

## Common bugs
- Rounding with `floor(v + 0.5)`. It disagrees with `rint` exactly at halves, and $x = 0$ always sits on a half because $\mu$ is odd.
- Padding **after** quantising with 0. Code 0 means $x = -1$, the loudest negative sample, not silence.
- Forgetting to clip, so inputs beyond $\pm1$ produce codes below 0 or above $\mu$.
- An off-by-one in the frame count, dropping the tail or adding an all-padding frame.

## Interview follow-ups
- How do you decode? (Map codes back to $[-1, 1]$ with $y = 2\,\text{code}/\mu - 1$, then invert the companding: $x = \operatorname{sign}(y)\,\big((1+\mu)^{|y|} - 1\big)/\mu$.)
- Why do spectrogram pipelines multiply each frame by a window (Hann, Hamming) before the FFT? (The frame edges are discontinuities that leak energy across frequencies; a taper suppresses that leakage.)
"""
