import numpy as np

META = dict(title='Symmetric int8 quantization', tags=['Quantization', 'Inference'], diff='Easy', runtime='numpy',
            lessons=['aieng:07.01', 'ieng:11.01'], related=['layer-norm'])

STATEMENT = r"""
Per-tensor symmetric quantization maps floats to int8 with a single scale:

$$s = \frac{\max_i |x_i|}{127}, \qquad q_i = \mathrm{clip}\bigl(\mathrm{round}(x_i / s),\, -127,\, 127\bigr), \qquad \hat x_i = s\, q_i$$

Write `quantize_int8(x)` that returns `(q, scale, max_err)`:

- `q`: `int8` array, same shape as `x`.
- `scale`: Python float. If `x` is all zeros, use `scale = 1.0`.
- `max_err`: the largest absolute reconstruction error $\max_i |x_i - \hat x_i|$, as a float.

Round half to even, which is what `np.round` does.
"""

STARTER = '''import numpy as np

def quantize_int8(x):
    """Return (q int8 array, scale float, max reconstruction error float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def quantize_int8(x):
    x = np.asarray(x, dtype=float)
    amax = float(np.abs(x).max())
    scale = amax / 127 if amax > 0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale, float(np.abs(x - q * scale).max())
'''

HINTS = [
    'The scale maps the largest magnitude to exactly 127. Guard the all-zero case before dividing.',
    'Round, clip to [-127, 127], **then** cast with `.astype(np.int8)`. Casting first would wrap around instead of saturating.',
    'Dequantize with `q * scale` to measure the error. It should never exceed `scale / 2`.',
]

_r = np.random.default_rng(10)
TESTS = [
    (np.array([0.0, 0.5, -1.0, 1.0]),),
    (np.zeros(5),),
    (np.array([127.0, -127.0, 63.5, 0.5]),),
    (_r.normal(size=(4, 8)),),
    (np.append(_r.normal(size=63) * 0.1, 50.0),),
    (np.array([[1e-3, -2e-3], [3e-3, 0.0]]),),
]

EXPLANATION = r"""
## The error bound
Each value is rounded to the nearest multiple of $s$, so $|x - \hat x| \le s/2$. Halving the error means halving $s$, which in turn means a smaller max magnitude.

## Outliers are the enemy
Test 5 has one value of 50 among values around ±0.1. The scale becomes 50/127 ≈ 0.39, so almost every small value rounds to 0 or ±1 and most of their information is gone. This is the core problem in LLM quantization: activation outlier channels. Remedies:
- **Finer granularity**: per-channel or per-group scales (e.g. groups of 128), so one outlier only affects its own group.
- **Outlier handling**: keep outlier channels in fp16 (LLM.int8()) or migrate the scale into the weights (SmoothQuant).

## Why −127, not −128
Using a symmetric range keeps zero exactly representable and makes $q \mapsto -q$ safe. The lost code (−128) is a negligible cost.

## Interview follow-ups
- Symmetric vs asymmetric (zero-point) quantization: when does a zero-point help? (Post-ReLU activations, which are all ≥ 0.)
- Why is weight-only int4 with fp16 activations popular for single-user LLM decoding? (Decoding is memory-bandwidth-bound.)
"""
