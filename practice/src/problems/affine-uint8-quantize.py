import numpy as np

META = dict(title='Affine quantisation with a zero point', tags=['Quantization'], diff='Easy', runtime='numpy',
            related=['int8-symmetric-quantize'])

STATEMENT = r"""
Write `affine_quantize(x, bits)` for unsigned codes $0 .. 2^{b}-1$.

1. Extend the range to include zero: $lo = \min(\min x, 0)$, $hi = \max(\max x, 0)$.
2. $scale = (hi - lo) / (2^b - 1)$; if $hi = lo$ use $scale = 1$.
3. $zp = \mathrm{clip}(\mathrm{round}(-lo / scale), 0, 2^b - 1)$ as an integer.
4. $q = \mathrm{clip}(\mathrm{round}(x / scale) + zp, 0, 2^b - 1)$; dequantised $\hat x = (q - zp) \cdot scale$.

Use `np.round` (round half to even). Also quantise symmetrically for comparison: $s_{sym} = \max|x| / (2^{b-1} - 1)$ (1 if zero), $q_{sym} = \mathrm{clip}(\mathrm{round}(x / s_{sym}), -(2^{b-1}-1), 2^{b-1}-1)$.

Return `{"q": int array, "scale", "zp", "max_err": max|x - x̂|, "sym_max_err"}`.
"""

STARTER = '''import numpy as np

def affine_quantize(x, bits):
    """Asymmetric quantisation with a zero point, plus a symmetric baseline."""
    x = np.asarray(x, float)
    s = np.abs(x).max() / (2 ** (bits - 1) - 1)
    q = np.round(x / s)
    return {"q": q.astype(int), "scale": float(s), "zp": 0, "max_err": float(np.abs(x - q * s).max()), "sym_max_err": 0.0}
'''

SOLUTION = '''import numpy as np

def affine_quantize(x, bits):
    x = np.asarray(x, float)
    qmax = 2 ** bits - 1
    lo, hi = min(x.min(), 0.0), max(x.max(), 0.0)
    scale = (hi - lo) / qmax if hi > lo else 1.0
    zp = int(np.clip(np.round(-lo / scale), 0, qmax))
    q = np.clip(np.round(x / scale) + zp, 0, qmax).astype(int)
    xh = (q - zp) * scale
    m = 2 ** (bits - 1) - 1
    amax = np.abs(x).max()
    ss = amax / m if amax > 0 else 1.0
    qs = np.clip(np.round(x / ss), -m, m)
    return {"q": q, "scale": float(scale), "zp": zp, "max_err": float(np.abs(x - xh).max()),
            "sym_max_err": float(np.abs(x - qs * ss).max())}
'''

HINTS = [
    'Including zero in the range guarantees that 0.0 is represented exactly, which matters for padding and ReLU outputs.',
    'The zero point is the code that represents real 0.0.',
    'For one-sided data such as ReLU activations, symmetric quantisation wastes half its codes on negative values.',
]

_r = np.random.default_rng(2)
TESTS = [
    ([0.0, 0.5, 1.0, 2.0], 8),
    (np.maximum(_r.normal(size=100), 0), 8),
    (_r.normal(size=100), 8),
    (_r.normal(2.0, 0.1, size=50), 4),
    ([0.0, 0.0], 8),
    ([-3.0, -1.0, -0.5], 4),
]

EXPLANATION = r"""
## Affine versus symmetric
Symmetric quantisation maps $[-\max|x|, \max|x|]$ onto signed codes and keeps zero at code 0, which makes integer matmuls simple. Affine quantisation maps $[\min, \max]$ onto all codes with a zero point, so skewed data (ReLU outputs, softmax probabilities) uses the full code range. For one-sided data it halves the step size.

## The cost
Integer matmuls with zero points need correction terms ($\sum q_x \cdot zp_w$ and similar), so weights are usually symmetric and activations sometimes affine. For storage-only weight quantisation (dequantise before the matmul) affine is common, as in group-wise int4.

## Interview follow-ups
- Why must the range include zero, even if every value is positive?
"""
