import numpy as np

META = dict(title='Capstone: compare weight quantisation schemes on a small MLP', tags=['Quantization'], diff='Hard', runtime='numpy',
            related=['affine-uint8-quantize', 'int4-with-outlier-columns', 'nf4-quantize', 'gptq-column-quantize'])

STATEMENT = r"""
Write `compare_schemes(X, W1, b1, W2, b2)` for the MLP $f(X) = \mathrm{relu}(XW_1^\top + b_1)\,W_2^\top + b_2$ (weights stored `(out, in)`). Quantise **both weight matrices** (biases stay exact) with each scheme, run the MLP, and report the relative output error $\|f - \hat f\|_F / \|f\|_F$ and the weight storage in bits (biases excluded):

- `"int8_tensor"`: symmetric int8, one scale for the whole matrix ($\delta = \max|W|/127$). Bits: 8 per weight + 16 per scale.
- `"int8_channel"`: symmetric int8 with one scale per output row. Bits: 8 per weight + 16 per row.
- `"int4_group"`: affine 4-bit per row in groups of 32 input channels ($lo = \min(\min,0)$, $hi = \max(\max,0)$, $scale = (hi-lo)/15$ (1 if equal), $zp = \mathrm{clip}(\mathrm{round}(-lo/scale),0,15)$). Bits: 4 per weight + 32 per group (fp16 scale and zero point).
- `"nf4"`: NF4 levels (given in the starter) on blocks of 64 consecutive weights of the row-major flattened matrix, absmax scale, nearest level with ties to the lower index. Bits: 4 per weight + 16 per block.

Rounding is `np.round`; every scale of 0 is replaced by 1. Return a dict from scheme name to `[rel_error, bits]`, plus `"fp16_bits"`: 16 per weight.
"""

_NF4 = '''NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
       0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
       0.7229568362236023, 1.0]
'''

STARTER = '''import numpy as np

''' + _NF4 + '''
def compare_schemes(X, W1, b1, W2, b2):
    """Relative output error and storage bits for four weight formats."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

''' + _NF4 + '''
def _safe(s):
    return np.where(s > 0, s, 1.0)

def _int8_tensor(W):
    d = _safe(np.array(np.abs(W).max() / 127))
    return np.clip(np.round(W / d), -127, 127) * d, 8 * W.size + 16

def _int8_channel(W):
    d = _safe(np.abs(W).max(1, keepdims=True) / 127)
    return np.clip(np.round(W / d), -127, 127) * d, 8 * W.size + 16 * W.shape[0]

def _int4_group(W, g=32):
    out, groups = np.empty_like(W), 0
    for r in range(W.shape[0]):
        for c in range(0, W.shape[1], g):
            w = W[r, c:c + g]
            lo, hi = min(w.min(), 0.0), max(w.max(), 0.0)
            s = (hi - lo) / 15 if hi > lo else 1.0
            zp = np.clip(np.round(-lo / s), 0, 15)
            out[r, c:c + g] = (np.clip(np.round(w / s) + zp, 0, 15) - zp) * s
            groups += 1
    return out, 4 * W.size + 32 * groups

def _nf4(W, block=64):
    L, flat = np.asarray(NF4), W.ravel()
    out = np.empty_like(flat)
    for i in range(0, len(flat), block):
        b = flat[i:i + block]
        s = float(_safe(np.array(np.abs(b).max())))
        out[i:i + block] = L[np.abs((b / s)[:, None] - L[None, :]).argmin(1)] * s
    return out.reshape(W.shape), 4 * W.size + 16 * math.ceil(W.size / block)

def compare_schemes(X, W1, b1, W2, b2):
    X, W1, b1, W2, b2 = (np.asarray(a, float) for a in (X, W1, b1, W2, b2))
    f = lambda A, B: np.maximum(X @ A.T + b1, 0) @ B.T + b2
    ref = f(W1, W2)
    out = {}
    for name, q in [("int8_tensor", _int8_tensor), ("int8_channel", _int8_channel), ("int4_group", _int4_group), ("nf4", _nf4)]:
        (A, ba), (B, bb) = q(W1), q(W2)
        out[name] = [float(np.linalg.norm(ref - f(A, B)) / np.linalg.norm(ref)), ba + bb]
    out["fp16_bits"] = 16 * (W1.size + W2.size)
    return out
'''

HINTS = [
    'Write each scheme as a function returning (dequantised matrix, bits) so the comparison loop is uniform.',
    'Per-channel scales fix the case where one row has much larger weights than the others.',
    'Count metadata bits honestly: small groups improve accuracy but cost storage.',
]

def _mlp(seed, n, d, h, o, row_outlier):
    r = np.random.default_rng(seed)
    W1 = r.normal(size=(h, d)) / np.sqrt(d)
    if row_outlier:
        W1[0] *= 30
    return (r.normal(size=(n, d)), W1, 0.1 * r.normal(size=h), r.normal(size=(o, h)) / np.sqrt(h), 0.1 * r.normal(size=o))

TESTS = [_mlp(1, 32, 64, 128, 16, False), _mlp(2, 32, 64, 128, 16, True), _mlp(3, 16, 96, 64, 8, False),
         _mlp(4, 8, 32, 32, 4, True), _mlp(5, 20, 40, 50, 10, False)]

EXPLANATION = r"""
## What the comparison shows
- Per-tensor int8 is fine until one row has larger weights; then every other row loses precision. Per-channel scales fix that for 16 extra bits per row.
- The 4-bit formats cost about a quarter to a third of fp16 storage. Here NF4 is slightly less accurate than int4 groups because its blocks are twice as large (64 versus 32) and it has no zero point, but it also uses fewer bits. Compare formats at equal bits per weight, not equal names.
- Blocks that run across rows are fragile: in the outlier cases one large row shares NF4 blocks with its neighbour and destroys it, while per-row groups contain the damage.
- Errors compound across layers: the relative error at the output is larger than either layer's alone.

## Measuring on a real model
The same loop scales to a language model: quantise every linear layer, then compare perplexity on held-out text and accuracy on a few tasks, not just layer errors. Libraries such as bitsandbytes (NF4, int8), GPTQ and AWQ implementations provide the kernels; the evaluation is what tells you which one is acceptable.

## Interview follow-ups
- Which layers of a transformer would you keep in higher precision first, and why?
"""
