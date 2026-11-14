import numpy as np

META = dict(title='NF4: map weights to the nearest normal-float level', tags=['Quantization'], diff='Medium', runtime='numpy')

STATEMENT = r"""
NF4 (from QLoRA) uses 16 fixed levels in $[-1, 1]$ spaced like quantiles of a normal distribution, so normally distributed weights use every code about equally. The levels are given as `NF4` in the starter.

Write `nf4(w, block)` for a 1-D weight array:

1. Split `w` into consecutive blocks of `block` values (the last may be shorter). Each block's scale is its absmax (1 if the block is all zeros).
2. Normalise by the scale and map each value to the **nearest** level (ties to the lower index). Dequantise as level × scale.
3. For comparison, quantise the same blocks with a uniform symmetric 4-bit grid: levels $k/7$ for $k = -7..7$, also scaled by the block absmax, nearest level.

Return `{"codes": int array, "scales": array, "w_hat": array, "mse_nf4": float, "mse_int4": float}`.
"""

_NF4 = '''NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
       0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
       0.7229568362236023, 1.0]
'''

STARTER = '''import numpy as np

''' + _NF4 + '''
def nf4(w, block):
    """Block-wise NF4 quantisation with a uniform int4 baseline."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

''' + _NF4 + '''
def _nearest(x, levels):
    return np.abs(x[:, None] - levels[None, :]).argmin(1)

def nf4(w, block):
    w = np.asarray(w, float)
    L = np.asarray(NF4)
    U = np.arange(-7, 8) / 7
    codes, scales, wh, wu = [], [], [], []
    for i in range(0, len(w), block):
        b = w[i:i + block]
        s = np.abs(b).max()
        s = s if s > 0 else 1.0
        c = _nearest(b / s, L)
        codes.append(c); scales.append(s)
        wh.append(L[c] * s); wu.append(U[_nearest(b / s, U)] * s)
    wh, wu = np.concatenate(wh), np.concatenate(wu)
    return {"codes": np.concatenate(codes), "scales": np.array(scales), "w_hat": wh,
            "mse_nf4": float(((w - wh) ** 2).mean()), "mse_int4": float(((w - wu) ** 2).mean())}
'''

HINTS = [
    'argmin over |x - level| returns the first minimum, which gives the lower-index tie-break for free.',
    'Normalise each block by its absmax so every block maps onto [-1, 1] before choosing levels.',
    'NF4 has an exact zero and is asymmetric: 8 positive levels and 7 negative ones plus -1.',
]

_r = np.random.default_rng(7)
TESTS = [
    ([0.0, 0.5, -1.0, 0.25], 4),
    (_r.normal(size=256), 64),
    (_r.normal(size=200), 64),
    (_r.uniform(-1, 1, size=128), 64),
    (np.zeros(10), 4),
    (_r.standard_t(3, size=256), 64),
]

EXPLANATION = r"""
## Why normal-spaced levels
Trained weights are roughly normally distributed, so a uniform grid wastes codes in the tails and is too coarse near zero, where most weights are. Placing levels at normal quantiles makes each code about equally likely, which minimises error for normal data. On the uniform test case the advantage disappears, as it should.

## Block-wise scales
A block of 64 weights shares one absmax scale, so one outlier only damages its own block. QLoRA then quantises the scales themselves to 8 bits ("double quantisation"), bringing the overhead to about 0.127 bits per weight.

## Where it is used
NF4 is a storage format: weights are dequantised to bf16 just before each matmul. That is ideal for QLoRA fine-tuning, where memory, not matmul speed, is the constraint.

## Interview follow-ups
- Why does NF4 need an exact zero level?
"""
