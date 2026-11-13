import numpy as np

META = dict(title='Block-wise FP8 (E4M3) weight quantization', tags=['Quantization', 'Inference'], diff='Hard', runtime='torch',
            lessons=['ieng:11.02', 'ieng:11.01'], tol=[1e-6, 1e-7],
            related=['group-wise-int4-quantization', 'per-channel-weight-quantization', 'int8-symmetric-quantize'])

STATEMENT = r"""
FP8 E4M3 has 4 exponent bits and 3 mantissa bits: it covers $\pm 448$ with about two significant digits, finer near zero and coarser near the top. Recent models store weights in FP8 with one scale per square **block** (for example $128 \times 128$), so each block is stretched to use the format's range.

Write `fp8_blockwise(W, block)` in PyTorch.

## Inputs
- `W`: float32 `(M, N)`. `block`: int $b \ge 1$. $M$ and $N$ need **not** be multiples of $b$: edge blocks are smaller.
- Block $(i, j)$ is `W[i*b:(i+1)*b, j*b:(j+1)*b]`; there are $\lceil M/b \rceil \times \lceil N/b \rceil$ of them.

## Per block (all arithmetic in float32)
1. $s = \max |w| / 448$; use $s = 1$ for an all-zero block.
2. Divide the block by $s$ and cast to `torch.float8_e4m3fn` (PyTorch rounds to the nearest representable value, ties to even). Dividing first guarantees every value lies in $[-448, 448]$.
3. Dequantize: cast back to float32 and multiply by $s$.

## Output
A tuple `(W_hat, scales)`: `W_hat` float32 `(M, N)` and `scales` float32 `(ceil(M/b), ceil(N/b))`. NumPy has no FP8 type, so return the dequantized weights rather than the FP8 tensor. Use tensor ops (padding is fine), no loops over blocks.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests. It needs PyTorch 2.1 or newer for the FP8 dtype.
"""

STARTER = '''import torch

def fp8_blockwise(W, block):
    """W float32 (M, N). Return (W_hat float32 (M, N), scales float32 (ceil(M/b), ceil(N/b)))."""
    s = W.abs().max() / 448                       # one scale for everything: wrong granularity
    return (W / s).to(torch.float8_e4m3fn).to(torch.float32) * s, s.reshape(1, 1)
'''

SOLUTION = '''import torch
import torch.nn.functional as F

def fp8_blockwise(W, block):
    M, N = W.shape
    gm, gn = -(-M // block), -(-N // block)                          # ceil division: number of blocks
    # zero-pad to whole blocks; padding never changes a block's max |w|
    P = F.pad(W, (0, gn * block - N, 0, gm * block - M))
    blocks = P.view(gm, block, gn, block)                            # (block row, r, block col, c)
    amax = blocks.abs().amax(dim=(1, 3))                             # (gm, gn)
    scales = torch.where(amax > 0, amax / 448, torch.ones_like(amax))
    s = scales[:, None, :, None]                                     # broadcast over each block's entries
    q = (blocks / s).to(torch.float8_e4m3fn)                         # values now lie in [-448, 448]
    W_hat = (q.to(torch.float32) * s).view(gm * block, gn * block)[:M, :N]
    return W_hat, scales
'''

HINTS = [
    'Pad `W` with zeros up to whole blocks (`F.pad`), then `view(gm, b, gn, b)`: axes 1 and 3 run inside a block.',
    'Reduce `abs()` over axes `(1, 3)` for the per-block max. Broadcast the `(gm, gn)` scales back as `scales[:, None, :, None]`.',
    'Divide, cast to `torch.float8_e4m3fn`, cast back to float32 and multiply. Crop the padding off with `[:M, :N]` at the end.',
]

_r = np.random.default_rng(71)
_f = lambda a: np.asarray(a, dtype=np.float32)
_w = _f(_r.normal(size=(6, 8)))
_w[0, 0] = 60.0                                                      # an outlier inflates only its own block
TESTS = [
    (_f([[448.0, 1.0], [-0.3, 224.0]]), 2),
    (_f([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]), 2),
    (_w, 4),
    (_w, 8),
    (_f(_r.normal(size=(5, 7)) * 0.01), 3),
    (np.zeros((3, 4), dtype=np.float32), 2),
    (_f(_r.standard_t(2, size=(16, 16))), 4),
    (_f(_r.normal(size=(9, 5))), 1),
]

EXPLANATION = r"""
## Why it works
Scaling a block so its largest magnitude maps to 448 uses the whole range of E4M3; the floating-point format then spends its precision **relatively**: every value keeps about 3 bits of mantissa (relative error up to $2^{-4}$) until it falls into the subnormal range below $2^{-6}$ of the scaled block. Small blocks keep a single outlier (test 3) from pushing the rest of the matrix toward the subnormals.

## Cost
One scale per $b^2$ weights: for $b = 128$ that is 0.002 extra bits per weight. On Hopper-class GPUs FP8 tensor cores run about twice as fast as BF16, and the per-block scales are applied in the GEMM's epilogue or between partial sums along $K$.

## Common bugs
- Casting to FP8 **before** dividing by the scale: anything above 448 becomes `nan` in `float8_e4m3fn`, which has no infinity.
- Reshaping `(M, N)` straight to `(gm, gn, b, b)`: that groups contiguous memory, not 2-D tiles. The tile view is `(gm, b, gn, b)`.
- Forgetting the edge blocks when $M$ or $N$ is not a multiple of $b$; padding with zeros is safe because it never raises a block's max.

## Interview follow-ups
- E4M3 vs E5M2: which for weights and activations, which for gradients? (E4M3 has more precision for forward tensors; E5M2's wider range suits gradients.)
- Why do FP8 training recipes use $1 \times 128$ tiles for activations but $128 \times 128$ blocks for weights? (Activation outliers are per token, so a per-token tile isolates them; weights are smoother.)
"""
