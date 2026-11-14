import numpy as np

META = dict(title='Dequantize MXFP4 weights', tags=['Quantization'], diff='Medium', runtime='numpy',
            lessons=['ieng:11.01'],
            related=['block-wise-fp8-quantization', 'group-wise-int4-quantization', 'int8-symmetric-quantize'])

STATEMENT = r"""
MXFP4 (an OCP microscaling format) stores weights as 4-bit floats in blocks of 32, with one shared power-of-two scale per block. gpt-oss ships its MoE expert weights this way, which is how a 120B-parameter model fits on one 80 GB GPU.

Write `mxfp4_dequant(blocks, scales)` that returns the float weight matrix.

## Encoding
- `blocks`: `uint8` array `(R, G, 16)`: row $r$ has $G$ groups of 32 values, packed two per byte. Byte $j$ holds element $2j$ in its **low** nibble and element $2j+1$ in its **high** nibble.
- `scales`: `uint8` array `(R, G)`, an E8M0 exponent: the block's scale is $2^{e - 127}$.
- Each 4-bit code is FP4 E2M1: bit 3 is the sign, bits 0-2 index the magnitudes
  `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`.
- Value $=$ sign $\times$ magnitude $\times$ scale.

## Output
Float64 array `(R, 32·G)`; group $g$ of row $r$ fills columns $32g .. 32g+31$ in element order.
"""

STARTER = '''import numpy as np

def mxfp4_dequant(blocks, scales):
    """blocks: uint8 (R, G, 16); scales: uint8 (R, G). Return float (R, 32*G)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def mxfp4_dequant(blocks, scales):
    b = np.asarray(blocks, dtype=np.uint8)
    R, G, _ = b.shape
    codes = np.empty((R, G, 32), dtype=np.uint8)
    codes[..., 0::2] = b & 0x0F                       # low nibble first
    codes[..., 1::2] = b >> 4
    vals = _MAG[codes & 0x7] * np.where(codes & 0x8, -1.0, 1.0)
    scale = np.ldexp(1.0, np.asarray(scales, dtype=np.int64) - 127)
    return (vals * scale[..., None]).reshape(R, G * 32)
'''

HINTS = [
    'Unpack with bit operations: `b & 0x0F` gives the even elements, `b >> 4` the odd ones. Interleave them into a `(R, G, 32)` array.',
    'Split each code into sign (`code & 8`) and magnitude index (`code & 7`), look the magnitude up in a table, and multiply by $2^{e-127}$ broadcast over the 32 elements of the block.',
]

_r = np.random.default_rng(4)
TESTS = [
    (np.array([[[0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=np.uint8), np.array([[127]], dtype=np.uint8)),
    (np.array([[[0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=np.uint8), np.array([[120]], dtype=np.uint8)),
    (_r.integers(0, 256, size=(2, 1, 16)).astype(np.uint8), np.array([[127], [130]], dtype=np.uint8)),
    (_r.integers(0, 256, size=(3, 2, 16)).astype(np.uint8), _r.integers(118, 134, size=(3, 2)).astype(np.uint8)),
    (np.full((1, 2, 16), 0x88, dtype=np.uint8), np.array([[127, 1]], dtype=np.uint8)),
    (_r.integers(0, 256, size=(4, 3, 16)).astype(np.uint8), _r.integers(100, 140, size=(4, 3)).astype(np.uint8)),
]

EXPLANATION = r"""
## Why this format
Four bits per value plus 8 bits per 32 values is 4.25 bits per weight. The shared power-of-two scale adapts each block to its own range (cheap to apply, it only shifts exponents), and the E2M1 element format spends its few codes on a log-like spread $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$, which suits bell-shaped weight distributions better than evenly spaced int4 levels.

## Details
- Code `0b1000` is negative zero; it dequantizes to `-0.0`, which compares equal to 0.
- The E8M0 scale has no mantissa and no sign: it can only be a power of two. Quantizers pick the block exponent from the block's largest magnitude so that it maps near 6, the largest element value.
- Hardware with native FP4 support (e.g. recent NVIDIA tensor cores) consumes this layout directly; elsewhere, kernels dequantize tiles into bf16 on the fly inside the matmul.

## Complexity
$O(RG \cdot 32)$, one table lookup per element.

## Common bugs
- Swapping nibble order (high nibble first), which permutes adjacent pairs.
- Treating the code as two's-complement int4 instead of sign-magnitude.
- Computing $2^{e}$ instead of $2^{e-127}$, or overflowing with an integer power.

## Interview follow-ups
- Write the quantizer: given a block of 32 floats, choose the exponent and round each element to the nearest E2M1 value.
- How much memory do 116B parameters take in MXFP4, compared with bf16?
"""
