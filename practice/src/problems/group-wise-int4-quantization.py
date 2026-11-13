import numpy as np

META = dict(title='Group-wise asymmetric int4 quantization with nibble packing', tags=['Quantization', 'Inference'], diff='Hard',
            runtime='torch', lessons=['ieng:11.03', 'ieng:11.04'],
            related=['per-channel-weight-quantization', 'int8-symmetric-quantize', 'block-wise-fp8-quantization'])

STATEMENT = r"""
Weight-only 4-bit formats (the kind GPTQ and AWQ produce) give every **group** of `g` consecutive weights in a row its own scale and zero point, and store two 4-bit codes per byte. Decode is memory-bound, so reading a quarter of the bytes of fp16 makes it much faster even though the kernel must unpack and dequantize on the fly.

Write `quantize_int4(W, group_size)` in PyTorch.

## Inputs
- `W`: float64 `(out, in)`. `group_size` $g$ divides `in`, and both are even. Group $j$ of row $r$ is `W[r, j*g:(j+1)*g]`.

## Per group (asymmetric, codes 0..15)
1. Range that always contains 0: $\;lo = \min(\min w, 0)$, $\;hi = \max(\max w, 0)$.
2. Scale $s = (hi - lo) / 15$; if $s = 0$ (an all-zero group) use $s = 1$.
3. Zero point $z = \mathrm{clamp}(\mathrm{round}(-lo / s), 0, 15)$, the code that represents 0.
4. Codes $q = \mathrm{clamp}(\mathrm{round}(w / s) + z,\, 0,\, 15)$; dequantized $\hat w = (q - z)\,s$.

Round half to even (`torch.round`) everywhere.

## Packing
Byte $b$ of a row holds codes $2b$ and $2b + 1$: `packed[r, b] = q[r, 2b] | (q[r, 2b+1] << 4)` (even index in the **low** nibble).

## Output
A tuple `(packed, scales, zeros, W_hat, max_err)`:
- `packed`: uint8 `(out, in/2)`; `scales`: float64 `(out, in/g)`; `zeros`: uint8 `(out, in/g)`.
- `W_hat`: float64 `(out, in)`, the dequantized weights (compute it **from `packed`**, by unpacking, so a packing bug shows up here too).
- `max_err`: Python float, $\max |W - \hat W|$.

No Python loops over rows or groups.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def quantize_int4(W, group_size):
    """Return (packed uint8 (out,in/2), scales (out,in/g), zeros uint8 (out,in/g), W_hat (out,in), max_err)."""
    raise NotImplementedError
'''

SOLUTION = '''import torch

def quantize_int4(W, group_size):
    out, n = W.shape
    G = W.view(out, n // group_size, group_size)                    # (out, groups, g)
    lo = G.amin(dim=-1).clamp(max=0)
    hi = G.amax(dim=-1).clamp(min=0)
    s = (hi - lo) / 15
    s = torch.where(s > 0, s, torch.ones_like(s))
    z = torch.clamp(torch.round(-lo / s), 0, 15)
    q = torch.clamp(torch.round(G / s[..., None]) + z[..., None], 0, 15).to(torch.uint8).view(out, n)

    packed = q[:, 0::2] | (q[:, 1::2] << 4)                         # even code -> low nibble

    # unpack exactly as a kernel would, then dequantize group by group
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(out, n // group_size, group_size)
    W_hat = ((codes.to(W.dtype) - z[..., None]) * s[..., None]).view(out, n)
    return packed, s, z.to(torch.uint8), W_hat, (W - W_hat).abs().max().item()
'''

HINTS = [
    'View `W` as `(out, in/g, g)`: per-group min, max, scale and zero point are then reductions over the last axis, shaped `(out, in/g)`.',
    'Compute the codes in float, clamp to [0, 15], then cast to `uint8`. Pack with slicing: `q[:, 0::2] | (q[:, 1::2] << 4)`.',
    'To unpack, `packed & 0xF` gives the even codes and `packed >> 4` the odd ones; `torch.stack(..., dim=-1)` interleaves them back into the original order.',
]

_r = np.random.default_rng(61)
_out = _r.normal(size=(4, 32))
_out[1, 5] = 9.0                                                     # one outlier only hurts its own group
TESTS = [
    (np.array([[0.0, 1.5, -1.5, 0.3]]), 4),
    (np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, 0.0, 0.0]]), 2),
    (_r.normal(size=(3, 16)), 8),
    (_out, 8),
    (_out, 32),
    (np.zeros((2, 8)), 4),
    (np.array([[0.1, 0.1, 0.1, 0.1, -3.0, 7.5, 0.0, 2.25]]), 4),
    (_r.uniform(0, 1, size=(5, 64)), 16),
]

EXPLANATION = r"""
## Why it works
An affine map $w \approx (q - z)\,s$ spends all 16 codes on the actual range of the group. Forcing the range to include 0 and rounding the zero point to an integer code makes 0 exactly representable, which matters for padded and pruned weights. The rounding error is at most $s/2$ inside the range, and the range is per group, so an outlier only stretches the scale of its own $g$ neighbours (compare tests 4 and 5: the same matrix with groups of 8 vs whole rows).

## Memory
Per weight: 4 bits plus the group's scale and zero point. With fp16 scales, 4-bit zeros and $g = 128$, the overhead is $(16 + 4) / 128 \approx 0.16$ bits, so about 4.16 bits per weight, roughly $3.8\times$ smaller than fp16.

## Common bugs
- Packing the even code into the **high** nibble while the kernel expects the low one (or vice versa): every pair of weights is swapped. Unpacking from `packed` in your own dequantization catches it.
- Shifting a signed or float tensor: build codes as `uint8` first, otherwise `<< 4` and `|` either fail or sign-extend.
- Leaving out the "range contains 0" step: a group of all-positive weights then has a zero point outside $[0, 15]$ after clamping and 0 is no longer exact.

## Interview follow-ups
- Why do GPTQ and AWQ need calibration data when this method does not? (They choose the rounding, or per-channel scaling, to minimise the layer's **output** error on real activations, not the weight error.)
- Where is dequantization done in a fast int4 kernel? (In registers, right before the multiply-accumulate, so only 4-bit data crosses the memory bus.)
"""
