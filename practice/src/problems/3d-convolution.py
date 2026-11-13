import numpy as np

META = dict(title='Multi-channel 3-D convolution on a 3-D grid', tags=['GPU Programming', 'Convolutional Networks'],
            diff='Hard', runtime='cuda', tol=[1e-4, 1e-5], lessons=['ieng:13.01', 'ieng:13.03'],
            related=['2d-convolution-2', '1d-convolution', '2d-convolution'])

STATEMENT = r"""
Video and medical-imaging models use 3-D convolutions: the filter slides over depth (time or slices), height and width. Implement the forward pass of a multi-channel 3-D convolution layer (cross-correlation, as in deep-learning frameworks) with zero padding and stride:

$$y_{o,d,h,w} = b_o + \sum_{i=0}^{C_{in}-1} \sum_{a=0}^{K_D-1} \sum_{e=0}^{K_H-1} \sum_{f=0}^{K_W-1} W_{o,i,a,e,f}\; x_{i,\ sd + a - p,\ sh + e - p,\ sw + f - p}$$

where $x$ is zero outside its bounds, $s$ is the stride and $p$ the padding (the same on every side and every axis). The output sizes are

$$D_o = \left\lfloor \frac{D + 2p - K_D}{s} \right\rfloor + 1, \quad H_o = \left\lfloor \frac{H + 2p - K_H}{s} \right\rfloor + 1, \quad W_o = \left\lfloor \frac{W + 2p - K_W}{s} \right\rfloor + 1$$

Write `conv3d(x, weight, bias, stride, padding)`.

## The launch: a 3-D grid with a folded axis
The output has **four** indices $(o, d, h, w)$ but a CUDA grid has three axes. Use:
- block `(8, 8, 4)` (256 threads), `w, h, z = cuda.grid(3)` with x → $w$ and y → $h$;
- fold output channel and depth into z: `z = o * D_o + d`, so `o = z // D_o`, `d = z % D_o`;
- grid $(\lceil W_o/8 \rceil, \lceil H_o/8 \rceil, \lceil C_{out} D_o / 4 \rceil)$.

One thread computes one output value, accumulates in a local `float32`, and adds the bias once. Padding is implicit: skip taps that fall outside the input.

## Inputs and output
- `x`: `float32` `(C_in, D, H, W)`; `weight`: `float32` `(C_out, C_in, K_D, K_H, K_W)`; `bias`: `float32` `(C_out,)`.
- `stride` $\ge 1$ and `padding` $\ge 0$ are Python ints; every output size is at least 1.
- Return `y`: `float32` `(C_out, D_o, H_o, W_o)`. Compared with relative tolerance $10^{-4}$, absolute $10^{-5}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda, float32

BLOCK = (8, 8, 4)

def conv3d(x, weight, bias, stride, padding):
    """x: (C_in, D, H, W), weight: (C_out, C_in, KD, KH, KW), bias: (C_out,), all float32.
    Returns float32 (C_out, Do, Ho, Wo)."""
    C_in, D, H, W = x.shape
    C_out, _, KD, KH, KW = weight.shape
    Do, Ho, Wo = D - KD + 1, H - KH + 1, W - KW + 1   # TODO: stride and padding
    y = np.zeros((C_out, Do, Ho, Wo), dtype=np.float32)
    # TODO: launch a kernel on a 3-D grid, folding (C_out, Do) into the z axis
    return y
'''

SOLUTION = '''import numpy as np
from numba import cuda, float32

BLOCK = (8, 8, 4)

def conv3d(x, weight, bias, stride, padding):
    C_in, D, H, W = x.shape
    C_out, _, KD, KH, KW = weight.shape
    Do = (D + 2 * padding - KD) // stride + 1
    Ho = (H + 2 * padding - KH) // stride + 1
    Wo = (W + 2 * padding - KW) // stride + 1
    d_x = cuda.to_device(np.ascontiguousarray(x))
    d_w = cuda.to_device(np.ascontiguousarray(weight))
    d_b = cuda.to_device(np.ascontiguousarray(bias))
    d_y = cuda.device_array((C_out, Do, Ho, Wo), dtype=np.float32)
    grid = ((Wo + BLOCK[0] - 1) // BLOCK[0],
            (Ho + BLOCK[1] - 1) // BLOCK[1],
            (C_out * Do + BLOCK[2] - 1) // BLOCK[2])      # z covers (channel, depth) pairs
    conv3d_kernel[grid, BLOCK](d_x, d_w, d_b, stride, padding, d_y)
    return d_y.copy_to_host()

@cuda.jit
def conv3d_kernel(x, wt, b, s, p, y):
    ow, oh, z = cuda.grid(3)
    C_out, Do, Ho, Wo = y.shape
    if ow >= Wo or oh >= Ho or z >= C_out * Do:
        return
    o, od = z // Do, z % Do                          # unfold the z axis
    C_in, D, H, W = x.shape
    KD, KH, KW = wt.shape[2], wt.shape[3], wt.shape[4]
    d0, h0, w0 = od * s - p, oh * s - p, ow * s - p  # input corner of this output's window

    acc = float32(0.0)
    for i in range(C_in):
        for a in range(KD):
            d = d0 + a
            if d < 0 or d >= D:
                continue                             # whole depth slice is padding
            for e in range(KH):
                h = h0 + e
                if h < 0 or h >= H:
                    continue
                for f in range(KW):
                    w = w0 + f
                    if 0 <= w < W:
                        acc += wt[o, i, a, e, f] * x[i, d, h, w]
    y[o, od, oh, ow] = acc + b[o]
'''

HINTS = [
    'Output sizes first: `(D + 2*padding - KD) // stride + 1`, and likewise for H and W. Then the grid: x over `Wo`, y over `Ho`, z over `C_out * Do`, each rounded up to the block size.',
    'In the kernel, unfold `o, od = z // Do, z % Do`. The window of output `(od, oh, ow)` starts at input `(od*s - p, oh*s - p, ow*s - p)`; tap `(a, e, f)` adds that offset.',
    'Loop `i, a, e, f`, skip any tap whose input coordinate is outside `[0, D)`, `[0, H)` or `[0, W)`, accumulate in a local `float32`, and write `acc + b[o]` once. Bounds-check all three output coordinates (including `z < C_out * Do`) before doing any work.',
]

_r = np.random.default_rng(53)
_f = lambda *s: _r.standard_normal(s).astype(np.float32)
TESTS = [
    (np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2), np.ones((1, 1, 2, 2, 2), np.float32),
     np.array([0.5], np.float32), 1, 0),                                         # one output: 28.5
    (np.ones((1, 3, 3, 3), np.float32), np.ones((2, 1, 3, 3, 3), np.float32),
     np.array([0.0, -1.0], np.float32), 1, 1),                                   # 'same' padding: corner counts
    (_f(2, 4, 5, 6), _f(3, 2, 1, 1, 1), _f(3), 1, 0),                            # 1x1x1: channel mixing
    (_f(1, 6, 6, 6), _f(1, 1, 3, 3, 3), _f(1), 2, 1),                            # stride 2 with padding
    (_f(3, 5, 4, 7), _f(2, 3, 2, 3, 2), _f(2), 1, 0),                            # non-cubic kernel
    (_f(2, 7, 9, 10), _f(4, 2, 3, 3, 3), _f(4), 1, 1),                           # several blocks on every axis
    (_f(1, 8, 8, 8), _f(2, 1, 3, 3, 3), _f(2), 3, 2),                            # stride 3, padding 2
]

EXPLANATION = r"""
## Why it works
Each thread owns one output coordinate and computes the full sum for it, so there are no write conflicts and no synchronisation. Folding $(o, d)$ into one axis is a bijection between $z \in [0, C_{out} D_o)$ and the pairs, so the grid covers the output exactly once; the bounds check discards the overhang on every axis. The input index of tap $(a, e, f)$ is the window corner $(sd - p, sh - p, sw - p)$ plus the tap offset, and skipping out-of-range taps is the same as multiplying by zero padding.

## Cost
Each thread does $2 C_{in} K_D K_H K_W$ FLOPs and the same number of loads. For a $3^3$ filter that is 54 loads per input channel per output, so this direct kernel is load-bound. Two standard improvements:
- **Shared-memory tiling** across $(h, w)$ and depth, as in the 2-D problem, with the halo now growing in three dimensions ($(8 + K - 1)^2 (4 + K - 1)$ pixels per channel).
- **Implicit GEMM**: treat the convolution as a matrix product of shape $(C_{out}) \times (C_{in} K^3)$ by $(C_{in} K^3) \times (D_o H_o W_o)$ and reuse a tiled matmul (or tensor cores). This is how cuDNN implements most 3-D convolutions.

## Grid limits
`gridDim.y` and `gridDim.z` are limited to 65,535 and `blockDim.z` to 64, while `gridDim.x` can reach $2^{31} - 1$. Folding is the usual way around the 3-axis limit, and when an axis can be huge (batch × channels × depth), putting it on x and folding the small ones into y and z keeps you within bounds.

## Common bugs
- Output sizes without the floor division, or with padding applied once instead of on both sides ($2p$).
- Using the output coordinate directly as the input coordinate (forgetting $s \cdot d - p$).
- Mapping x to depth: adjacent threads should differ in the fastest-varying (last) axis $w$, so that their loads of `x[i, d, h, w]` coalesce.
- Adding the bias inside the channel loop ($C_{in}$ times).

## Interview follow-ups
- Derive the arithmetic intensity of the implicit-GEMM form for $C_{in} = C_{out} = 64$, $K = 3$, and explain why it is far higher than the direct kernel's.
- How would you implement the backward pass with respect to the input? (A transposed convolution: scatter, or gather with flipped filters.)
"""
