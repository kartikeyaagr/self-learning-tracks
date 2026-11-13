import numpy as np

META = dict(title='Fused multiply-add kernel', tags=['GPU Programming'], diff='Easy', runtime='triton', tol=[1e-5, 1e-5],
            lessons=['ieng:13.02', 'ieng:13.04'], related=['silu', 'vectorized-vector-add'])

STATEMENT = r"""
Your first Triton kernel. Each **program** (Triton's unit of parallel work, roughly one CUDA thread block) handles one contiguous block of `BLOCK` elements, and every load and store is masked so the last, partial block never touches memory past the end.

Write `fma(x, y, z)` that returns `x * y + z` elementwise, computed by a `@triton.jit` kernel.

## Inputs
- `x`, `y`, `z`: `float32` CUDA tensors of the **same shape**, contiguous. Any shape and any number of elements, including 0 and sizes that are not a multiple of the block size.

## Output
- A new `float32` CUDA tensor with the shape of `x`.
- If `x` has no elements, return an empty tensor without launching the kernel (a grid of 0 programs is an error).

## Requirements
- Treat the tensors as flat arrays of `n = x.numel()` elements: program `pid` covers offsets `pid * BLOCK + [0, BLOCK)`.
- `BLOCK` must be a `tl.constexpr` power of two; launch `triton.cdiv(n, BLOCK)` programs.

This problem runs in **Colab with a GPU runtime** (Runtime → Change runtime type → GPU). Expected outputs were computed with an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def fma(x, y, z):
    """x, y, z: float32 CUDA tensors of one shape. Return x * y + z from a Triton kernel."""
    # TODO: allocate the output, pick BLOCK, compute the grid, launch _fma_kernel
    raise NotImplementedError


@triton.jit
def _fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # TODO: offsets for this program, a mask for the tail, masked loads, compute, masked store
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)       # this program's slice of the flat array
    mask = offs < n                                # the last block may hang past the end
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = tl.load(z_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * y + z, mask=mask)  # the compiler emits one fused FMA instruction


def fma(x, y, z):
    assert x.shape == y.shape == z.shape and x.is_cuda
    x, y, z = x.contiguous(), y.contiguous(), z.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:                                     # grid of zero programs is not allowed
        return out
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _fma_kernel[grid](x, y, z, out, n, BLOCK=BLOCK)
    return out
'''

REFERENCE = '''import torch

def fma(x, y, z):
    return x * y + z
'''

HINTS = [
    'Program `pid` owns offsets `pid * BLOCK + tl.arange(0, BLOCK)`. Passing a tensor to a kernel passes a pointer to its first element, so `x_ptr + offs` is a block of pointers.',
    'Build `mask = offs < n` once and pass it to every `tl.load` and `tl.store`. Without it the last program reads and writes past the end of the buffers.',
    'On the host: `out = torch.empty_like(x)`, `grid = (triton.cdiv(n, BLOCK),)`, then `_fma_kernel[grid](x, y, z, out, n, BLOCK=1024)`. Return early when `n == 0`.',
]

_r = np.random.default_rng(3)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
TESTS = [
    (np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32),
     np.array([0.5, -1.0, 0.0], dtype=np.float32)),
    (np.array([[1.0, -2.0], [0.0, 3.0]], dtype=np.float32), np.full((2, 2), 2.0, dtype=np.float32),
     np.ones((2, 2), dtype=np.float32)),
    (np.array([7.0], dtype=np.float32), np.array([-3.0], dtype=np.float32), np.array([1.0], dtype=np.float32)),
    (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)),
    (_f(1024), _f(1024), _f(1024)),
    (_f(3000), _f(3000), _f(3000)),
    (_f(7, 13, 5), _f(7, 13, 5), _f(7, 13, 5)),
    (_f(1025) * 100, _f(1025) * 100, _f(1025)),
]

EXPLANATION = r"""
## The block model
CUDA asks you to think about one thread; Triton asks you to think about one **program** working on a whole block of data at once. `tl.arange(0, BLOCK)` is a vector of `BLOCK` lane indices, `x_ptr + offs` is a vector of addresses, and `tl.load` fetches them all. The compiler decides how to split that block across the threads of a warp and how wide each memory transaction is.

## Why the mask matters
With `n = 3000` and `BLOCK = 1024` there are 3 programs. The third covers offsets 2048..3071, and the last 72 lanes are past the end. A masked load returns `other` (undefined if not given, which is fine because those lanes are never stored) and a masked store simply skips the lane. Test 6 and test 8 (one element past a full block) catch a missing mask.

## Performance
Each element moves 16 bytes (three loads and one store of 4 bytes) for 2 flops, so this kernel is entirely **memory-bound**: its speed is set by DRAM bandwidth, not arithmetic. Fusing `x * y + z` into one kernel instead of `t = x * y; t + z` saves a full write and read of the temporary `t`, i.e. 8 of 24 bytes per element.

## Common bugs
- Forgetting the mask, or computing it from `BLOCK` instead of `n`.
- A non-power-of-two `BLOCK`: `tl.arange` requires a power of two.
- Launching a grid of 0 programs for an empty tensor.
- Passing a non-contiguous tensor: flat offsets assume contiguous storage.

## Interview follow-ups
- How would you support broadcasting, e.g. `z` of shape `(N,)` added to `x * y` of shape `(M, N)`? (Load `z` with `offs % N`, or use a 2-D grid.)
- At what point does a bigger `BLOCK` stop helping, and what does `num_warps` change?
"""
