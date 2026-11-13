import numpy as np

META = dict(title='Fused row-wise softmax', tags=['GPU Programming', 'Attention'], diff='Medium', runtime='triton',
            tol=[1e-5, 1e-6], lessons=['ieng:13.05', 'aieng:06.03'],
            related=['online-softmax', 'row-wise-logsumexp', 'softmax-rows'])

STATEMENT = r"""
PyTorch's eager softmax is several passes over memory (max, subtract, exp, sum, divide). When a row fits in one block, a fused kernel loads it **once**, does everything in registers, and stores it once.

Write `softmax_rows(x)`.

## Inputs
- `x`: `float32` CUDA tensor of shape `(M, N)` with $1 \le N \le 8192$. Rows may be non-contiguous in memory: use `x.stride(0)` (the column stride is 1).
- Entries may be `-inf` (masked positions), but every row has at least one finite entry.

## Output
- `float32` tensor of shape `(M, N)`, contiguous: the softmax of each row, $y_{ij} = e^{x_{ij} - m_i} / \sum_k e^{x_{ik} - m_i}$ with $m_i = \max_k x_{ik}$.

## Requirements
- One program per row. `BLOCK = triton.next_power_of_2(N)` so the whole row is one block.
- Padding lanes must not change the max or the sum.
- Subtract the row max before `exp`: rows contain values up to about 1000.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def softmax_rows(x):
    """x: (M, N) float32 CUDA tensor (row stride may differ from N). Return the row softmax, (M, N)."""
    raise NotImplementedError


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, N, stride_x, stride_y, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    # padding lanes load -inf: they never win the max and exp(-inf) = 0 adds nothing to the sum
    x = tl.load(x_ptr + row.to(tl.int64) * stride_x + cols, mask=mask, other=float("-inf"))
    z = x - tl.max(x, axis=0)            # row has a finite entry, so the max is finite
    num = tl.exp(z)
    y = num / tl.sum(num, axis=0)
    tl.store(y_ptr + row.to(tl.int64) * stride_y + cols, y, mask=mask)


def softmax_rows(x):
    assert x.ndim == 2 and x.stride(1) == 1
    M, N = x.shape
    y = torch.empty((M, N), dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 2048 else 8      # more warps for long rows keeps registers per thread low
    _softmax_kernel[(M,)](x, y, N, x.stride(0), y.stride(0), BLOCK=BLOCK, num_warps=num_warps)
    return y
'''

REFERENCE = '''import torch

def softmax_rows(x):
    return torch.softmax(x.to(torch.float64), dim=1).to(torch.float32)
'''

HINTS = [
    'One program per row, `cols = tl.arange(0, BLOCK)` with `BLOCK = triton.next_power_of_2(N)`. The row starts at `x_ptr + row * stride_x`.',
    'Load with `other=float("-inf")`. Then `tl.max` ignores padding, and after subtracting the max, `tl.exp` turns padding (and masked `-inf` inputs) into exact zeros.',
    'Divide by `tl.sum(num, axis=0)` and store with the same mask. Pass `x.stride(0)` and `y.stride(0)` separately: they differ when `x` is a column slice of a wider tensor.',
]

_r = np.random.default_rng(17)
_ninf = _r.normal(size=(3, 10)).astype(np.float32)
_ninf[0, 3:] = -np.inf
_ninf[2, [0, 4, 9]] = -np.inf
TESTS = [
    (np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32),),
    (np.array([[1000.0, 1001.0, 999.0, 1000.0]], dtype=np.float32),),
    (_ninf,),
    (np.array([[5.0], [-7.0]], dtype=np.float32),),
    (_r.normal(size=(8, 37)).astype(np.float32) * 5,),
    (_r.normal(size=(4, 1000)).astype(np.float32),),
    (_r.normal(size=(2, 4097)).astype(np.float32) * 2,),
    (np.array([[-1000.0, 0.0, 1000.0]], dtype=np.float32),),
]

EXPLANATION = r"""
## What fusion buys
Eager softmax on an `(M, N)` tensor runs a max kernel, a subtract kernel, an exp kernel, a sum kernel and a divide kernel. Each reads and writes the whole tensor, so about 8 to 10 passes over memory. The fused kernel reads `x` once and writes `y` once: roughly a 4x to 5x speedup, because softmax is completely memory-bound.

## Masking with `-inf`
Padding lanes (when $N$ is not a power of two, tests 5 and 7) and genuinely masked inputs (test 3) both become `-inf`. `tl.max` skips them, `exp(-inf - m) = 0` removes them from the sum, and their outputs are 0. The guarantee of one finite entry per row keeps the max finite; a fully masked row would give `-inf - (-inf) = NaN` and needs an explicit rule.

## Stability
Without the max subtraction, `exp(1000)` overflows to `inf` in float32 (test 2). Subtracting the row max puts the largest exponent at exactly 0 without changing the result, since the factor $e^{-m}$ cancels.

## Limits
The whole row lives in registers, so `BLOCK` up to a few thousand is fine; beyond that registers spill and occupancy drops. Long rows need the chunked **online softmax** instead, which streams over the row while keeping a running max and sum.

## Interview follow-ups
- Why is `num_warps` raised for long rows, and what does it trade off?
- How would you fuse a causal mask and a `1/sqrt(d)` scale into this kernel for attention scores?
"""
