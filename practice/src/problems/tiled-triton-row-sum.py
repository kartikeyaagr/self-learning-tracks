import numpy as np

META = dict(title='Tiled sum along either axis in Triton', tags=['GPU Programming'], diff='Medium', runtime='triton',
            tol=[1e-4, 1e-3], lessons=['ieng:13.04', 'ieng:13.03'],
            related=['vector-sum-reduction', 'triton-row-wise-softmax', 'masked-triton-gelu-kernel'])

STATEMENT = r"""
Reductions over long rows cannot load a whole row into one block. The standard pattern is a **2-D tile** of `BLOCK_M` rows by `BLOCK_N` columns that walks along the reduced axis, accumulating in `float32`, followed by one in-register reduction at the end. Written against strides, the same kernel sums rows of a matrix or rows of its transposed **view**, which is a column sum.

Write `tiled_sum(x, dim)`.

## Inputs
- `x`: contiguous CUDA tensor `(M, N)` of dtype `float16` or `float32`, $M, N \ge 1$; $N$ can be much larger than a block (up to $10^6$).
- `dim`: `1` to sum each row (output length $M$), `0` to sum each column (output length $N$).

## Output
- `float32` tensor of the sums, length $M$ for `dim=1` and $N$ for `dim=0`.

## Requirements
- One kernel that reduces along the second axis of a `(R, C)` view given by `(stride_r, stride_c)`. For `dim=0`, call it on `x.t()` (no copy: pass the transposed strides).
- Each program owns `BLOCK_M` rows and loops over the columns in steps of `BLOCK_N`, masking both the row and the column edges.
- Accumulate a `(BLOCK_M, BLOCK_N)` `float32` tile and reduce it with `tl.sum(acc, axis=1)` once, after the loop. `float16` accumulation is not allowed: it cannot even count past 2048 exactly.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def tiled_sum(x, dim):
    """x: (M, N) float16/float32 CUDA tensor. Return float32 sums along dim (0 or 1)."""
    raise NotImplementedError


@triton.jit
def _row_sum_kernel(x_ptr, out_ptr, R, C, stride_r, stride_c, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(x_ptr, out_ptr, R, C, stride_r, stride_c, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < R
    row_off = rows.to(tl.int64)[:, None] * stride_r                 # (BLOCK_M, 1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, C, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = row_ok[:, None] & (cols < C)[None, :]
        tile = tl.load(x_ptr + row_off + cols.to(tl.int64)[None, :] * stride_c, mask=mask, other=0.0)
        acc += tile.to(tl.float32)                                  # elementwise: no reduction inside the loop
    tl.store(out_ptr + rows, tl.sum(acc, axis=1), mask=row_ok)


def tiled_sum(x, dim):
    assert x.ndim == 2 and x.dtype in (torch.float16, torch.float32) and dim in (0, 1)
    v = x if dim == 1 else x.t()                                    # a view: only the strides change
    R, C = v.shape
    out = torch.empty(R, dtype=torch.float32, device=x.device)
    BLOCK_M, BLOCK_N = 16, 128
    _row_sum_kernel[(triton.cdiv(R, BLOCK_M),)](v, out, R, C, v.stride(0), v.stride(1),
                                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
'''

REFERENCE = '''import torch

def tiled_sum(x, dim):
    return x.to(torch.float64).sum(dim=dim).to(torch.float32)
'''

HINTS = [
    'Row offsets are `rows[:, None] * stride_r` and column offsets `cols[None, :] * stride_c`; their sum is a `(BLOCK_M, BLOCK_N)` grid of pointers. The mask is the outer AND of `rows < R` and `cols < C`.',
    'Keep `acc` as a full `(BLOCK_M, BLOCK_N)` float32 tile and add each loaded tile into it elementwise. Call `tl.sum(acc, axis=1)` only once, after the loop.',
    'For `dim=0`, `x.t()` has strides `(1, N)`. Passing those to the same kernel makes it walk down columns. Loads with stride $N$ between neighbouring lanes are uncoalesced; that is acceptable here, and the follow-ups discuss the fix.',
]

_r = np.random.default_rng(37)
TESTS = [
    (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), 1),
    (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), 0),
    (np.ones((3, 5000), dtype=np.float16), 1),
    (np.ones((5000, 3), dtype=np.float16), 0),
    (_r.normal(size=(37, 1000)).astype(np.float32), 1),
    (_r.normal(size=(37, 1000)).astype(np.float32), 0),
    (_r.normal(size=(1, 200_003)).astype(np.float16), 1),
    (np.array([[7.5]], dtype=np.float32), 0),
    ((_r.normal(size=(129, 17)) * 100).astype(np.float16), 1),
]

EXPLANATION = r"""
## Why it works
The accumulator is a tile of independent partial sums: lane $(i, c)$ sums elements $c, c + B_N, c + 2B_N, \dots$ of row $i$. Adding tiles elementwise inside the loop costs no cross-lane communication; the single `tl.sum` at the end combines the $B_N$ partial sums of each row. Masked lanes load 0, the identity of addition, so ragged edges in both directions are harmless.

Strides make the layout irrelevant to the kernel: `x.t()` is the same memory with strides swapped, so a column sum is a row sum of the transposed view. No transpose copy is made.

## Precision
`float16` has an 11-bit significand: $2048 + 1 = 2048$ in fp16, so an fp16 running sum of 5000 ones (tests 3 and 4) stops at 2048. Accumulating in fp32 also reduces rounding error for long rows (test 7), because each lane sums only $N / B_N$ terms before the final tree reduction.

## Performance
Row sums read memory contiguously along each row: coalesced. Column sums through the transposed view read with stride $N$ between adjacent lanes, which wastes most of each 32-byte sector. A dedicated column-sum kernel would instead put the contiguous axis on the lanes (tile over rows inside the loop and keep one partial sum per column).

## Common bugs
- Calling `tl.sum` inside the loop and adding scalars, which serialises the reduction and forces a cross-lane reduction every iteration.
- Masking only the columns: the last program's rows past $R$ read out of bounds.
- Forgetting `int64` offsets: $M N$ can exceed $2^{31}$ for large tensors.

## Interview follow-ups
- Rewrite the `dim=0` case so loads are coalesced. Which axis maps to `tl.arange` lanes, and what does the loop iterate over?
- When $M$ is tiny and $N$ huge (test 7: one row), one program does all the work. How would a split-reduction (several programs per row plus a second pass or `tl.atomic_add`) fix that, and what does atomic addition do to determinism?
"""
