import numpy as np

META = dict(title='Tiled matrix transpose', tags=['GPU Programming', 'Linear Algebra'], diff='Medium', runtime='triton',
            lessons=['ieng:13.03', 'air-fanout-v1:03.03'],
            related=['matrix-transpose', 'autotuned-matrix-multiplication', 'block-pointer-matmul'])

STATEMENT = r"""
A transpose does no arithmetic, yet a naive one is slow: if each program reads a row, it writes a column, and consecutive threads store to addresses `M` elements apart. Working on 2-D tiles lets every load *and* every store touch contiguous memory.

Write `transpose(x)` returning a **contiguous** copy of `x.T`.

## Inputs
- `x`: CUDA tensor `(M, N)`, $M, N \ge 1$, of any dtype (tests use `float32` and `int32`). Use `x.stride(0)` and `x.stride(1)`; do not assume contiguity.

## Output
- A new contiguous tensor of shape `(N, M)` and the same dtype, with `out[j, i] = x[i, j]`. Values are copied exactly.

## Requirements
- A 2-D grid of `(cdiv(M, BM), cdiv(N, BN))` programs. Program `(pid_m, pid_n)` loads the tile `x[pid_m*BM : +BM, pid_n*BN : +BN]` with a 2-D mask, transposes it in registers with `tl.trans`, and stores it as the tile `out[pid_n*BN : +BN, pid_m*BM : +BM]`.
- `BM`, `BN` are `tl.constexpr` powers of two (e.g. 32).

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def transpose(x):
    """x: (M, N) CUDA tensor, any dtype and strides. Return x.T as a new contiguous (N, M) tensor."""
    raise NotImplementedError


@triton.jit
def _transpose_kernel(x_ptr, out_ptr, M, N, stride_xm, stride_xn, stride_on, stride_om,
                      BM: tl.constexpr, BN: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(x_ptr, out_ptr, M, N, stride_xm, stride_xn, stride_on, stride_om,
                      BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    in_mask = (rm[:, None] < M) & (rn[None, :] < N)                       # (BM, BN)
    tile = tl.load(x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn, mask=in_mask)
    tile_t = tl.trans(tile)                                                # (BN, BM), via shared memory
    out_mask = (rn[:, None] < N) & (rm[None, :] < M)
    # out row j = rn, out column i = rm: consecutive lanes now write consecutive addresses
    tl.store(out_ptr + rn[:, None] * stride_on + rm[None, :] * stride_om, tile_t, mask=out_mask)


def transpose(x):
    assert x.ndim == 2
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    BM = BN = 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _transpose_kernel[grid](x, out, M, N, x.stride(0), x.stride(1), out.stride(0), out.stride(1), BM=BM, BN=BN)
    return out
'''

REFERENCE = '''import torch

def transpose(x):
    return x.t().contiguous()
'''

HINTS = [
    'Program `(pid_m, pid_n)` owns rows `rm = pid_m*BM + tl.arange(0, BM)` and columns `rn = pid_n*BN + tl.arange(0, BN)` of `x`. Its addresses are `x_ptr + rm[:, None]*stride_xm + rn[None, :]*stride_xn`.',
    'The 2-D mask is `(rm[:, None] < M) & (rn[None, :] < N)`. After `tl.trans(tile)` the tile is `(BN, BM)`, so the output mask is built the other way round.',
    'Output element `(j, i)` lives at `out_ptr + j*stride_on + i*stride_om`, with `j` from `rn` (rows of the output) and `i` from `rm`. The output has shape `(N, M)`.',
]

_r = np.random.default_rng(37)
TESTS = [
    (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),),
    (np.arange(12, dtype=np.int32).reshape(3, 4),),
    (np.array([[9.0]], dtype=np.float32),),
    (np.arange(5, dtype=np.float32).reshape(1, 5),),
    (_r.normal(size=(64, 64)).astype(np.float32),),
    (_r.normal(size=(33, 70)).astype(np.float32),),
    (_r.integers(-1000, 1000, size=(100, 3)).astype(np.int32),),
    (_r.normal(size=(129, 31)).astype(np.float32),),
]

EXPLANATION = r"""
## Coalescing is the whole game
Global memory is fetched in 32-byte sectors. A warp whose 32 threads read 32 consecutive floats uses 4 sectors; if they read 32 floats one row apart, it needs 32 sectors, 8x the traffic for the same data. A row-to-column transpose can make loads coalesced or stores coalesced, never both.

The tiled version reads a `BM x BN` tile row by row (coalesced), exchanges it through **shared memory** (that is what `tl.trans` compiles to), and writes the transposed tile row by row (coalesced again). In CUDA you would allocate the shared buffer, pad it by one column to avoid bank conflicts and add `__syncthreads()`; Triton's compiler does all three.

## Performance
Two bytes moved per byte of output, zero flops: a well-tiled transpose should reach close to device-to-device `memcpy` bandwidth. If it falls well short, look at the store pattern.

## Edges
Tests 6 to 8 have sizes that don't divide 32, so boundary tiles are partial in one or both directions. The output mask must use the swapped bounds: rows of the output are bounded by `N`, columns by `M`.

## Common bugs
- Swapping the strides of the output, which writes a correct-looking but transposed-again result for square inputs only.
- Building the output mask with the input's orientation.
- Returning `x.T` (a view) instead of a new contiguous tensor.

## Interview follow-ups
- Why is `x.t()` free in PyTorch, and when does the real copy happen?
- How would you transpose the last two dimensions of a batched `(B, M, N)` tensor? (A third grid axis and a batch stride.)
"""
