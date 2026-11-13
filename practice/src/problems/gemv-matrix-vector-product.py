import numpy as np

META = dict(title='GEMV kernel with alpha and beta', tags=['GPU Programming', 'Inference'], diff='Medium',
            runtime='triton', tol=[1e-4, 1e-5], lessons=['ieng:13.03', 'ieng:03.03'],
            related=['matrix-vector-multiply', 'autotuned-matrix-multiplication', 'vector-sum-reduction'])

STATEMENT = r"""
During LLM decoding with batch size 1, every linear layer is a matrix-vector product: the weights are read once and each weight is used for only two flops, so the kernel is limited by memory bandwidth. Write the BLAS-style GEMV:

$$y_{\text{out}} = \alpha\, A x + \beta\, y$$

Write `gemv(A, x, y, alpha, beta)`.

## Inputs
- `A`: `float32` CUDA tensor `(M, N)`, row-major (use `A.stride(0)`; the column stride is 1).
- `x`: `float32` `(N,)`; `y`: `float32` `(M,)`.
- `alpha`, `beta`: Python floats.

## Output
- A new `float32` tensor `(M,)`. `y` itself is not modified.
- **BLAS rule:** when `beta == 0`, `y` is not read at all, so NaN or garbage in `y` must not reach the output.

## Requirements
- A 1-D grid of `cdiv(M, BLOCK_M)` programs; each owns `BLOCK_M` rows and loops over the columns in chunks of `BLOCK_N`.
- Each step loads a `(BLOCK_M, BLOCK_N)` tile of `A` and a `(BLOCK_N,)` chunk of `x`, multiplies with broadcasting and accumulates into a `(BLOCK_M, BLOCK_N)` `float32` accumulator. Reduce over the columns once after the loop. No `tl.dot`.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def gemv(A, x, y, alpha, beta):
    """A: (M, N), x: (N,), y: (M,) float32 CUDA. Return alpha * A @ x + beta * y; y unread when beta == 0."""
    raise NotImplementedError


@triton.jit
def _gemv_kernel(a_ptr, x_ptr, y_ptr, out_ptr, M, N, stride_am, alpha, beta,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, USE_Y: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(a_ptr, x_ptr, y_ptr, out_ptr, M, N, stride_am, alpha, beta,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, USE_Y: tl.constexpr):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_ok = cols < N
        a = tl.load(a_ptr + rows[:, None].to(tl.int64) * stride_am + cols[None, :],
                    mask=row_ok[:, None] & col_ok[None, :], other=0.0)
        xv = tl.load(x_ptr + cols, mask=col_ok, other=0.0)
        acc += a * xv[None, :]                          # elementwise; reduce once after the loop
    out = alpha * tl.sum(acc, axis=1)                   # (BLOCK_M,)
    if USE_Y:                                           # compile-time branch: beta == 0 never touches y
        out += beta * tl.load(y_ptr + rows, mask=row_ok, other=0.0)
    tl.store(out_ptr + rows, out, mask=row_ok)


def gemv(A, x, y, alpha, beta):
    assert A.ndim == 2 and A.stride(1) == 1
    M, N = A.shape
    out = torch.empty(M, dtype=torch.float32, device=A.device)
    BLOCK_M, BLOCK_N = 16, 128
    grid = (triton.cdiv(M, BLOCK_M),)
    _gemv_kernel[grid](A, x, y, out, M, N, A.stride(0), alpha, beta,
                       BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, USE_Y=(beta != 0))
    return out
'''

REFERENCE = '''import torch

def gemv(A, x, y, alpha, beta):
    out = alpha * (A.to(torch.float64) @ x.to(torch.float64))
    if beta != 0:
        out = out + beta * y.to(torch.float64)
    return out.to(torch.float32)
'''

HINTS = [
    'Program `pid` owns `rows = pid*BLOCK_M + tl.arange(0, BLOCK_M)`. Inside the loop, the A tile is at `a_ptr + rows[:, None]*stride_am + cols[None, :]`; broadcasting `x` as `xv[None, :]` multiplies every row by the same chunk.',
    'Accumulate `a * xv[None, :]` into a 2-D `float32` accumulator and call `tl.sum(acc, axis=1)` once after the loop. Masked loads with `other=0.0` handle both edges.',
    'Implement the `beta == 0` rule as a `tl.constexpr` flag (`USE_Y`) computed on the host, so the load of `y` is not even compiled in. Multiplying by 0 is not enough: `0 * NaN` is NaN.',
]

_r = np.random.default_rng(41)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
_nan_y = np.full(70, np.nan, dtype=np.float32)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32),
     np.array([10.0, 20.0], dtype=np.float32), 1.0, 1.0),
    (np.array([[1.0, 0.0, 2.0]], dtype=np.float32), np.array([3.0, 4.0, 5.0], dtype=np.float32),
     np.array([1.0], dtype=np.float32), 2.0, -1.0),
    (_f(70, 300), _f(300), _nan_y, 1.0, 0.0),
    (_f(1, 1), _f(1), _f(1), 0.5, 0.5),
    (_f(256, 1024), _f(1024), _f(256), 1.0, 0.0),
    (_f(33, 129), _f(129), _f(33), -1.5, 2.0),
    (_f(1000, 17), _f(17), _f(1000), 1.0, 1.0),
    (_f(5, 4000), _f(4000), _f(5), 0.01, 3.0),
]

EXPLANATION = r"""
## GEMV is a bandwidth problem
`A` has $MN$ elements, each used once for one multiply and one add. In float32 that is 2 flops per 4 bytes, an arithmetic intensity of 0.5 flops/byte. A GPU with 2 TB/s of bandwidth and 60 TFLOP/s of float32 compute would need about 30 flops/byte to be compute-bound, so a good GEMV is judged only by how close it gets to peak bandwidth. That is also why LLM decoding at batch size 1 is memory-bound, and why weight quantization (fewer bytes per weight) speeds it up almost proportionally.

## Kernel shape
Each program streams `BLOCK_M` rows of `A`. Within a tile, consecutive lanes read consecutive columns, so loads are coalesced. `x` chunks are reloaded by every program, but `x` is tiny and stays in L2. Accumulating a 2-D tile and reducing once at the end keeps the loop free of cross-lane shuffles.

## The `beta == 0` rule
BLAS defines `beta = 0` to mean "don't read y", so callers can pass an uninitialised buffer. `0 * NaN = NaN` in IEEE arithmetic, so the rule has to be implemented as a branch, not a multiply (test 3). Making it a `tl.constexpr` creates two compiled variants and costs nothing at run time.

## Common bugs
- Reducing along the wrong axis (`axis=0` sums over rows).
- Masking only rows or only columns: both edges are partial in tests 6 to 8.
- Using `tl.dot` with a vector: it needs 2-D tiles of at least 16 in every dimension and wastes tensor-core work here.

## Interview follow-ups
- For a very wide `A` (say $M = 8$, $N = 10^6$), only a few programs run. How would split-N fix that?
- How would you fuse int4 weight dequantization into this kernel, and why does it help decode latency?
"""
