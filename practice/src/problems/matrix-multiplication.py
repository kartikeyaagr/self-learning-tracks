import numpy as np

META = dict(title='Naive matrix multiplication kernel', tags=['GPU Programming', 'Linear Algebra'], diff='Medium',
            runtime='cuda', tol=[1e-4, 1e-5], lessons=['ieng:13.01', 'ieng:12.07'],
            related=['tiled-matrix-multiplication', 'matrix-addition', 'matrix-vector-multiplication'])

STATEMENT = r"""
Matrix multiplication is the operation GPUs exist for. Start with the direct mapping: one thread computes one output entry,

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}$$

Write `matmul(A, B)`.

## Inputs and output
- `A`: `float32` `(M, K)`, `B`: `float32` `(K, N)`, with $M, N, K \ge 1$. Neither dimension needs to be a multiple of the block size.
- Return `C` as `float32` `(M, N)`.

## Requirements
- Block `(16, 16)`, grid `(ceil(N/16), ceil(M/16))`; `col, row = cuda.grid(2)` with x → column.
- Accumulate the dot product in a **local variable** (a register) and write `C[row, col]` once at the end, rather than accumulating into global memory.
- Results are compared with relative tolerance $10^{-4}$: different summation orders round differently in `float32`.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda

BLOCK = (16, 16)

def matmul(A, B):
    """A: float32 (M, K), B: float32 (K, N). Returns A @ B as float32 (M, N)."""
    M, K = A.shape
    N = B.shape[1]
    d_A, d_B = cuda.to_device(A), cuda.to_device(B)
    d_C = cuda.device_array((M, N), dtype=np.float32)
    grid = ((N + 15) // 16, (M + 15) // 16)
    matmul_kernel[grid, BLOCK](d_A, d_B, d_C)
    return d_C.copy_to_host()

@cuda.jit
def matmul_kernel(A, B, C):
    col, row = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        # TODO: dot product of row `row` of A with column `col` of B
        C[row, col] = 0.0
'''

SOLUTION = '''import numpy as np
from numba import cuda, float32

BLOCK = (16, 16)

def matmul(A, B):
    M, K = A.shape
    N = B.shape[1]
    d_A, d_B = cuda.to_device(A), cuda.to_device(B)
    d_C = cuda.device_array((M, N), dtype=np.float32)
    grid = ((N + BLOCK[0] - 1) // BLOCK[0], (M + BLOCK[1] - 1) // BLOCK[1])
    matmul_kernel[grid, BLOCK](d_A, d_B, d_C)
    return d_C.copy_to_host()

@cuda.jit
def matmul_kernel(A, B, C):
    col, row = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        acc = float32(0.0)                    # lives in a register
        for k in range(A.shape[1]):
            acc += A[row, k] * B[k, col]      # A: same address across a half-warp (broadcast); B: coalesced
        C[row, col] = acc                     # one global write per thread
'''

HINTS = [
    'Thread `(row, col)` loops `k` over `A.shape[1]` and sums `A[row, k] * B[k, col]`.',
    'Keep the sum in a local `acc` and store it once. `float32(0.0)` (from `numba import float32`) keeps the arithmetic in single precision.',
    'Leave the bounds check around the whole loop, so threads in the edge blocks don\'t read past the end of `A` or `B` either.',
]

_r = np.random.default_rng(12)
_f = lambda *s: _r.standard_normal(s).astype(np.float32)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)),
    (np.array([[1.0, 0.0, -1.0]], dtype=np.float32), np.array([[2.0], [3.0], [4.0]], dtype=np.float32)),
    (np.array([[3.0]], dtype=np.float32), np.array([[-2.0]], dtype=np.float32)),
    (_f(4, 1), _f(1, 5)),                     # K = 1: an outer product
    (np.eye(17, dtype=np.float32), _f(17, 9)),
    (_f(16, 16), _f(16, 16)),
    (_f(33, 20), _f(20, 18)),
    (_f(5, 64), _f(64, 37)),
]

EXPLANATION = r"""
## Why it works
Every output entry is an independent dot product, so one thread per entry needs no communication. The grid covers $C$ and the bounds check discards the overhang.

## Why it is slow: arithmetic intensity
Each thread reads $2K$ floats ($8K$ bytes) to do $2K$ FLOPs: **0.25 FLOP/byte**. A modern GPU needs tens of FLOPs per byte of DRAM traffic to be compute-bound (that ratio is the ridge point of the roofline). So the naive kernel runs at a few percent of peak, limited by memory traffic.

The waste is reuse: row $i$ of $A$ is read by all $N$ threads of output row $i$, and column $j$ of $B$ by all $M$ threads of column $j$. Caches catch some of it, but the fix is to **stage tiles in shared memory** so each value is fetched from DRAM once per tile and then reused by 16 threads. That is the tiled-matmul problem in this plan; beyond that come register tiling (each thread computes a small block of $C$) and tensor cores.

## Access pattern
Within a half-warp (same `row`, consecutive `col`), `A[row, k]` is the same address for all threads (one broadcast), and `B[k, col]` is 16 consecutive floats (coalesced). With x mapped to rows instead, `B` would still be fine, but `A[row, k]` would be 16 addresses $K$ floats apart.

## Common bugs
- `C[row, col] += ...` inside the loop: $K$ global read-modify-writes per thread, and wrong if `C` isn't zeroed (`device_array` is uninitialised).
- Looping over `B.shape[0]` vs `A.shape[1]`: same value, but mixing up `M`/`N`/`K` in the grid is the usual error for non-square shapes.
- Expecting bit-exact equality with NumPy: BLAS sums in a different order.

## Interview follow-ups
- Compute the arithmetic intensity of an $N \times N$ matmul if every input is read exactly once. Why does it grow with $N$? ($2N^3$ FLOPs over $12N^2$ bytes: $N/6$.)
- Why do LLM decoding steps (matrix × vector) not benefit from tiling the way prefill (matrix × matrix) does?
"""
