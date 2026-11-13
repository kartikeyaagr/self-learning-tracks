import numpy as np

META = dict(title='Matrix-vector product: one warp per row', tags=['GPU Programming', 'Inference'], diff='Medium',
            runtime='cuda', tol=[1e-4, 1e-5], lessons=['ieng:13.03', 'ieng:03.03'],
            related=['matrix-multiplication', 'matrix-vector-multiply', 'dot-product-2'])

STATEMENT = r"""
During LLM decoding, every linear layer multiplies a weight matrix by a single activation vector. That is $y = Ax$, with $A$ of shape `(M, K)` in row-major order.

The obvious mapping, one thread per row, is badly **uncoalesced**: at each step $k$, the 32 threads of a warp read `A[r, k]`, `A[r+1, k]`, ..., addresses $4K$ bytes apart. Instead, give each row to a **group of 32 threads** that read the row together, so a warp reads 32 consecutive floats per step.

## Launch configuration
- Block shape `(32, 4)`: `threadIdx.x` is the lane within a row (0..31), `threadIdx.y` picks one of 4 rows handled by the block. Grid: `ceil(M / 4)` blocks.
- Row `r = blockIdx.x * 4 + threadIdx.y`. Lane `l` sums `A[r, k] * x[k]` for `k = l, l + 32, l + 64, ...`.
- The 32 partial sums of each row are combined with a tree reduction in a `float32` shared array of shape `(4, 32)`.
- Rows with `r >= M` (in the last block) must still reach every `cuda.syncthreads()`, but must not read `A` or write `y`.

## Inputs and output
- `A`: `float32` `(M, K)`, `x`: `float32` `(K,)`, with $M, K \ge 1$.
- Return `y` as `float32` `(M,)`. Compared with relative tolerance $10^{-4}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda, float32

LANES, ROWS = 32, 4   # block = (LANES, ROWS)

def matvec(A, x):
    """A: float32 (M, K), x: float32 (K,). Returns A @ x as float32 (M,)."""
    M = A.shape[0]
    d_A, d_x = cuda.to_device(A), cuda.to_device(x)
    d_y = cuda.device_array(M, dtype=np.float32)
    matvec_kernel[(M + ROWS - 1) // ROWS, (LANES, ROWS)](d_A, d_x, d_y)
    return d_y.copy_to_host()

@cuda.jit
def matvec_kernel(A, x, y):
    lane = cuda.threadIdx.x
    r = cuda.blockIdx.x * ROWS + cuda.threadIdx.y
    buf = cuda.shared.array((ROWS, LANES), float32)
    # TODO: strided partial dot product per lane, tree reduction over the 32 lanes, lane 0 writes y[r]
    pass
'''

SOLUTION = '''import numpy as np
from numba import cuda, float32

LANES, ROWS = 32, 4   # block = (LANES, ROWS): 32 threads cooperate on each row

def matvec(A, x):
    M = A.shape[0]
    d_A, d_x = cuda.to_device(A), cuda.to_device(x)
    d_y = cuda.device_array(M, dtype=np.float32)
    matvec_kernel[(M + ROWS - 1) // ROWS, (LANES, ROWS)](d_A, d_x, d_y)
    return d_y.copy_to_host()

@cuda.jit
def matvec_kernel(A, x, y):
    lane = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    r = cuda.blockIdx.x * ROWS + ty
    buf = cuda.shared.array((ROWS, LANES), float32)

    acc = float32(0.0)
    if r < A.shape[0]:                         # out-of-range rows contribute 0 but still hit the barriers
        for k in range(lane, A.shape[1], LANES):
            acc += A[r, k] * x[k]              # 32 lanes read 32 consecutive floats of row r
    buf[ty, lane] = acc
    cuda.syncthreads()

    step = LANES // 2
    while step > 0:                            # 16, 8, 4, 2, 1
        if lane < step:
            buf[ty, lane] += buf[ty, lane + step]
        cuda.syncthreads()
        step //= 2

    if lane == 0 and r < A.shape[0]:
        y[r] = buf[ty, 0]
'''

HINTS = [
    'Each lane computes a partial dot product over columns `lane, lane+32, ...`; store it in `buf[threadIdx.y, lane]`.',
    'Reduce along the lane axis only: `buf[ty, lane] += buf[ty, lane + step]` for `step = 16, 8, 4, 2, 1`, with a `cuda.syncthreads()` after each step, outside any `if`.',
    'Guard the reads of `A` and the final write with `r < M`, but let every thread run the reduction loop so the barriers line up.',
]

_r = np.random.default_rng(13)
_f = lambda *s: _r.standard_normal(s).astype(np.float32)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32), np.array([1.0, -1.0], dtype=np.float32)),
    (np.array([[2.0, 0.0, 1.0]], dtype=np.float32), np.array([3.0, 7.0, -1.0], dtype=np.float32)),
    (_f(1, 1), _f(1)),
    (_f(4, 32), _f(32)),         # one block, one column per lane
    (_f(5, 33), _f(33)),         # second block holds one live row; one lane gets two columns
    (_f(10, 7), _f(7)),          # most lanes have nothing to do
    (_f(13, 200), _f(200)),
    (_f(3, 1000), _f(1000)),     # long rows
]

EXPLANATION = r"""
## Why it works
Lane $l$ of row $r$ sums the terms with $k \equiv l \pmod{32}$; together the 32 lanes cover every column exactly once. The tree reduction adds the 32 partials in 5 steps. Each row has its own slice of shared memory (`buf[ty, :]`), so the four rows in a block never interfere.

## Coalescing
With block `(32, 4)`, a warp is exactly one row's 32 lanes. At each loop step it reads `A[r, k..k+31]`: one 128-byte transaction. Thread-per-row would issue 32 separate transactions for the same amount of useful data. The vector `x` is read by every row, so it stays hot in L1/L2 cache.

## Performance
Matrix-vector multiply reads every element of $A$ once and does 2 FLOPs per element: **0.5 FLOP/byte**, firmly memory-bound however you write it. The best possible kernel streams $A$ at full DRAM bandwidth. This is why LLM decoding at batch size 1 is bandwidth-bound and why weight quantization (fewer bytes per weight) speeds it up almost proportionally.

Since a warp is the unit of execution, real kernels replace the shared-memory tree with warp shuffles (`cuda.shfl_down_sync(0xffffffff, acc, step)`): registers only, no barriers.

## Common bugs
- Returning early for `r >= M` before the barriers: those threads never reach `cuda.syncthreads()`.
- Reducing over the whole `(4, 32)` buffer instead of along each row.
- Grid sized `ceil(M / 32)` (confusing lanes with rows): most rows are never computed.

## Interview follow-ups
- At batch size 8, how would you change the kernel to reuse each loaded row of $A$ for all 8 vectors, and how does that change the arithmetic intensity?
- If $A$ were stored column-major, which simple mapping becomes coalesced? (Thread per row: at step $k$ the warp reads column $k$, which is contiguous.)
"""
