import numpy as np

META = dict(title='Row-wise L2 normalisation: eight rows per block', tags=['GPU Programming', 'Inference'], diff='Medium',
            runtime='cuda', tol=[1e-5, 1e-7], lessons=['ieng:13.03', 'ieng:13.04'],
            related=['l1-normalization', 'rms-normalization', 'cosine-similarity-2'])

STATEMENT = r"""
Embedding models L2-normalise every output vector so that a dot product becomes a cosine similarity:

$$y_{r,:} = \frac{x_{r,:}}{\max\!\left(\lVert x_{r,:} \rVert_2,\ \varepsilon\right)}, \qquad \varepsilon = 10^{-12}$$

Embedding rows are short (a few hundred numbers), so a whole block per row would leave most threads idle on small dimensions. Instead, pack **several rows into one block** with a 2-D block shape.

## The mapping
- Block shape `(32, 8)`: `threadIdx.y` picks one of 8 rows handled by the block, and the 32 threads with the same `threadIdx.y` (one warp) cooperate on that row. Grid: $\lceil \text{rows}/8 \rceil$ blocks.
- Row index `r = blockIdx.x * 8 + threadIdx.y`. Lane `tx` sums squares over columns `tx, tx + 32, ...`.
- Partial sums go into a **2-D shared array** `cuda.shared.array((8, 32), float32)`; reduce along the second axis with a tree (`step = 16, 8, 4, 2, 1`), calling `cuda.syncthreads()` after every step.
- Then every lane writes its columns of `y`.

The last block may contain rows past the end. Those threads must **still reach every `cuda.syncthreads()`**: guard the loads and stores, never return early.

## Inputs and output
- `x`: `float32`, shape `(rows, dim)`, `rows >= 1`, `dim >= 1`.
- Return `y`: `float32`, same shape. An all-zero row stays zero.
- Compared with relative tolerance $10^{-5}$, absolute $10^{-7}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda, float32

LANES, ROWS_PER_BLOCK = 32, 8
EPS = 1e-12

def l2_normalize_rows(x):
    """x: float32 (rows, dim). Returns float32 (rows, dim) with unit-L2 rows."""
    rows, dim = x.shape
    d_x = cuda.to_device(np.ascontiguousarray(x))
    d_y = cuda.device_array((rows, dim), dtype=np.float32)
    grid = (rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    l2_kernel[grid, (LANES, ROWS_PER_BLOCK)](d_x, d_y)
    return d_y.copy_to_host()

@cuda.jit
def l2_kernel(x, y):
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    r = cuda.blockIdx.x * ROWS_PER_BLOCK + ty
    # TODO: partial sum of squares per lane, tree-reduce in a (8, 32) shared array, then scale
    pass
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda, float32

LANES, ROWS_PER_BLOCK = 32, 8
EPS = 1e-12

def l2_normalize_rows(x):
    rows, dim = x.shape
    d_x = cuda.to_device(np.ascontiguousarray(x))
    d_y = cuda.device_array((rows, dim), dtype=np.float32)
    grid = (rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    l2_kernel[grid, (LANES, ROWS_PER_BLOCK)](d_x, d_y)
    return d_y.copy_to_host()

@cuda.jit
def l2_kernel(x, y):
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    r = cuda.blockIdx.x * ROWS_PER_BLOCK + ty
    rows, dim = x.shape
    buf = cuda.shared.array((ROWS_PER_BLOCK, LANES), float32)   # one 32-slot strip per row

    acc = float32(0.0)
    if r < rows:                                  # guard the work, not the barriers
        for j in range(tx, dim, LANES):
            v = x[r, j]
            acc += v * v
    buf[ty, tx] = acc
    cuda.syncthreads()

    step = LANES // 2
    while step > 0:                               # each row's strip reduces independently
        if tx < step:
            buf[ty, tx] += buf[ty, tx + step]
        cuda.syncthreads()                        # reached by all 256 threads, out-of-range rows too
        step //= 2

    if r < rows:
        inv = float32(1.0) / max(math.sqrt(buf[ty, 0]), float32(EPS))
        for j in range(tx, dim, LANES):
            y[r, j] = x[r, j] * inv
'''

HINTS = [
    'Each thread computes `acc` = sum of `x[r, j]**2` over `j = tx, tx+32, ...` (0 if `r` is out of range) and stores it at `buf[ty, tx]`.',
    'The tree is the usual one, but indexed `buf[ty, tx] += buf[ty, tx + step]`. All 8 rows reduce in the same steps, sharing the same barriers.',
    'Do not `return` early for `r >= rows`: a thread that leaves the kernel never arrives at `cuda.syncthreads()`. Wrap the loads and the final writes in `if r < rows:` instead.',
]

_r = np.random.default_rng(42)
_z = _r.standard_normal((10, 5)).astype(np.float32)
_z[3] = 0.0
TESTS = [
    (np.array([[3.0, 4.0], [0.0, -2.0]], dtype=np.float32),),
    (np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),),          # one row in an 8-row block
    (_z,),                                                           # a zero row, 2 blocks
    (_r.standard_normal((9, 1)).astype(np.float32),),                # dim 1: every entry +-1
    (_r.standard_normal((16, 100)).astype(np.float32),),             # dim > 32: lanes loop
    ((_r.standard_normal((5, 32)) * 1e-23).astype(np.float32),),     # squares underflow to 0
    ((_r.standard_normal((37, 64)) * 30).astype(np.float32),),
]

EXPLANATION = r"""
## Why it works
The 2-D block is really 8 independent row reductions that share barriers. Thread $(tx, ty)$ only ever reads and writes strip `buf[ty, :]`, so rows never interfere; the shared `cuda.syncthreads()` simply orders each step for all 8 strips at once. After 5 steps, `buf[ty, 0]` holds $\lVert x_{r,:} \rVert_2^2$.

## Barriers and out-of-range threads
`cuda.syncthreads()` waits until **every** thread of the block arrives. If the threads of a missing row returned early, the remaining ones would wait forever (or, on newer hardware, behave undefinedly). The general rule: make control flow around barriers uniform across the block and move bounds checks onto the memory accesses.

## Choosing the shape
With `dim = 64` and one 64-thread block per row, a 10,000-row batch needs 10,000 small blocks; each finishes in a few hundred cycles and the per-block scheduling overhead becomes visible. Eight rows per block amortises that. Mapping each row to exactly one warp also means a real implementation can drop shared memory for warp shuffles (`__shfl_xor_sync`), which need no barrier at all. Very long rows go the other way: one block (or several) per row.

## Precision
Squaring values around $10^{-23}$ underflows to 0 in `float32` (test 6): the norm comes out as 0, the clamp kicks in, and the kernel returns $x/\varepsilon$ (entries near $10^{-11}$) where the true answer is a unit vector. Production code scales by the row's max-abs first, or accumulates in higher precision; here the specification (and the test) accepts the underflow, which is what the textbook formula does in `float32`.

## Common bugs
- Early `return` before the barriers (deadlock in the last block).
- Indexing the shared array as `buf[tx, ty]` with shape `(8, 32)`: out of range for `tx >= 8`.
- Forgetting the barrier between writing `buf` and the first tree step.

## Interview follow-ups
- Rewrite the reduction with warp shuffles. Why are the shared array and the barriers no longer needed?
- How would you fuse this normalisation into the kernel that produced the embeddings (for example the last matmul plus mean pooling)?
"""
