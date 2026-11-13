import numpy as np

META = dict(title='Matrix addition on a 2-D grid', tags=['GPU Programming', 'Linear Algebra'], diff='Easy',
            runtime='cuda', lessons=['ieng:13.01', 'ieng:13.03'], related=['vector-addition', 'matrix-multiplication'])

STATEMENT = r"""
Matrices map naturally onto **2-D grids**: blocks of $16 \times 16$ threads tile the output, and each thread owns one entry. Write `matrix_add(A, B)` returning $C = A + B$.

## Inputs and output
- `A`, `B`: `float32` arrays of shape `(M, N)`, with $M, N \ge 1$ and not necessarily multiples of 16. Row-major (C-contiguous).
- Return `C` as a `float32` array of shape `(M, N)`.

## Launch configuration
- Block: `(16, 16)` threads. Grid: `(ceil(N / 16), ceil(M / 16))` blocks.
- Use `col, row = cuda.grid(2)`. The **x** coordinate (first value) is the one that varies fastest across consecutive threads in a warp, so map it to the **column**: in row-major storage, consecutive columns are consecutive addresses.
- Threads outside the matrix (in the partial blocks along the right and bottom edges) must not write.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda

BLOCK = (16, 16)

def matrix_add(A, B):
    """A, B: float32 (M, N). Returns A + B as float32 (M, N)."""
    M, N = A.shape
    d_A, d_B = cuda.to_device(A), cuda.to_device(B)
    d_C = cuda.device_array((M, N), dtype=np.float32)
    grid = (1, 1)  # TODO: enough blocks to cover N columns (x) and M rows (y)
    add_kernel[grid, BLOCK](d_A, d_B, d_C)
    return d_C.copy_to_host()

@cuda.jit
def add_kernel(A, B, C):
    col, row = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]
'''

SOLUTION = '''import numpy as np
from numba import cuda

BLOCK = (16, 16)  # (x = columns, y = rows)

def matrix_add(A, B):
    M, N = A.shape
    d_A, d_B = cuda.to_device(A), cuda.to_device(B)
    d_C = cuda.device_array((M, N), dtype=np.float32)
    grid = ((N + BLOCK[0] - 1) // BLOCK[0],   # x: blocks across columns
            (M + BLOCK[1] - 1) // BLOCK[1])   # y: blocks down rows
    add_kernel[grid, BLOCK](d_A, d_B, d_C)
    return d_C.copy_to_host()

@cuda.jit
def add_kernel(A, B, C):
    col, row = cuda.grid(2)                   # x varies fastest within a warp -> column
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]
'''

HINTS = [
    'The starter launches a single 16×16 block, so only the top-left corner is computed. Work out how many blocks you need in each direction.',
    'Grid x counts column blocks: `(N + 15) // 16`. Grid y counts row blocks: `(M + 15) // 16`. The order is (x, y), the opposite of NumPy\'s (rows, cols).',
    'Keep both bounds checks; with `M = 17`, the second row of blocks is almost entirely outside the matrix.',
]

_r = np.random.default_rng(11)
_f = lambda *s: _r.standard_normal(s).astype(np.float32)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)),
    (np.array([[1.0, -1.0, 0.5]], dtype=np.float32), np.array([[0.5, 0.5, 0.5]], dtype=np.float32)),
    (_f(1, 1), _f(1, 1)),
    (_f(16, 16), _f(16, 16)),       # exactly one block
    (_f(17, 33), _f(17, 33)),       # partial blocks on both edges
    (_f(40, 3), _f(40, 3)),         # tall and thin
    (_f(2, 70), _f(2, 70)),         # short and wide
]

EXPLANATION = r"""
## Why it works
The grid covers a $16\lceil N/16 \rceil \times 16\lceil M/16 \rceil$ rectangle that contains the matrix. Each in-range thread writes exactly one distinct entry, and the bounds check drops the overhang.

## Coalescing: why x maps to columns
A warp is 32 threads with consecutive `threadIdx.x` (for a 16-wide block, that is two rows of 16). With `col` from x, those threads read `A[row, col .. col+15]`: 64 contiguous bytes per row segment, served by one or two memory transactions. Swap the mapping (`row, col = cuda.grid(2)`) and the 16 threads of each half-warp read addresses `N * 4` bytes apart: 16 separate transactions for the same data. On a real GPU that is several times slower for large matrices, although the answer is identical, which is why this bug survives testing.

## Performance
Like vector addition, this is purely bandwidth-bound: 12 bytes of traffic per FLOP. For element-wise ops on contiguous matrices, a 1-D launch over the flattened array is just as fast and simpler. 2-D grids pay off when the computation needs row/column structure, as in matmul, transpose or convolution.

## Common bugs
- Grid in (rows, cols) order: wrong coverage when $M \ne N$ (tests 6 and 7).
- `row, col = cuda.grid(2)`: correct results, uncoalesced loads.
- Checking only one bound.

## Interview follow-ups
- What's the maximum number of threads per block, and why is $32 \times 32$ the largest square block? (1024 on current GPUs.)
- For `C = A + B.T`, which operand's reads become uncoalesced, and how would shared memory fix it? (The transposed one; stage a tile in shared memory, as in a tiled transpose.)
"""
