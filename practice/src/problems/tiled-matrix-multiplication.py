import numpy as np

META = dict(title='Tiled matrix multiplication in shared memory', tags=['GPU Programming', 'Linear Algebra'], diff='Hard',
            runtime='cuda', tol=[1e-4, 1e-5], lessons=['ieng:13.03', 'ieng:12.07'],
            related=['matrix-multiplication', '2d-convolution-2', 'matrix-vector-multiplication'])

STATEMENT = r"""
The naive matmul kernel (one thread per output, reading straight from global memory) performs only about 0.25 FLOP per byte loaded: every element of $A$ is fetched $N$ times and every element of $B$ is fetched $M$ times. **Tiling** stages square sub-blocks of $A$ and $B$ in shared memory so that each value fetched from global memory is reused by a whole row or column of threads.

## The algorithm
With tile size $T = 16$, block `(16, 16)` computes the tile $C[r_0 : r_0+16,\ c_0 : c_0+16]$, and grid $(\lceil N/16 \rceil, \lceil M/16 \rceil)$ with x → columns. The block walks along the shared dimension in $\lceil K/16 \rceil$ phases. In phase $p$:

1. Every thread loads **one** element of each tile: `As[ty, tx] = A[row, p*16 + tx]` and `Bs[ty, tx] = B[p*16 + ty, col]`, writing 0 where the index is outside the matrix.
2. `cuda.syncthreads()`: both tiles are complete.
3. Every thread accumulates `As[ty, k] * Bs[k, tx]` for `k = 0..15` into a register.
4. `cuda.syncthreads()` again: nobody overwrites the tiles in the next phase while a slower thread is still reading them.

After the last phase, threads inside the matrix write `C[row, col]`.

Write `tiled_matmul(A, B)`.

## Inputs and output
- `A`: `float32` `(M, K)`, `B`: `float32` `(K, N)`, $M, N, K \ge 1$, none of them necessarily a multiple of 16.
- Return `C = A @ B`: `float32` `(M, N)`. Compared with relative tolerance $10^{-4}$, absolute $10^{-5}$.
- Threads whose `(row, col)` lies outside `C` still take part in loading and in every barrier; they just do not write.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda, float32

TILE = 16

def tiled_matmul(A, B):
    """A: float32 (M, K), B: float32 (K, N). Returns float32 (M, N)."""
    M, N = A.shape[0], B.shape[1]
    d_A = cuda.to_device(np.ascontiguousarray(A))
    d_B = cuda.to_device(np.ascontiguousarray(B))
    d_C = cuda.device_array((M, N), dtype=np.float32)
    grid = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE)
    tiled_kernel[grid, (TILE, TILE)](d_A, d_B, d_C)
    return d_C.copy_to_host()

@cuda.jit
def tiled_kernel(A, B, C):
    As = cuda.shared.array((TILE, TILE), float32)
    Bs = cuda.shared.array((TILE, TILE), float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    col, row = cuda.grid(2)
    # TODO: loop over phases: load one element of each tile (0 outside), barrier,
    #       accumulate 16 products from shared memory, barrier. Then write C.
    pass
'''

SOLUTION = '''import numpy as np
from numba import cuda, float32

TILE = 16

def tiled_matmul(A, B):
    M, N = A.shape[0], B.shape[1]
    d_A = cuda.to_device(np.ascontiguousarray(A))
    d_B = cuda.to_device(np.ascontiguousarray(B))
    d_C = cuda.device_array((M, N), dtype=np.float32)
    grid = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE)
    tiled_kernel[grid, (TILE, TILE)](d_A, d_B, d_C)
    return d_C.copy_to_host()

@cuda.jit
def tiled_kernel(A, B, C):
    As = cuda.shared.array((TILE, TILE), float32)
    Bs = cuda.shared.array((TILE, TILE), float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    col, row = cuda.grid(2)                         # x -> column, y -> row
    M, K = A.shape
    N = B.shape[1]

    acc = float32(0.0)
    for p in range((K + TILE - 1) // TILE):
        k0 = p * TILE
        # 1. Each thread loads one element of each tile; zeros pad the ragged edges.
        As[ty, tx] = A[row, k0 + tx] if row < M and k0 + tx < K else float32(0.0)
        Bs[ty, tx] = B[k0 + ty, col] if k0 + ty < K and col < N else float32(0.0)
        cuda.syncthreads()                          # 2. tiles complete
        for k in range(TILE):                       # 3. 16 FMAs from shared memory
            acc += As[ty, k] * Bs[k, tx]
        cuda.syncthreads()                          # 4. done reading before the next overwrite

    if row < M and col < N:                         # out-of-range threads helped load; they don't write
        C[row, col] = acc
'''

HINTS = [
    'The phase loop runs `for p in range(ceil(K / 16))` with `k0 = p * 16`. Thread `(tx, ty)` loads `A[row, k0 + tx]` into `As[ty, tx]` and `B[k0 + ty, col]` into `Bs[ty, tx]`.',
    'Pad with zeros instead of skipping: a zero in `As` or `Bs` contributes nothing to the sum, so the inner loop can always run all 16 steps, even in the last, partial phase.',
    'Two barriers per phase: one after the loads, one after the 16 multiply-adds. Do the final bounds check only around the write to `C`; never `return` early, because out-of-range threads still load tile elements and must reach every barrier.',
]

_r = np.random.default_rng(56)
_f = lambda *s: _r.standard_normal(s).astype(np.float32)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)),
    (np.arange(6, dtype=np.float32).reshape(2, 3), np.ones((3, 4), dtype=np.float32)),
    (_f(1, 1), _f(1, 1)),
    (_f(16, 16), _f(16, 16)),                       # exactly one tile, one phase
    (_f(1, 40), _f(40, 1)),                         # a dot product: K spans 3 phases
    (_f(33, 1), _f(1, 17)),                         # outer product: K = 1
    (_f(20, 37), _f(37, 50)),                       # ragged in every dimension
    (_f(64, 48), _f(48, 32)),
]

EXPLANATION = r"""
## Why it works
$C_{ij} = \sum_k A_{ik} B_{kj}$ can be split into phases over consecutive $k$-ranges of length 16 and the partial sums added. In phase $p$, the 16 values `A[row, k0:k0+16]` that thread `(ty, tx)` needs are exactly row `ty` of `As`, and the values `B[k0:k0+16, col]` are column `tx` of `Bs`; the block as a whole loads both tiles once. The two barriers make the phases behave as if they ran one after the other for the whole block. Zero padding leaves the sums unchanged.

## Reuse and arithmetic intensity
Each element loaded into `As` is used by the 16 threads of its row and each element of `Bs` by the 16 threads of its column. Global loads drop from $2K$ per thread to $2K/16$, raising arithmetic intensity from 0.25 to about 4 FLOP/byte for $T = 16$, and in general to $T/4$. Larger tiles help further but cost shared memory and registers, which limits how many blocks fit on an SM (**occupancy**). High-performance kernels add **register tiling** (each thread computes, say, an $8 \times 8$ patch of $C$, reaching tens of FLOP/byte), double-buffering (loading the next tile while computing the current one), and tensor-core instructions.

## Shared-memory access in the inner loop
For fixed `k`, a warp (two rows `ty` of 16 threads) reads `As[ty, k]`: two addresses, each broadcast to 16 threads. It reads `Bs[k, tx]`: 16 consecutive words, the same for both halves, so 16 distinct banks with broadcast. Neither causes bank conflicts. The loads into the tiles are coalesced too: consecutive `tx` read consecutive columns of `A` and of `B`.

## Common bugs
- Missing the second barrier: a fast thread starts the next phase and overwrites `As`/`Bs` while others are still multiplying. Results are wrong only sometimes, which makes it hard to debug.
- Returning early for threads outside `C`: they never load their tile element (leaving garbage) and never reach the barriers.
- `As[ty, tx] = A[row, k0 + ty]` (swapped indices) loads the transpose of the tile.

## Interview follow-ups
- With 48 KB of shared memory per block and `float32` tiles of $A$ and $B$, what is the largest square tile, and why might you still choose a smaller one?
- Explain why decode-time LLM inference (batch 1, matrix × vector) gains almost nothing from this kernel, and what limits its speed instead.
"""
