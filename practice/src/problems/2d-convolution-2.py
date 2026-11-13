import numpy as np

META = dict(title='2-D convolution with a shared-memory tile and halo', tags=['GPU Programming', 'Convolutional Networks'],
            diff='Medium', runtime='cuda', tol=[1e-5, 1e-5], lessons=['ieng:13.03', 'ieng:13.01'],
            related=['1d-convolution', '3d-convolution', 'tiled-matrix-multiplication'])

STATEMENT = r"""
A "valid" 2-D cross-correlation of an $H \times W$ image with a $K \times K$ filter produces an $(H-K+1) \times (W-K+1)$ output:

$$y_{r,c} = \sum_{a=0}^{K-1} \sum_{b=0}^{K-1} w_{a,b}\, x_{r+a,\, c+b}$$

One thread per output works, but each input pixel is then fetched from global memory by up to $K^2$ different threads. Here, each block first copies the input region it needs into **shared memory** and computes from there.

## The tiling
- Block `(16, 16)` computes a $16 \times 16$ tile of outputs; grid $(\lceil W_o/16 \rceil, \lceil H_o/16 \rceil)$ with x → columns.
- The tile's outputs read an input patch of $(16 + K - 1) \times (16 + K - 1)$ pixels: the tile plus a **halo** of $K - 1$ extra rows and columns. Allocate `cuda.shared.array((20, 20), float32)` (the size for the largest allowed $K = 5$) and use its top-left corner.
- **Cooperative load**: the patch has more pixels than the block has threads, so number the threads `t = ty * 16 + tx` and let thread `t` load patch entries `t, t + 256, t + 512, ...` (flattened row-major). Pixels outside the image are stored as 0 (they are only ever needed by outputs outside the valid range).
- Also copy the $K \times K$ filter into a `(5, 5)` shared array.
- `cuda.syncthreads()`, then each thread with a valid output computes its sum from shared memory and writes `y`.

## Inputs and output
- `x`: `float32`, shape `(H, W)`; `w`: `float32`, shape `(K, K)` with $1 \le K \le 5$ and $K \le H, W$.
- Return `y`: `float32`, shape `(H-K+1, W-K+1)`. Compared with relative and absolute tolerance $10^{-5}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda, float32

TILE, KMAX = 16, 5
PATCH = TILE + KMAX - 1   # 20: shared tile size for the largest filter

def conv2d_valid(x, w):
    """x: float32 (H, W), w: float32 (K, K), K <= 5. Returns float32 (H-K+1, W-K+1)."""
    H, W = x.shape
    K = w.shape[0]
    Ho, Wo = H - K + 1, W - K + 1
    d_x = cuda.to_device(np.ascontiguousarray(x))
    d_w = cuda.to_device(np.ascontiguousarray(w))
    d_y = cuda.device_array((Ho, Wo), dtype=np.float32)
    grid = ((Wo + TILE - 1) // TILE, (Ho + TILE - 1) // TILE)
    conv2d_kernel[grid, (TILE, TILE)](d_x, d_w, d_y)
    return d_y.copy_to_host()

@cuda.jit
def conv2d_kernel(x, w, y):
    patch = cuda.shared.array((PATCH, PATCH), float32)
    wsh = cuda.shared.array((KMAX, KMAX), float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    # TODO 1: cooperative load of the (TILE + K - 1)^2 patch (zeros outside the image)
    # TODO 2: load the filter into wsh
    # TODO 3: barrier, then compute y[row, col] from shared memory
    pass
'''

SOLUTION = '''import numpy as np
from numba import cuda, float32

TILE, KMAX = 16, 5
PATCH = TILE + KMAX - 1   # 20: shared tile size for the largest filter

def conv2d_valid(x, w):
    H, W = x.shape
    K = w.shape[0]
    Ho, Wo = H - K + 1, W - K + 1
    d_x = cuda.to_device(np.ascontiguousarray(x))
    d_w = cuda.to_device(np.ascontiguousarray(w))
    d_y = cuda.device_array((Ho, Wo), dtype=np.float32)
    grid = ((Wo + TILE - 1) // TILE, (Ho + TILE - 1) // TILE)
    conv2d_kernel[grid, (TILE, TILE)](d_x, d_w, d_y)
    return d_y.copy_to_host()

@cuda.jit
def conv2d_kernel(x, w, y):
    patch = cuda.shared.array((PATCH, PATCH), float32)
    wsh = cuda.shared.array((KMAX, KMAX), float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    t = ty * TILE + tx                         # flat thread id, 0..255
    H, W = x.shape
    K = w.shape[0]
    side = TILE + K - 1                        # patch side actually used
    r0 = cuda.blockIdx.y * TILE                # top-left input pixel of this tile
    c0 = cuda.blockIdx.x * TILE

    # 1. Cooperative load: 256 threads stride over side*side patch entries.
    for idx in range(t, side * side, TILE * TILE):
        pr, pc = idx // side, idx % side
        gr, gc = r0 + pr, c0 + pc
        patch[pr, pc] = x[gr, gc] if gr < H and gc < W else float32(0.0)
    # 2. The filter is read by every thread: stage it too.
    if t < K * K:
        wsh[t // K, t % K] = w[t // K, t % K]
    cuda.syncthreads()                         # the patch is complete before anyone reads it

    # 3. Compute from shared memory only.
    row, col = r0 + ty, c0 + tx
    if row < y.shape[0] and col < y.shape[1]:
        acc = float32(0.0)
        for a in range(K):
            for b in range(K):
                acc += wsh[a, b] * patch[ty + a, tx + b]
        y[row, col] = acc
'''

HINTS = [
    'The tile at block `(bx, by)` starts at input pixel `(by*16, bx*16)`. Output `(ty, tx)` of the tile needs `patch[ty + a, tx + b]` for `a, b < K`, so the patch is `16 + K - 1` on each side.',
    'Cooperative load: `t = ty*16 + tx`; `for idx in range(t, side*side, 256): pr, pc = divmod(idx, side)` and copy `x[r0+pr, c0+pc]` (or 0 if outside the image) into `patch[pr, pc]`.',
    'Load the filter with the first $K^2$ threads, then one `cuda.syncthreads()` before computing. Put the bounds check on the output **after** the barrier, so every thread reaches it.',
]

_r = np.random.default_rng(52)
_f = lambda *s: _r.standard_normal(s).astype(np.float32)
TESTS = [
    (np.arange(16, dtype=np.float32).reshape(4, 4), np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)),
    (np.ones((5, 6), dtype=np.float32), np.ones((3, 3), dtype=np.float32)),                 # all 9s
    (_f(7, 9), np.array([[2.0]], dtype=np.float32)),                                         # K = 1
    (_f(5, 5), _f(5, 5)),                                                                    # one output
    (_f(20, 20), _f(5, 5)),                                                                  # tile boundary: Ho = Wo = 16
    (_f(21, 37), _f(3, 3)),                                                                  # partial tiles
    (_f(40, 34), _f(4, 4)),                                                                  # even K
]

EXPLANATION = r"""
## Why it works
Every output of the tile reads only inside the tile's patch, and the patch is loaded completely (the barrier guarantees it) before any thread computes. The cooperative loop covers each of the `side * side` patch entries exactly once, whatever $K$ is. Zero-filled pixels outside the image are only read by outputs that fail the final bounds check, so they never affect the result.

## How much it saves
Without tiling, each block fetches $256 K^2$ input values from global memory. With tiling it fetches $(16+K-1)^2$: for $K = 5$, 400 instead of 6,400, a 16x reduction in global loads. The arithmetic is unchanged, so this moves the kernel from memory-bound towards compute-bound. The halo is the overhead: $(16+K-1)^2 / 16^2 = 1.56$ for $K = 5$, which is why larger tiles (32 × 32, or each thread computing several outputs) pay off for larger filters.

## Shared-memory banks
Shared memory is split into 32 banks of 4-byte words; a warp's accesses to distinct addresses in the same bank serialise. In the compute loop, a warp (16 threads of row `ty` and 16 of row `ty+1`) reads `patch[ty + a, tx + b]`: 16 consecutive words from each of two patch rows that are 20 words apart. Relative to some base bank, the first half hits banks $0..15$ and the second $20..35$, which wraps onto banks $0..3$: four lanes see a 2-way conflict. A row stride congruent to 16 mod 32 (padding the patch to 48 columns) would make the halves disjoint; here the cost is small enough to ignore. The filter read `wsh[a, b]` is the same address for every thread: a broadcast, no conflict.

## Common bugs
- Sizing the shared array with the runtime $K$: shared array shapes must be compile-time constants, hence the `KMAX` bound.
- Returning early for out-of-range outputs **before** the barrier: those threads also have load work to do, and they must reach the barrier.
- Treating `(tx, ty)` as `(row, col)`: x is the fast (column) dimension.

## Interview follow-ups
- How does cuDNN turn convolution into a matrix multiplication (im2col / implicit GEMM), and why does that map well to tensor cores?
- With $C_{in}$ input channels, how would you tile over channels so the patch still fits in 48 KB of shared memory?
"""
