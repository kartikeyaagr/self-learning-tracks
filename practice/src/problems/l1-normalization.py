import numpy as np

META = dict(title='Column-wise L1 normalisation with coalesced threads', tags=['GPU Programming'], diff='Medium',
            runtime='cuda', tol=[1e-5, 1e-7], lessons=['ieng:13.03', 'ieng:13.01'],
            related=['l2-normalization', 'matrix-addition', 'softmax-2'])

STATEMENT = r"""
Normalise every **column** of a matrix to unit L1 norm:

$$y_{ij} = \frac{x_{ij}}{\max\!\left(\sum_{r} |x_{rj}|,\ \varepsilon\right)}, \qquad \varepsilon = 10^{-12}$$

The clamp keeps an all-zero column at zero instead of producing NaN.

## The mapping, and why it is the fast one
Use **one thread per column**: blocks of 64 threads, grid $\lceil \text{cols}/64 \rceil$, `j = cuda.grid(1)`. Each thread walks down its column twice (once to sum, once to write).

That sounds like a strided access pattern, but look at what a **warp** does at each loop step: 32 threads with consecutive `j` read `x[r, j]` for the same `r`, which are 32 consecutive floats in a row-major array. One 128-byte transaction serves the whole warp. The opposite mapping (one thread per row, looping over columns) would have the 32 threads of a warp read 32 addresses `cols` floats apart.

No shared memory and no synchronisation are needed: columns are independent.

## Inputs and output
- `x`: `float32`, shape `(rows, cols)`, both at least 1, row-major (C-contiguous).
- Return `y` as `float32` with the same shape. Accumulate the column sum in `float32`.
- Compared with relative tolerance $10^{-5}$, absolute $10^{-7}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda

TPB = 64
EPS = 1e-12

def l1_normalize_columns(x):
    """x: float32 (rows, cols). Returns float32 (rows, cols) with unit-L1 columns."""
    rows, cols = x.shape
    d_x = cuda.to_device(np.ascontiguousarray(x))
    d_y = cuda.device_array((rows, cols), dtype=np.float32)
    blocks = (rows + TPB - 1) // TPB
    l1_kernel[blocks, TPB](d_x, d_y)
    return d_y.copy_to_host()

@cuda.jit
def l1_kernel(x, y):
    r = cuda.grid(1)                   # TODO: one thread per ROW is the slow mapping, and it
    if r < x.shape[0]:                 # normalises rows, not columns
        s = 0.0
        for j in range(x.shape[1]):
            s += x[r, j]               # TODO: absolute value, epsilon clamp
        for j in range(x.shape[1]):
            y[r, j] = x[r, j] / s
'''

SOLUTION = '''import numpy as np
from numba import cuda, float32

TPB = 64
EPS = 1e-12

def l1_normalize_columns(x):
    rows, cols = x.shape
    d_x = cuda.to_device(np.ascontiguousarray(x))
    d_y = cuda.device_array((rows, cols), dtype=np.float32)
    blocks = (cols + TPB - 1) // TPB          # one thread per column
    l1_kernel[blocks, TPB](d_x, d_y)
    return d_y.copy_to_host()

@cuda.jit
def l1_kernel(x, y):
    j = cuda.grid(1)
    if j >= x.shape[1]:
        return
    s = float32(0.0)
    for r in range(x.shape[0]):               # a warp reads 32 adjacent floats of row r: coalesced
        s += abs(x[r, j])
    inv = float32(1.0) / max(s, float32(EPS))  # all-zero column -> 0 / eps = 0, not NaN
    for r in range(x.shape[0]):
        y[r, j] = x[r, j] * inv
'''

HINTS = [
    'Size the grid by `cols`, not `rows`, and let `j = cuda.grid(1)` pick the column. Return early when `j >= cols`.',
    'First loop: `s += abs(x[r, j])` over all rows. Then `inv = 1 / max(s, EPS)` and a second loop writing `y[r, j] = x[r, j] * inv`.',
    'Check the access pattern by fixing `r` and varying `j` across a warp: consecutive addresses. That is what "coalesced" means, even though each thread individually strides by `cols`.',
]

_r = np.random.default_rng(41)
_zero_col = _r.standard_normal((5, 4)).astype(np.float32)
_zero_col[:, 2] = 0.0
TESTS = [
    (np.array([[1.0, -2.0], [3.0, 2.0]], dtype=np.float32),),
    (np.array([[1.0, 2.0, 3.0]], dtype=np.float32),),                  # one row: every entry becomes +-1
    (_zero_col,),                                                       # zero column stays zero
    (np.full((7, 1), -0.5, dtype=np.float32),),                        # one column
    (_r.standard_normal((3, 130)).astype(np.float32),),                # cols > 64: 3 blocks
    ((_r.standard_normal((40, 20)) * 1e-3).astype(np.float32),),
    ((_r.standard_normal((64, 64)) * 100).astype(np.float32),),
]

EXPLANATION = r"""
## Why it works
Each column's norm depends only on that column, so one thread can own it end to end. The L1 norm is $\sum_r |x_{rj}|$; dividing by it makes the absolute values of the column sum to 1. The $\varepsilon$ clamp only matters for a column that is exactly zero, where $0/\varepsilon = 0$.

## Coalescing is about warps, not threads
The memory system serves a warp's 32 loads together. What matters is the set of addresses one warp touches **in the same instruction**. Here, at loop step `r`, they are `x[r, j0 .. j0+31]`: contiguous, one transaction. Per thread the access is strided by `cols`, and that is fine. The row-per-thread mapping in the starter is the reverse: contiguous per thread, scattered per warp, up to 32 transactions per load. On real hardware that is often a 5-10x difference for a bandwidth-bound kernel like this one.

## Cost and limits
Two reads and one write of the matrix: $12 \cdot \text{rows} \cdot \text{cols}$ bytes. Parallelism is only `cols` threads, which is too few when the matrix is tall and narrow (say $10^6 \times 16$). Then you would split the rows across blocks: each block computes partial column sums, combines them (atomics or a second kernel), and a final pass divides.

## Common bugs
- Forgetting `abs`: columns with mixed signs get a too-small (or zero) norm.
- `x / s` without the clamp: NaN for zero columns.
- Grid sized by rows: threads past `cols` index out of range, or columns are missed.

## Interview follow-ups
- For row-wise L1 normalisation of a short-and-wide matrix, which mapping would you use instead, and why does it need a reduction across threads?
- How would you transpose-free normalise columns of a column-major (Fortran-order) array?
"""
