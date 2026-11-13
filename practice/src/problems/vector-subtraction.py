import numpy as np

META = dict(title='Vector subtraction with a grid-stride loop', tags=['GPU Programming'], diff='Easy', runtime='cuda',
            lessons=['ieng:13.01', 'ieng:13.04'], related=['vector-addition', 'relu'])

STATEMENT = r"""
In the previous problem the grid grew with the input: one thread per element. Production kernels usually launch a **fixed** grid sized to the GPU (a few blocks per streaming multiprocessor) and let each thread loop over the array with a stride equal to the total number of threads. This is the **grid-stride loop**.

Write `vector_sub(a, b)` returning $\text{out}_i = a_i - b_i$.

## Inputs
- `a`, `b`: 1-D `float32` arrays of the same length `n` (`n` may be 0 or much larger than the grid).

## Launch configuration (fixed)
- Exactly **4 blocks of 64 threads** (256 threads total), whatever `n` is.
- Thread `t = cuda.grid(1)` handles elements `t, t + S, t + 2S, ...` while they are `< n`, where `S = cuda.gridsize(1)` is the total thread count.

## Output
A `float32` array of shape `(n,)`. Because the grid no longer depends on `n`, the `n == 0` case needs no special handling: every thread's loop simply runs zero times.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda

BLOCKS, TPB = 4, 64

def vector_sub(a, b):
    """a, b: float32 (n,). Returns a - b as float32 (n,), using a fixed 4 x 64 grid."""
    d_a, d_b = cuda.to_device(a), cuda.to_device(b)
    d_out = cuda.device_array(a.shape[0], dtype=np.float32)
    sub_kernel[BLOCKS, TPB](d_a, d_b, d_out)
    return d_out.copy_to_host()

@cuda.jit
def sub_kernel(a, b, out):
    i = cuda.grid(1)
    # TODO: this only covers the first 256 elements. Loop with a stride of cuda.gridsize(1).
    if i < out.shape[0]:
        out[i] = a[i] - b[i]
'''

SOLUTION = '''import numpy as np
from numba import cuda

BLOCKS, TPB = 4, 64

def vector_sub(a, b):
    d_a, d_b = cuda.to_device(a), cuda.to_device(b)
    d_out = cuda.device_array(a.shape[0], dtype=np.float32)
    sub_kernel[BLOCKS, TPB](d_a, d_b, d_out)    # grid size is independent of n
    return d_out.copy_to_host()

@cuda.jit
def sub_kernel(a, b, out):
    start = cuda.grid(1)                        # this thread's first element
    stride = cuda.gridsize(1)                   # total threads in the grid = 256
    for i in range(start, out.shape[0], stride):
        out[i] = a[i] - b[i]
'''

HINTS = [
    '`cuda.gridsize(1)` returns `blockDim.x * gridDim.x`, the number of threads in the whole grid.',
    'Replace the `if` with `for i in range(start, n, stride):`. The loop bound doubles as the bounds check.',
    'On iteration $k$, thread $t$ touches element $t + kS$. At each step, the 32 threads of a warp still read 32 consecutive floats, so access stays coalesced.',
]

_r = np.random.default_rng(2)
_f = lambda n: _r.standard_normal(n).astype(np.float32)
TESTS = [
    (np.array([5.0, 3.0, 1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)),
    (np.array([0.5, -2.0], dtype=np.float32), np.array([-0.5, -2.0], dtype=np.float32)),
    (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)),
    (_f(256), _f(256)),                 # exactly one pass of the grid
    (_f(257), _f(257)),                 # one element needs a second pass
    (_f(1500), _f(1500)),               # about six passes
    (np.arange(700, dtype=np.float32), np.arange(700, dtype=np.float32)[::-1].copy()),
]

EXPLANATION = r"""
## Why it works
The grid has $S = 256$ threads. Thread $t$ visits $t, t+S, t+2S, \dots$. Every index $i < n$ is written exactly once, by thread $i \bmod S$ on iteration $\lfloor i / S \rfloor$, so no element is missed or written twice.

## Why use a fixed grid
- **Decoupling**: the same launch works for 10 or 10 million elements, and for `n = 0` without a special case.
- **Amortised setup**: work done once per thread (loading constants, computing a base pointer) is reused across many elements.
- **Tuning knob**: you size the grid to the hardware, typically a small multiple of the number of SMs, rather than to the data.
- Debugging is easier too: run with `[1, 1]` and the single thread processes the whole array serially.

## Performance
Coalescing is preserved: on every iteration, adjacent threads touch adjacent addresses. Too small a grid, though, leaves SMs idle and cannot hide memory latency; too large brings back the one-thread-per-element case.

## Common bugs
- Stride of `cuda.blockDim.x` instead of the whole grid: blocks overlap and redo each other's work.
- Looping `for k in range(n // S)` and indexing `t + k*S`: drops the tail when `n` is not a multiple of `S`.
- Strided chunking (thread $t$ takes a contiguous range `[t*c, (t+1)*c)`): correct, but neighbouring threads then read addresses `c` apart, which breaks coalescing.

## Interview follow-ups
- How would you pick the number of blocks on a real GPU? (Query the SM count, e.g. `cuda.get_current_device().MULTIPROCESSOR_COUNT`, and launch a few blocks per SM.)
- When is one thread per element still better? (When each element is heavy and independent, and the grid is large enough to fill the GPU anyway.)
"""
