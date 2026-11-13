import numpy as np

META = dict(title='Vector addition: your first CUDA kernel', tags=['GPU Programming'], diff='Easy', runtime='cuda',
            lessons=['ieng:13.01', 'ieng:13.04'], related=['vector-subtraction', 'matrix-addition'])

STATEMENT = r"""
A CUDA kernel is a function that thousands of threads run at once. Each thread works out **which element it owns** from its block and thread coordinates, then does a tiny piece of work. Vector addition is the "hello world" of this model:

$$\text{out}_i = a_i + b_i, \qquad i = 0, \dots, n-1$$

Write a Numba CUDA kernel and a host function `vector_add(a, b)`.

## Inputs
- `a`, `b`: 1-D NumPy arrays of dtype `float32` with the same length `n` (`n` may be 0).

## What the host function must do
1. Copy `a` and `b` to the GPU (`cuda.to_device`) and allocate the output there (`cuda.device_array`).
2. Launch the kernel with **128 threads per block** and just enough blocks to cover `n`: $\lceil n / 128 \rceil$.
3. Copy the result back and return it as a `float32` NumPy array of shape `(n,)`.

## Indexing
The global index of a thread in a 1-D launch is

$$i = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$$

which `cuda.grid(1)` computes for you. The last block is usually only partly used, so threads with $i \ge n$ must do nothing.

## Edge case
A launch with zero blocks is an error. If `n == 0`, return an empty `float32` array without launching.

## Runtime
Runs on a GPU in Colab (**Runtime → Change runtime type → T4 GPU**). The tests here run on Numba's CUDA simulator, which executes every thread on the CPU, so sizes are small.
"""

STARTER = '''import numpy as np
from numba import cuda

TPB = 128  # threads per block

def vector_add(a, b):
    """a, b: float32 (n,). Returns a + b as float32 (n,)."""
    n = a.shape[0]
    out = np.zeros(n, dtype=np.float32)
    # TODO: copy a and b to the device, launch add_kernel[blocks, TPB](...), copy the result back
    return out

@cuda.jit
def add_kernel(a, b, out):
    # TODO: compute this thread's global index, guard against i >= n, write out[i] = a[i] + b[i]
    pass
'''

SOLUTION = '''import numpy as np
from numba import cuda

TPB = 128  # threads per block

def vector_add(a, b):
    n = a.shape[0]
    if n == 0:                    # a zero-block launch is invalid
        return np.zeros(0, dtype=np.float32)
    d_a, d_b = cuda.to_device(a), cuda.to_device(b)
    d_out = cuda.device_array(n, dtype=np.float32)
    blocks = (n + TPB - 1) // TPB # ceil(n / TPB) without floats
    add_kernel[blocks, TPB](d_a, d_b, d_out)
    return d_out.copy_to_host()

@cuda.jit
def add_kernel(a, b, out):
    i = cuda.grid(1)              # blockIdx.x * blockDim.x + threadIdx.x
    if i < out.shape[0]:          # the last block overhangs the array
        out[i] = a[i] + b[i]
'''

HINTS = [
    'Inside the kernel, `i = cuda.grid(1)` gives the global thread index. Every thread handles exactly one element.',
    'The grid has `blocks * 128` threads, which is usually more than `n`. Wrap the write in `if i < out.shape[0]:`.',
    'Host side: `blocks = (n + TPB - 1) // TPB`, then `add_kernel[blocks, TPB](d_a, d_b, d_out)` and `d_out.copy_to_host()`. Return early when `n == 0`.',
]

_r = np.random.default_rng(1)
_f = lambda *s: _r.standard_normal(s).astype(np.float32)
TESTS = [
    (np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([10.0, 20.0, 30.0], dtype=np.float32)),
    (np.array([-1.5, 0.0, 2.25, 4.0], dtype=np.float32), np.array([1.5, -3.0, 0.75, -4.0], dtype=np.float32)),
    (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)),
    (np.array([7.0], dtype=np.float32), np.array([-2.0], dtype=np.float32)),
    (_f(128), _f(128)),                 # exactly one full block
    (_f(129), _f(129)),                 # one element spills into a second block
    (_f(1000), _f(1000)),
    (np.full(300, 1e30, dtype=np.float32), np.full(300, -1e30, dtype=np.float32)),
]

EXPLANATION = r"""
## Why it works
The launch `add_kernel[blocks, TPB]` creates `blocks × 128` threads. `cuda.grid(1)` gives each a unique index, so every element in `[0, n)` is written by exactly one thread and no two threads touch the same address. No synchronisation is needed.

The bounds check matters because the grid size is rounded up to a whole number of blocks. With `n = 129`, the second block has 127 threads with nothing to do. Without the guard they write past the end of `out`, which on a real GPU silently corrupts memory or raises an illegal-address error.

## Performance
Each element needs 8 bytes read and 4 bytes written for one addition. That is far below what the arithmetic units can do, so the kernel is **memory-bandwidth-bound**: its speed is set by how fast the GPU moves bytes, not how fast it adds. Neighbouring threads in a warp read neighbouring addresses, so loads are **coalesced** into a few wide memory transactions, the best case for bandwidth.

For tiny arrays like these, the host↔device copies and the launch overhead (a few microseconds) dominate. A GPU only pays off when data stays on the device across many kernels.

## Common bugs
- Missing `if i < n`: out-of-bounds writes in the last block.
- `blocks = n // TPB`: drops the tail; with `n = 129` the last element is never written.
- Launching with `n = 0`: Numba rejects a grid of zero blocks.
- Passing NumPy arrays straight to the kernel works, but Numba then copies **every** argument back to the host after the launch. Explicit `to_device` / `copy_to_host` keeps transfers under your control.

## Interview follow-ups
- Why are block sizes usually a multiple of 32? (Threads are scheduled in warps of 32; a block of 100 wastes 28 lanes in its last warp.)
- How would you change the kernel so a fixed, small grid can process an array of any length? (A grid-stride loop, the next problem.)
"""
