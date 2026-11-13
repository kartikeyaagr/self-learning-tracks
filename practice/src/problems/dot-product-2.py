import numpy as np

META = dict(title='Dot product: a two-stage reduction', tags=['GPU Programming', 'Linear Algebra'], diff='Medium',
            runtime='cuda', tol=[1e-4, 1e-4], lessons=['ieng:13.04', 'ieng:13.03'],
            related=['sum-of-array', 'l2-normalization', 'dot-product'])

STATEMENT = r"""
A reduction turns $n$ values into one. Threads in **different blocks** cannot synchronise inside a kernel, so a grid-wide reduction is split into stages:

1. **Stage 1** (`BLOCKS = 8` blocks of `TPB = 64` threads): each thread accumulates $a_i b_i$ over a grid-stride loop, each block tree-reduces its 64 partials in shared memory, and thread 0 writes the block's total to `partials[blockIdx.x]`.
2. **Stage 2** (1 block of 64 threads): reduce the 8 partials the same way and write the final value to a 1-element device array.

Kernel launches on the same stream run in order, so stage 2 sees everything stage 1 wrote. No host round trip in between.

## Task
Write `dot(a, b)` for two `float32` arrays of length `n` (`n` may be 0) and return $\sum_i a_i b_i$ as a Python `float` (0.0 for empty inputs).

- Accumulate in `float32`. Summation order differs between implementations, so results are compared with relative and absolute tolerance $10^{-4}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda, float32

BLOCKS, TPB = 8, 64

def dot(a, b):
    """a, b: float32 (n,). Returns sum(a * b) as a Python float."""
    raise NotImplementedError

@cuda.jit
def partial_dot_kernel(a, b, partials):
    pass

@cuda.jit
def final_sum_kernel(partials, out):
    pass
'''

SOLUTION = '''import numpy as np
from numba import cuda, float32

BLOCKS, TPB = 8, 64

def dot(a, b):
    d_a, d_b = cuda.to_device(a), cuda.to_device(b)
    d_partials = cuda.device_array(BLOCKS, dtype=np.float32)
    d_out = cuda.device_array(1, dtype=np.float32)
    partial_dot_kernel[BLOCKS, TPB](d_a, d_b, d_partials)  # stage 1: one partial per block
    final_sum_kernel[1, TPB](d_partials, d_out)            # stage 2: runs after stage 1 (same stream)
    return float(d_out.copy_to_host()[0])

@cuda.jit(device=True)
def block_sum(buf, tid):
    """Tree-reduce buf[0:TPB] into buf[0]. Every thread of the block must call this."""
    cuda.syncthreads()
    step = TPB // 2
    while step > 0:
        if tid < step:
            buf[tid] += buf[tid + step]
        cuda.syncthreads()
        step //= 2

@cuda.jit
def partial_dot_kernel(a, b, partials):
    tid = cuda.threadIdx.x
    buf = cuda.shared.array(TPB, float32)
    acc = float32(0.0)
    for i in range(cuda.grid(1), a.shape[0], cuda.gridsize(1)):
        acc += a[i] * b[i]
    buf[tid] = acc
    block_sum(buf, tid)
    if tid == 0:
        partials[cuda.blockIdx.x] = buf[0]

@cuda.jit
def final_sum_kernel(partials, out):
    tid = cuda.threadIdx.x
    buf = cuda.shared.array(TPB, float32)
    acc = float32(0.0)
    for i in range(tid, partials.shape[0], TPB):
        acc += partials[i]
    buf[tid] = acc
    block_sum(buf, tid)
    if tid == 0:
        out[0] = buf[0]
'''

HINTS = [
    'Stage 1 is three steps per block: a grid-stride loop into a local `acc`, `buf[tid] = acc`, then a tree reduction. Thread 0 writes `partials[blockIdx.x]`.',
    'Both stages need the same block-wide sum. Put it in a `@cuda.jit(device=True)` helper that takes the shared array and `tid`; shared arrays can be passed to device functions.',
    'Launch the two kernels back to back on device arrays, then copy the single result out with `float(d_out.copy_to_host()[0])`. With `n = 0` every accumulator stays 0, so no special case is needed.',
]

_r = np.random.default_rng(15)
_f = lambda n: _r.standard_normal(n).astype(np.float32)
TESTS = [
    (np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)),
    (np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)),
    (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)),
    (np.ones(512, dtype=np.float32), np.full(512, 0.5, dtype=np.float32)),   # exactly one grid pass
    (np.ones(1000, dtype=np.float32), np.arange(1000, dtype=np.float32)),
    (_f(777), _f(777)),
    (_f(3000), _f(3000)),
]

EXPLANATION = r"""
## Why it works
Within a block, `cuda.syncthreads()` orders shared-memory writes and reads, so the tree reduction is safe. Across blocks there is no such barrier inside a kernel, but there is one **between** kernels: a launch on the same stream starts only after the previous one finishes. Splitting the reduction at the block boundary turns an impossible grid-wide barrier into a kernel boundary.

## Cost
Stage 1 reads $8n$ bytes and does $2n$ FLOPs: bandwidth-bound, like every reduction. Stage 2 is tiny; its main cost is the launch overhead (a few µs). That is why the grid of stage 1 is kept small and fixed (a few blocks per SM in practice): it bounds the number of partials so stage 2 always fits in one block.

## Floating-point order
The tree adds values in a different order from NumPy's pairwise sum, so the last few bits differ. The tree (and pairwise) order has error growth $O(\log n)$ rather than the $O(n)$ of a naive running sum, which is a nice side benefit of parallel reductions.

## Alternatives
- **Atomics**: each block does `cuda.atomic.add(out, 0, block_total)`. One kernel, but floating-point atomics finish in a nondeterministic order, so results can change run to run in the last bits. Many training frameworks offer "deterministic mode" partly to avoid this.
- **Warp shuffles** for the last 32 lanes, and cooperative groups' grid sync for single-kernel reductions.

## Common bugs
- `cuda.syncthreads()` inside the `if tid < step` branch.
- Forgetting the barrier between `buf[tid] = acc` and the first reduction step.
- Sizing `partials` by `TPB` instead of `BLOCKS`, or reading uninitialised entries in stage 2.

## Interview follow-ups
- How many kernel launches does a three-stage reduction need for $n = 10^9$ with 1024-thread blocks and no grid-stride loop? Why is the grid-stride version better?
- Why does `torch.use_deterministic_algorithms(True)` make some operations slower?
"""
