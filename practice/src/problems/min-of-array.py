import numpy as np

META = dict(title='Array min by repeated kernel launches', tags=['GPU Programming'], diff='Easy',
            runtime='cuda', lessons=['ieng:13.04', 'ieng:13.03'],
            related=['max-of-array', 'sum-of-array', 'argmin'])

STATEMENT = r"""
A reduction can also skip atomics and grid-stride loops altogether. Each launch shrinks the array by a large factor, and the host keeps launching until one value is left:

```
n = 5000  ->  ceil(5000 / 128) = 40 partials  ->  1 partial  ->  done
```

Write `array_min(x)` that returns the smallest element of a 1-D `float32` array this way.

## The kernel: `min_pass(src, dst)`
- Blocks of **64 threads**, and each block covers **128** consecutive elements of `src`: block `b` owns indices `[128 b, 128 b + 128)`.
- **First step during the load**: thread `t` reads elements `base + t` and `base + t + 64` (where they exist) and stores their min in shared memory. This halves the number of threads that sit idle in the first tree step.
- A tree reduction over the 64 shared values; thread 0 writes the block's min to `dst[blockIdx.x]`.
- Indices past the end of `src` contribute $+\infty$, the identity of min.

## The host: `array_min(x)`
- Launch `min_pass` with $\lceil n/128 \rceil$ blocks, feed its output into the next launch, and stop when one element remains. Keep the intermediate arrays on the device: no copies to the host between passes.
- `x`: `float32`, shape `(n,)` with `n >= 1`, no NaNs. Return the minimum as a Python `float`. For `n == 1`, no launch is needed.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda, float32

TPB = 64            # threads per block
PER_BLOCK = 2 * TPB  # elements per block

def array_min(x):
    """x: float32 (n,), n >= 1. Returns min(x) as a Python float."""
    d = cuda.to_device(x)
    blocks = (x.shape[0] + PER_BLOCK - 1) // PER_BLOCK
    out = cuda.device_array(blocks, dtype=np.float32)
    min_pass[blocks, TPB](d, out)
    return float(out.copy_to_host()[0])   # TODO: only one pass; block 0's min is not the global min

@cuda.jit
def min_pass(src, dst):
    pass  # TODO
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda, float32

TPB = 64            # threads per block (a power of two)
PER_BLOCK = 2 * TPB  # each thread loads two elements

def array_min(x):
    d = cuda.to_device(x)
    n = x.shape[0]
    while n > 1:                                   # each pass shrinks n by a factor of 128
        blocks = (n + PER_BLOCK - 1) // PER_BLOCK
        out = cuda.device_array(blocks, dtype=np.float32)
        min_pass[blocks, TPB](d, out)              # launches on one stream run in order
        d, n = out, blocks
    return float(d.copy_to_host()[0])

@cuda.jit
def min_pass(src, dst):
    tid = cuda.threadIdx.x
    base = cuda.blockIdx.x * PER_BLOCK
    buf = cuda.shared.array(TPB, float32)

    # Load two elements per thread and combine them immediately (+inf past the end).
    m = math.inf
    i = base + tid
    if i < src.shape[0]:
        m = src[i]
    if i + TPB < src.shape[0]:
        m = min(m, src[i + TPB])
    buf[tid] = m
    cuda.syncthreads()

    step = TPB // 2
    while step > 0:
        if tid < step:
            buf[tid] = min(buf[tid], buf[tid + step])
        cuda.syncthreads()
        step //= 2
    if tid == 0:
        dst[cuda.blockIdx.x] = buf[0]
'''

HINTS = [
    'The host loop is `while n > 1:` launch $\\lceil n/128 \\rceil$ blocks into a fresh device array, then make that array the new input. Stop when `n == 1` and copy that single value back.',
    'In the kernel, `i = blockIdx.x * 128 + tid`; take `src[i]` and `src[i + 64]` when each is in range, starting from `math.inf`, and store their min in `buf[tid]`.',
    'Finish with the usual tree: `step = 32, 16, ..., 1`, `if tid < step: buf[tid] = min(buf[tid], buf[tid + step])`, `cuda.syncthreads()` after the `if`, and thread 0 writes `dst[blockIdx.x]`.',
]

_r = np.random.default_rng(32)
_late = np.ones(3000, dtype=np.float32)
_late[2999] = -1.0                              # the min lives in the last, partial block
TESTS = [
    (np.array([4.0, 1.5, 9.0, -2.0, 3.0], dtype=np.float32),),
    (np.array([7.0], dtype=np.float32),),                        # n = 1: no launch
    (np.arange(128, 256, dtype=np.float32),),                    # exactly one full block
    (np.full(129, 5.0, dtype=np.float32),),                      # one element spills into a 2nd block
    (_late,),
    (np.array([np.inf, 3.0, -np.inf, 0.0], dtype=np.float32),),
    (_r.standard_normal(2500).astype(np.float32),),
    ((_r.standard_normal(4096) * 50 + 20).astype(np.float32),),
]

EXPLANATION = r"""
## Why it works
Each pass replaces $n$ values by $\lceil n/128 \rceil$ block minima, and min of the block minima is the global min. Kernel launches on the same stream run in order, so pass $k+1$ always sees the complete output of pass $k$: the kernel boundary acts as the grid-wide barrier that blocks cannot perform among themselves.

## Cost
For $n$ elements there are $\lceil \log_{128} n \rceil$ launches (4096 elements: 2 launches; $10^9$: 5). Total memory traffic is $4n(1 + 1/128 + \dots) \approx 4.03n$ bytes, so the extra passes are nearly free in bandwidth, but each launch costs a few microseconds of overhead. That is why production reductions prefer a fixed-size grid with a grid-stride loop and **exactly two** launches (as in the dot-product problem), or one launch with atomics.

## Combining during the load
In a plain tree, half the threads go idle after the first step. Loading two elements per thread and combining them before the tree does that first step "for free" in registers, so a 64-thread block reduces 128 elements. Extending the idea (each thread folds many elements before the tree) is exactly what a grid-stride loop does.

## Common bugs
- Copying to the host between passes, or reallocating on the host: correct but slow.
- Forgetting the `i + 64 < n` check in the last block, which reads past the end (the simulator raises an IndexError; real hardware silently reads garbage).
- Returning `dst[0]` after one pass, which is only block 0's min.

## Interview follow-ups
- Why is a warp-shuffle reduction for the last 32 lanes faster than shared memory, and why does it need no `__syncthreads`?
- With 64-thread blocks, how many blocks per SM would you need to keep the GPU busy, and how does that interact with occupancy limits?
"""
