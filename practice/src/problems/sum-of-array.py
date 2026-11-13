import numpy as np

META = dict(title='Integer sum with atomics', tags=['GPU Programming'], diff='Easy', runtime='cuda',
            lessons=['ieng:13.04'], related=['dot-product-2', 'min-of-array'])

STATEMENT = r"""
An **atomic** operation is a read-modify-write that the hardware performs indivisibly: if 1,000 threads each run `cuda.atomic.add(total, 0, v)`, no update is lost. Atomics make a reduction a single kernel.

Write `array_sum(x)` that sums a 1-D `int32` array with one kernel launch.

## Requirements
- Fixed grid: **4 blocks of 64 threads**, grid-stride loop.
- Each thread first sums its own elements into a **local** variable, then issues **one** `cuda.atomic.add` into a 1-element `int64` device array that starts at 0 (`cuda.to_device(np.zeros(1, np.int64))`).
- Accumulate in `int64`: the total can exceed the `int32` range even when every element fits.
- Return the total as a Python `int`. An empty array sums to 0.

Integer addition is associative and exact, so the result is the same whatever order the atomics land in.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda

BLOCKS, TPB = 4, 64

def array_sum(x):
    """x: int32 (n,). Returns the sum as a Python int."""
    d_x = cuda.to_device(x)
    d_total = cuda.to_device(np.zeros(1, dtype=np.int64))
    sum_kernel[BLOCKS, TPB](d_x, d_total)
    return int(d_total.copy_to_host()[0])

@cuda.jit
def sum_kernel(x, total):
    i = cuda.grid(1)
    if i < x.shape[0]:
        total[0] += x[i]   # TODO: a data race; and only the first 256 elements are visited
'''

SOLUTION = '''import numpy as np
from numba import cuda, int64

BLOCKS, TPB = 4, 64

def array_sum(x):
    d_x = cuda.to_device(x)
    d_total = cuda.to_device(np.zeros(1, dtype=np.int64))   # atomics need a zeroed target
    sum_kernel[BLOCKS, TPB](d_x, d_total)
    return int(d_total.copy_to_host()[0])

@cuda.jit
def sum_kernel(x, total):
    acc = int64(0)                                    # widen before adding: avoids int32 overflow
    for i in range(cuda.grid(1), x.shape[0], cuda.gridsize(1)):
        acc += x[i]
    cuda.atomic.add(total, 0, acc)                    # one atomic per thread, not per element
'''

HINTS = [
    '`total[0] += v` is a read, an add and a write. Two threads can read the same old value and one update is lost. `cuda.atomic.add(total, 0, v)` makes it indivisible.',
    'Atomics on one address serialise. Sum your strided elements into a local `acc` first, then do one atomic per thread: 256 atomics instead of $n$.',
    'Start the accumulator as `int64(0)` (from `numba import int64`) so `acc += x[i]` is done in 64-bit arithmetic.',
]

_r = np.random.default_rng(16)
TESTS = [
    (np.array([1, 2, 3, 4], dtype=np.int32),),
    (np.array([-5, 10, -5, 7], dtype=np.int32),),
    (np.zeros(0, dtype=np.int32),),
    (np.arange(1000, dtype=np.int32),),                                 # 499500
    (np.full(3000, 2_000_000_000, dtype=np.int32),),                    # overflows int32 many times over
    (_r.integers(-1000, 1000, size=2047, dtype=np.int32),),
    (np.array([2147483647, 1, -2147483648], dtype=np.int32),),
]

EXPLANATION = r"""
## Why it works
Every element is visited by exactly one thread (grid-stride loop), and every thread's partial lands in `total[0]` through an atomic, so no contribution is lost. Integer addition is associative and exact, so the final value doesn't depend on which atomic arrives first.

## The race in the starter
`total[0] += x[i]` compiles to load, add, store. Two threads that load the same old value both store `old + v`, and one contribution disappears. On a GPU with thousands of threads, nearly all of them are lost. (The starter also only covers the first 256 elements.)

## Contention
All atomics hit one address, so the memory system serialises them. One atomic per element would turn a parallel reduction into a serial one. Local accumulation reduces it to one per thread; a block-level shared-memory reduction followed by one atomic per **block** reduces it further (see the min-of-array problem). Modern GPUs perform atomics in the L2 cache, so a few hundred are cheap.

## Floats are different
Floating-point addition is not associative, so float atomics give results that vary in the last bits between runs. That is acceptable for some workloads (histograms of float weights, gradient scatter) and a problem for reproducible training.

## Common bugs
- `acc = 0` with `int32` input: Numba infers `int64` for a literal 0 on most platforms, but `acc = x[0] * 0` would keep `int32` and overflow (test 5). Be explicit.
- Forgetting to zero the target: `cuda.device_array` contents are undefined.
- Returning a NumPy scalar instead of `int(...)`.

## Interview follow-ups
- Implement a 256-bin histogram of bytes. Why does a per-block shared-memory histogram, merged with global atomics at the end, beat global atomics alone?
- How is `atomicAdd` on `float64` implemented on GPUs that lack it natively? (A compare-and-swap loop.)
"""
