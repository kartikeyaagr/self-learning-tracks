import numpy as np

META = dict(title='Argmin with one packed 64-bit atomic', tags=['GPU Programming'], diff='Medium',
            runtime='cuda', lessons=['ieng:13.04'],
            related=['argmax', 'min-of-array', 'sum-of-array'])

STATEMENT = r"""
An arg-reduction normally carries two values (the best value and where it is). Atomics work on **one** word, but you can make a pair fit in one word: put the value in the high 32 bits and the index in the low 32 bits,

$$\text{key}(v, i) = v \cdot 2^{32} + i, \qquad 0 \le i < 2^{32}$$

Comparing two keys as signed 64-bit integers compares the values first and breaks ties by the smaller index. So `cuda.atomic.min` on keys computes the argmin, ties going to the first occurrence, in a single kernel.

Write `argmin(x)` for a 1-D `int32` array.

## Requirements
- Fixed grid: **4 blocks of 64 threads**, grid-stride loop.
- Each thread computes the smallest key of its own elements in a local `int64`, then issues **one** `cuda.atomic.min` into a 1-element `int64` device array initialised to the largest `int64` ($2^{63} - 1$). Threads with no elements skip the atomic.
- Build the key as `(int64(v) << 32) + i`. Every `int32` value fits: $v \cdot 2^{32}$ lies in $[-2^{63}, 2^{63} - 2^{32}]$, and adding $i < 2^{32}$ cannot overflow.
- Decode the index on the host from the low 32 bits: `key & 0xFFFFFFFF`.

## Inputs and output
- `x`: `int32`, shape `(n,)`, `1 <= n <= 4096`; any `int32` values, including the extremes.
- Return the index as a Python `int`, equal to `np.argmin(x)`.

Integer min is exact and order-independent, so the result is deterministic.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda, int64

BLOCKS, TPB = 4, 64
KEY_MAX = np.iinfo(np.int64).max

def argmin(x):
    """x: int32 (n,), n >= 1. Returns the index of the first minimum as a Python int."""
    d_x = cuda.to_device(x)
    d_best = cuda.to_device(np.full(1, KEY_MAX, dtype=np.int64))
    argmin_kernel[BLOCKS, TPB](d_x, d_best)
    return int(d_best.copy_to_host()[0])   # TODO: this is the key, not the index

@cuda.jit
def argmin_kernel(x, best):
    i = cuda.grid(1)
    if i < x.shape[0]:
        cuda.atomic.min(best, 0, x[i])     # TODO: pack (value, index) into one int64 key
'''

SOLUTION = '''import numpy as np
from numba import cuda, int64

BLOCKS, TPB = 4, 64
KEY_MAX = np.iinfo(np.int64).max          # identity of min over keys

def argmin(x):
    d_x = cuda.to_device(x)
    d_best = cuda.to_device(np.full(1, KEY_MAX, dtype=np.int64))
    argmin_kernel[BLOCKS, TPB](d_x, d_best)
    key = int(d_best.copy_to_host()[0])
    return key & 0xFFFFFFFF                # low 32 bits hold the index

@cuda.jit
def argmin_kernel(x, best):
    local = int64(KEY_MAX)
    for i in range(cuda.grid(1), x.shape[0], cuda.gridsize(1)):
        key = (int64(x[i]) << 32) + i      # value in the high bits, index in the low bits
        local = min(local, key)
    if local != KEY_MAX:                   # idle threads (n < 256) have nothing to publish
        cuda.atomic.min(best, 0, local)
'''

HINTS = [
    'Inside the loop, form `key = (int64(x[i]) << 32) + i` and keep the smallest key in a local `int64` that starts at `KEY_MAX`.',
    'After the loop, one `cuda.atomic.min(best, 0, local)` per thread. Because the grid-stride loop also fixes the "only 256 elements" bug, this is the whole kernel.',
    'On the host, `key & 0xFFFFFFFF` recovers the index. It works for negative values too: Python integers use two\'s-complement semantics for `&`, and the index sits in the low bits.',
]

_r = np.random.default_rng(34)
_i32 = np.iinfo(np.int32)
_tie = _r.integers(0, 100, 3000).astype(np.int32)
_tie[[2500, 999, 1500]] = -7                    # three equal minima: index 999 wins
TESTS = [
    (np.array([5, 2, 8, 2, 9], dtype=np.int32),),                       # tie: index 1
    (np.array([-3, 4, -10, 0], dtype=np.int32),),
    (np.array([17], dtype=np.int32),),
    (np.array([_i32.max, _i32.min, _i32.min, 0], dtype=np.int32),),      # extreme values
    (np.full(300, _i32.max, dtype=np.int32),),                          # all equal: index 0
    (np.arange(4000, 0, -1).astype(np.int32),),                         # min at the last index
    (_tie,),
    (_r.integers(-10**9, 10**9, 4096).astype(np.int32),),
]

EXPLANATION = r"""
## Why it works
For keys $k_1 = v_1 2^{32} + i_1$ and $k_2 = v_2 2^{32} + i_2$ with $0 \le i < 2^{32}$: if $v_1 < v_2$ then $k_1 \le v_1 2^{32} + 2^{32} - 1 < v_2 2^{32} \le k_2$, and if $v_1 = v_2$ the comparison falls to $i_1$ vs $i_2$. So the order on keys is exactly "smaller value, then smaller index", and the minimum key is the first minimum. A single associative, exact operation (integer min) does both jobs, so no shared memory or second kernel is needed.

## Cost
One pass over $4n$ bytes plus 256 atomics. The packed key also halves the state the tree reduction in the argmax problem has to move around.

## Floats
For `float32` values the same trick works after mapping the float to an integer with the same ordering: take its bit pattern $b$ as `int32`; if $b < 0$ (negative float) use $b \oplus \texttt{0x7FFFFFFF}$, otherwise $b$. That flips the order of negative floats, which IEEE stores as sign-magnitude. It is how many GPU libraries implement float `atomicMin`/`atomicMax` and radix sorts.

## Common bugs
- Packing as `(i << 32) + v`: orders by index first.
- Shifting an `int32` without widening: `x[i] << 32` overflows in 32-bit arithmetic on the GPU.
- Initialising `best` to 0: every key with a non-negative value loses to it.
- Returning the key, or using `key >> 32` (that is the value).

## Interview follow-ups
- How many bits would you need for an argmin over $2^{40}$ elements of `int32` values? What breaks?
- k-means assigns each point to its nearest centroid: an argmin per row of a distance matrix. Would you use this atomic trick or one warp per row, and why?
"""
