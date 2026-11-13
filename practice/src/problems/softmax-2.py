import numpy as np

META = dict(title='Row softmax: one block per row', tags=['GPU Programming', 'Attention'], diff='Medium',
            runtime='cuda', tol=[1e-5, 1e-6], lessons=['ieng:13.05', 'ieng:13.03'],
            related=['softmax-rows', 'cross-entropy-loss', 'max-of-array'])

STATEMENT = r"""
Softmax over the last dimension shows up in every attention layer and every classifier head:

$$\mathrm{softmax}(x)_{r,j} = \frac{e^{x_{r,j} - m_r}}{\sum_k e^{x_{r,k} - m_r}}, \qquad m_r = \max_k x_{r,k}$$

Subtracting the row max $m_r$ changes nothing mathematically but keeps every exponent $\le 0$, so nothing overflows.

## The mapping
Assign **one block of 64 threads per row** (grid = `rows` blocks). Inside a block, the threads cooperate:
1. Each thread scans columns `tid, tid + 64, tid + 128, ...` and keeps a running max. The 64 partial maxima go into a `float32` shared-memory array and are combined with a **tree reduction** (halve the active threads each step, `cuda.syncthreads()` between steps). Thread 0's slot then holds $m_r$.
2. Same pattern for the sum of $e^{x - m_r}$.
3. Every thread writes its columns of the output.

## Inputs and outputs
- `x`: `float32` array of shape `(rows, cols)`; `rows` may be 0, `cols` may be any positive number (smaller or larger than 64). Entries may be `-inf` (masked positions), but every row has at least one finite entry. Values up to $\pm 10^4$.
- Return `softmax(x)` as a `float32` array of shape `(rows, cols)`. For `rows == 0`, return an empty `(0, cols)` array without launching.
- Compared with relative tolerance $10^{-5}$, absolute $10^{-6}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda, float32

TPB = 64  # threads per block = threads per row

def softmax(x):
    """x: float32 (rows, cols). Returns the row-wise softmax, float32 (rows, cols)."""
    rows, cols = x.shape
    if rows == 0:
        return np.zeros((0, cols), dtype=np.float32)
    d_x = cuda.to_device(np.ascontiguousarray(x))
    d_out = cuda.device_array((rows, cols), dtype=np.float32)
    softmax_kernel[rows, TPB](d_x, d_out)
    return d_out.copy_to_host()

@cuda.jit
def softmax_kernel(x, out):
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    buf = cuda.shared.array(TPB, float32)
    # TODO 1: block-wide max of x[row, :]
    # TODO 2: block-wide sum of exp(x[row, :] - max)
    # TODO 3: write out[row, j] for this thread's columns
    pass
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda, float32

TPB = 64  # threads per block = threads per row (a power of two for the tree reduction)

def softmax(x):
    rows, cols = x.shape
    if rows == 0:
        return np.zeros((0, cols), dtype=np.float32)
    d_x = cuda.to_device(np.ascontiguousarray(x))
    d_out = cuda.device_array((rows, cols), dtype=np.float32)
    softmax_kernel[rows, TPB](d_x, d_out)       # one block per row
    return d_out.copy_to_host()

@cuda.jit
def softmax_kernel(x, out):
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    cols = x.shape[1]
    buf = cuda.shared.array(TPB, float32)

    # 1. Row max: strided scan, then a tree reduction in shared memory.
    m = -math.inf
    for j in range(tid, cols, TPB):
        m = max(m, x[row, j])
    buf[tid] = m
    cuda.syncthreads()
    step = TPB // 2
    while step > 0:
        if tid < step:
            buf[tid] = max(buf[tid], buf[tid + step])
        cuda.syncthreads()                      # every thread, every step: never inside the if
        step //= 2
    row_max = buf[0]
    cuda.syncthreads()                          # all threads read buf[0] before it is overwritten

    # 2. Row sum of shifted exponentials, same pattern.
    acc = 0.0
    for j in range(tid, cols, TPB):
        acc += math.exp(x[row, j] - row_max)   # exponent <= 0; exp(-inf) = 0 for masked entries
    buf[tid] = acc
    cuda.syncthreads()
    step = TPB // 2
    while step > 0:
        if tid < step:
            buf[tid] += buf[tid + step]
        cuda.syncthreads()
        step //= 2
    inv = 1.0 / buf[0]

    # 3. Normalise. Adjacent threads write adjacent columns: coalesced.
    for j in range(tid, cols, TPB):
        out[row, j] = math.exp(x[row, j] - row_max) * inv
'''

HINTS = [
    'Get one reduction right first: each thread folds its strided columns into a local value, writes it to `buf[tid]`, then the block halves `step` from 32 down to 1 with `buf[tid] = op(buf[tid], buf[tid + step])` for `tid < step`.',
    '`cuda.syncthreads()` must be reached by **every** thread of the block: put it after the `if tid < step`, not inside it. Add one more barrier after reading `buf[0]` before reusing `buf` for the sum.',
    'Start the max at `-math.inf` so threads with no columns (when `cols < 64`) contribute the identity. `math.exp(-inf - m)` is exactly 0, which handles masked entries for free.',
]

_r = np.random.default_rng(9)
_masked = _r.standard_normal((3, 40)).astype(np.float32)
_masked[0, 5:] = -np.inf
_masked[2, ::3] = -np.inf
TESTS = [
    (np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32),),
    (np.array([[1000.0, 1001.0, 1002.0, 999.0]], dtype=np.float32),),
    (np.full((2, 70), -3.0, dtype=np.float32),),                     # uniform rows: every entry 1/70
    (np.array([[-1e4, 0.0, 1e4]], dtype=np.float32),),
    (_masked,),                                                       # -inf = masked positions
    (_r.standard_normal((4, 1)).astype(np.float32),),                 # one column: all ones
    ((_r.standard_normal((5, 200)) * 3).astype(np.float32),),         # cols > 64: each thread loops
    ((_r.standard_normal((8, 64)) * 10).astype(np.float32),),
]

EXPLANATION = r"""
## Why it works
Each block owns one row, so rows never interact and need no global synchronisation. Within a block, the strided scan reduces `cols` values to 64 partials, and the tree reduction combines them in $\log_2 64 = 6$ steps. Because max and addition are associative, any grouping gives the right answer (up to float rounding for the sum). The barrier after each step guarantees that a thread reading `buf[tid + step]` sees the value written in the previous step.

## Performance
- The kernel makes **three passes** over the row (max, sum, write), reading `x` from global memory three times. Real implementations keep the row in registers or shared memory when it fits, or use the **online softmax**: a single pass that carries $(m, s)$ and rescales $s$ by $e^{m_{\text{old}} - m_{\text{new}}}$ whenever the max grows. Online softmax is the core trick behind FlashAttention.
- One block per row is ideal when there are many rows (attention: `batch × heads × queries`). With a few very long rows, most of the GPU is idle and you would split rows across blocks.
- Real kernels replace the shared-memory tree with **warp shuffles** (`cuda.shfl_down_sync`) for the last 32 lanes: no shared memory traffic and no barriers within a warp.

## Common bugs
- `cuda.syncthreads()` inside `if tid < step`: threads that skip it never arrive, and the block deadlocks (or, on some hardware, behaves undefinedly).
- Reusing `buf` for the sum without a barrier after `row_max = buf[0]`: a fast thread overwrites `buf[0]` before a slow one has read the max.
- Initialising the max with `0.0`: wrong for rows that are all negative.
- No max subtraction: `exp(1000)` overflows (test 2).

## Interview follow-ups
- Write the online-softmax update for merging two partial states $(m_1, s_1)$ and $(m_2, s_2)$.
- How does this kernel change for causal attention scores, where row $r$ only has $r+1$ valid columns? (Skip the masked tail instead of reading `-inf`.)
"""
