import numpy as np

META = dict(title='Fused residual add + RMSNorm kernel', tags=['GPU Programming', 'Transformers'], diff='Medium',
            runtime='cuda', tol=[1e-5, 1e-6], lessons=['ieng:13.05', 'ieng:02.03'],
            related=['layer-norm', 'softmax-2', 'l2-normalization'])

STATEMENT = r"""
In a Llama-style decoder, every sub-layer ends with a residual add and the next one starts with RMSNorm:

$$h = x + r, \qquad y_{i,j} = \frac{h_{i,j}}{\sqrt{\frac1D \sum_{k} h_{i,k}^2 + \varepsilon}}\; g_j$$

Inference engines fuse the two into one kernel that reads `x` and `r` once and writes both `h` (the new residual stream) and `y` (the input to the next attention or MLP block).

## Task
Write `fused_add_rmsnorm(x, r, g, eps)` returning the tuple `(h, y)`.

- `x`, `r`: `float32` `(rows, D)`; `g`: `float32` `(D,)`; `eps`: Python float. `rows, D >= 1`; `D` may be larger or smaller than the block.
- `h` and `y`: `float32` `(rows, D)`.

## Mapping
- One block of **64 threads per row**. Each thread handles columns `tid, tid + 64, ...`: it computes $h$, writes it, and accumulates $\sum h^2$ locally.
- Combine the 64 partial sums with a tree reduction in shared memory, then every thread scales its own columns.
- Accumulate in `float32`. Compared with relative tolerance $10^{-5}$, absolute $10^{-6}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda, float32

TPB = 64

def fused_add_rmsnorm(x, r, g, eps):
    """x, r: float32 (rows, D); g: float32 (D,). Returns (h, y): h = x + r, y = RMSNorm(h) * g."""
    raise NotImplementedError

@cuda.jit
def add_rmsnorm_kernel(x, r, g, eps, h, y):
    pass
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda, float32

TPB = 64  # threads per row; a power of two for the tree reduction

def fused_add_rmsnorm(x, r, g, eps):
    rows, D = x.shape
    d_x, d_r, d_g = cuda.to_device(x), cuda.to_device(r), cuda.to_device(g)
    d_h = cuda.device_array((rows, D), dtype=np.float32)
    d_y = cuda.device_array((rows, D), dtype=np.float32)
    add_rmsnorm_kernel[rows, TPB](d_x, d_r, d_g, eps, d_h, d_y)   # one block per row
    return d_h.copy_to_host(), d_y.copy_to_host()

@cuda.jit
def add_rmsnorm_kernel(x, r, g, eps, h, y):
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    D = x.shape[1]
    buf = cuda.shared.array(TPB, float32)

    ss = float32(0.0)
    for j in range(tid, D, TPB):              # pass 1: residual add + local sum of squares
        v = x[row, j] + r[row, j]
        h[row, j] = v
        ss += v * v
    buf[tid] = ss
    cuda.syncthreads()

    step = TPB // 2
    while step > 0:                           # tree reduction of the 64 partials
        if tid < step:
            buf[tid] += buf[tid + step]
        cuda.syncthreads()
        step //= 2
    inv_rms = 1.0 / math.sqrt(buf[0] / D + eps)

    for j in range(tid, D, TPB):              # pass 2: each thread re-reads only what it wrote
        y[row, j] = h[row, j] * inv_rms * g[j]
'''

HINTS = [
    'Launch `rows` blocks of 64 threads. In the first loop, each thread writes `h[row, j] = x + r` and adds `h*h` to a local sum.',
    'Store the local sum in a shared array, tree-reduce it (with `cuda.syncthreads()` after every step), then compute `inv_rms = 1 / sqrt(total / D + eps)`.',
    'In the second loop, each thread only reads back the `h` values **it** wrote, so no extra barrier is needed for `h` itself; the barrier inside the reduction already orders everything else.',
]

_r = np.random.default_rng(14)
_f = lambda *s: _r.standard_normal(s).astype(np.float32)
TESTS = [
    (np.array([[1.0, 2.0, 3.0]], dtype=np.float32), np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
     np.array([1.0, 1.0, 1.0], dtype=np.float32), 1e-6),
    (np.array([[3.0, -4.0], [1.0, 1.0]], dtype=np.float32), np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32),
     np.array([2.0, 0.5], dtype=np.float32), 1e-5),
    (np.zeros((2, 4), dtype=np.float32), np.zeros((2, 4), dtype=np.float32), _f(4), 1e-5),  # all zeros: eps prevents 0/0
    (_f(1, 1), _f(1, 1), _f(1), 1e-6),
    (_f(4, 64), _f(4, 64), _f(64), 1e-6),
    (_f(3, 200), _f(3, 200), _f(200), 1e-5),
    ((_f(6, 48) * 50), _f(6, 48), _f(48), 1e-6),
]

EXPLANATION = r"""
## Why it works
RMSNorm needs one statistic per row, the mean of squares, so the row is a natural unit of work for one block. The residual add is element-wise, so it fits into the same pass that accumulates $\sum h^2$; writing `h` out at the same time gives the caller the updated residual stream for free.

## Why fuse
Unfused, the model runs `h = x + r` (read 2, write 1) and then RMSNorm (read 1 for the statistic, read 1 again to normalise, write 1). Fused: read 2, write 2, plus re-reading `h`, which each thread wrote moments ago, so it is likely still in cache. At decode time these memory-bound kernels and their launch overheads are a visible fraction of per-token latency, which is why vLLM and TensorRT-LLM ship exactly this fused op.

## RMSNorm vs LayerNorm
RMSNorm drops the mean subtraction and the bias. One reduction instead of two (or one Welford pass), with no measurable quality loss in LLMs.

## Common bugs
- Dividing by `TPB` or by the number of active threads instead of `D`.
- Putting `eps` outside the square root: `1/(sqrt(ms) + eps)` is a different formula, and the all-zero test catches it only if `eps` is large.
- Accumulating into `buf[tid]` directly in the loop is fine; forgetting to zero it first is not (shared memory starts with garbage).

## Interview follow-ups
- With $D = 8192$ and 64 threads, each thread handles 128 columns. How would you instead keep the whole row in registers to avoid re-reading `h`?
- Why do some engines compute the statistic in `float32` even when activations are `bf16`?
"""
