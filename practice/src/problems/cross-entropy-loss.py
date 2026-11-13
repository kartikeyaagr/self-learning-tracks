import numpy as np

META = dict(title='Fused cross-entropy: loss and gradient in one kernel', tags=['GPU Programming', 'Language Models'],
            diff='Medium', runtime='cuda', tol=[1e-5, 1e-6], lessons=['ieng:13.05', 'ieng:13.06'],
            related=['softmax-2', 'argmax', 'max-of-array'])

STATEMENT = r"""
For a language model with a 100k-token vocabulary, the logits of one batch can take gigabytes, and the standard pipeline (softmax, then log, then loss, then a separate backward pass) reads and writes them several times. A **fused** kernel reads each row of logits once and produces both the loss and the gradient.

For row $r$ with logits $z_{r,:}$ and target class $t_r$:

$$\ell_r = \log \sum_j e^{z_{r,j}} - z_{r,t_r}, \qquad \frac{\partial L}{\partial z_{r,j}} = \frac{\mathrm{softmax}(z_r)_j - [j = t_r]}{N_{\text{valid}}}$$

where $L = \frac{1}{N_{\text{valid}}} \sum_{r \text{ valid}} \ell_r$ is the mean over rows whose target is not `ignore_index = -100` (padding tokens). Ignored rows contribute 0 to the loss and have a zero gradient row.

## Task
Write `cross_entropy_fwd_bwd(logits, targets)` returning `(loss, grad)`.

- Kernel: **one block of 64 threads per row**. The block computes the row max and $\sum_j e^{z_j - m}$ with shared-memory tree reductions (the stable log-sum-exp $m + \log \sum_j e^{z_j - m}$), then thread 0 writes $\ell_r$ to a per-row `float32` array and all threads write the gradient row.
- The host counts $N_{\text{valid}}$ from `targets` before the launch and passes it in (the gradient needs it). It returns `loss = sum of per-row losses / N_valid` as a Python float computed in `float64` on the host, or `0.0` when no row is valid (all gradients 0 then).
- Ignored rows return from the kernel only after writing zeros; since the whole block takes the same branch, no barrier is skipped by part of a block.

## Inputs and output
- `logits`: `float32` `(N, C)`, $N \ge 1$, $C \ge 1$, finite values up to $\pm 10^3$. `targets`: `int64` `(N,)`, each in `[0, C)` or equal to `-100`.
- `grad`: `float32` `(N, C)`. Compared with relative tolerance $10^{-5}$, absolute $10^{-6}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda, float32

TPB = 64
IGNORE = -100

def cross_entropy_fwd_bwd(logits, targets):
    """logits: float32 (N, C), targets: int64 (N,). Returns (mean loss: float, grad: float32 (N, C))."""
    N, C = logits.shape
    n_valid = N                                   # TODO: rows with target -100 do not count
    d_z = cuda.to_device(np.ascontiguousarray(logits))
    d_t = cuda.to_device(targets)
    d_loss = cuda.device_array(N, dtype=np.float32)
    d_grad = cuda.device_array((N, C), dtype=np.float32)
    ce_kernel[N, TPB](d_z, d_t, n_valid, d_loss, d_grad)   # one block per row
    return float(d_loss.copy_to_host().astype(np.float64).sum() / n_valid), d_grad.copy_to_host()

@cuda.jit
def ce_kernel(z, targets, n_valid, loss, grad):
    r = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    buf = cuda.shared.array(TPB, float32)
    # TODO 1: ignored rows write zeros and return
    # TODO 2: block-wide max and sum of exp(z - max)
    # TODO 3: thread 0 writes loss[r]; every thread writes its columns of grad[r, :]
    pass
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda, float32

TPB = 64
IGNORE = -100

def cross_entropy_fwd_bwd(logits, targets):
    N, C = logits.shape
    n_valid = int(np.count_nonzero(targets != IGNORE))
    d_z = cuda.to_device(np.ascontiguousarray(logits))
    d_t = cuda.to_device(targets)
    d_loss = cuda.device_array(N, dtype=np.float32)
    d_grad = cuda.device_array((N, C), dtype=np.float32)
    ce_kernel[N, TPB](d_z, d_t, max(n_valid, 1), d_loss, d_grad)   # one block per row
    if n_valid == 0:
        return 0.0, d_grad.copy_to_host()
    return float(d_loss.copy_to_host().astype(np.float64).sum() / n_valid), d_grad.copy_to_host()

@cuda.jit
def ce_kernel(z, targets, n_valid, loss, grad):
    r = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    C = z.shape[1]
    t = targets[r]
    if t == IGNORE:                       # uniform across the block: safe to leave early
        for j in range(tid, C, TPB):
            grad[r, j] = 0.0
        if tid == 0:
            loss[r] = 0.0
        return

    buf = cuda.shared.array(TPB, float32)
    # Row max.
    m = -math.inf
    for j in range(tid, C, TPB):
        m = max(m, z[r, j])
    buf[tid] = m
    cuda.syncthreads()
    step = TPB // 2
    while step > 0:
        if tid < step:
            buf[tid] = max(buf[tid], buf[tid + step])
        cuda.syncthreads()
        step //= 2
    m = buf[0]
    cuda.syncthreads()                    # everyone has read the max before buf is reused

    # Sum of shifted exponentials.
    s = float32(0.0)
    for j in range(tid, C, TPB):
        s += math.exp(z[r, j] - m)
    buf[tid] = s
    cuda.syncthreads()
    step = TPB // 2
    while step > 0:
        if tid < step:
            buf[tid] += buf[tid + step]
        cuda.syncthreads()
        step //= 2
    total = buf[0]

    if tid == 0:
        loss[r] = m + math.log(total) - z[r, t]          # log-sum-exp minus the target logit
    inv = float32(1.0) / (total * n_valid)
    for j in range(tid, C, TPB):
        g = math.exp(z[r, j] - m) * inv                  # softmax / N_valid
        if j == t:
            g -= float32(1.0) / n_valid
        grad[r, j] = g
'''

HINTS = [
    'Count `n_valid = (targets != -100).sum()` on the host and pass it to the kernel; launch `N` blocks of 64 threads. Whole blocks are either ignored or not, so an ignored row can write zeros and `return` before any barrier.',
    'Reuse the softmax pattern: block max, barrier, block sum of `exp(z - m)`. The per-row loss is `m + log(sum) - z[r, t]`, written by thread 0.',
    'Gradient: each thread writes `exp(z[r, j] - m) / (sum * n_valid)` for its columns and subtracts `1 / n_valid` at `j == t`. On the host, sum the per-row losses in float64 and divide by `n_valid` (return `0.0` when it is 0).',
]

_r = np.random.default_rng(54)
_t = lambda *v: np.array(v, dtype=np.int64)
TESTS = [
    (np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32), _t(0, 2)),
    (np.array([[1.0, 2.0], [3.0, 1.0], [0.5, 0.5]], dtype=np.float32), _t(1, -100, 0)),   # one ignored row
    (np.array([[1000.0, -1000.0, 999.0]], dtype=np.float32), _t(2)),                       # needs the max shift
    (_r.standard_normal((3, 5)).astype(np.float32), _t(-100, -100, -100)),                 # nothing valid
    (_r.standard_normal((4, 1)).astype(np.float32), _t(0, 0, 0, 0)),                       # C = 1: loss 0
    ((_r.standard_normal((6, 200)) * 4).astype(np.float32), _r.integers(0, 200, 6)),       # C > 64
    ((_r.standard_normal((8, 64)) * 10).astype(np.float32), np.where(_r.random(8) < 0.3, -100, _r.integers(0, 64, 8))),
]

EXPLANATION = r"""
## Why it works
$\ell_r = -\log \mathrm{softmax}(z_r)_{t_r}$, rewritten with log-sum-exp so nothing overflows: $\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}$ with every exponent $\le 0$. The gradient of $-\log \mathrm{softmax}_t$ with respect to $z_j$ is $\mathrm{softmax}_j - [j = t]$, and the mean over valid rows divides by $N_{\text{valid}}$. Both need exactly the two row statistics $m$ and $\sum_j e^{z_j - m}$, so one kernel computes them once and emits everything.

## Why fuse
Unfused, PyTorch materialises the softmax (or log-softmax) in the forward pass and reads it again in backward: several full passes over an $N \times C$ tensor. For LLM training with $C \approx 10^5$ that tensor can dominate peak memory. Fused kernels (as in Liger or the losses shipped with recent training stacks) go further: they write the gradient **into the logits buffer** and even chunk the final matmul so the full logits never exist at once.

## Determinism and precision
Per-row losses go to an array and are summed on the host in `float64` rather than with a floating-point atomic, so the result does not depend on block scheduling. A real kernel would do the final sum in a second small kernel for the same reason.

## Common bugs
- Dividing by $N$ instead of $N_{\text{valid}}$, or including ignored rows in the mean.
- Indexing `z[r, -100]` for an ignored row (negative indices silently wrap in NumPy, and read out of bounds on a GPU).
- A non-uniform early `return` inside a block (fine here only because the branch depends on `blockIdx` alone).
- Computing `log(softmax)` as two steps: `log(0)` for strongly negative logits gives `-inf`.

## Interview follow-ups
- Add label smoothing $\epsilon$: what are the new loss and gradient, and does the kernel still need only $m$ and $\sum e^{z - m}$?
- The vocabulary is 256k and the batch has 16k tokens. Why might you compute the loss chunk by chunk together with the final projection, rather than materialising the logits?
"""
