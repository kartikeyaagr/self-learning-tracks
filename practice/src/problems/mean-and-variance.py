import numpy as np

META = dict(title="Mean and variance in one pass: Welford meets the tree reduction", tags=['GPU Programming'],
            diff='Medium', runtime='cuda', tol=[1e-4, 1e-5], lessons=['ieng:13.04', 'ieng:13.05'],
            related=['dot-product-2', 'rms-normalization', 'sum-of-array'])

STATEMENT = r"""
The textbook formula $\operatorname{Var}(x) = \mathbb{E}[x^2] - \mathbb{E}[x]^2$ needs one pass, but in `float32` it is useless when the mean is large compared with the spread: for data around $10^4$ with unit variance, $\mathbb{E}[x^2] \approx 10^8$ and `float32` keeps only about 7 digits, so the difference is noise.

The stable alternative tracks a triple $(c, \mu, M_2)$ (count, mean, sum of squared deviations from the mean). Two triples merge exactly:

$$c = c_a + c_b,\qquad \delta = \mu_b - \mu_a,\qquad \mu = \mu_a + \delta\,\frac{c_b}{c},\qquad M_2 = M_{2,a} + M_{2,b} + \delta^2\,\frac{c_a c_b}{c}$$

Adding a single value $x$ is the special case $(1, x, 0)$ (Welford's update). Because the merge is associative, it plugs straight into a GPU tree reduction.

## Task
Write `mean_var(x)` that returns `(mean, variance)` with the **population** variance $M_2 / c$ (NumPy's default, `ddof=0`).

1. **Stage 1**, `BLOCKS = 4` blocks of `TPB = 64` threads: each thread folds its grid-stride elements into a triple, the block tree-reduces 64 triples with the merge rule (three `float32` shared arrays), and thread 0 writes the block's triple to three partial arrays of length 4.
2. **Stage 2**, one block of 64 threads: merge the 4 partials the same way and write the final mean and $M_2/c$.

Threads with no elements hold $(0, 0, 0)$. The merge must handle $c = 0$ (two empty triples) without dividing by zero.

## Inputs and output
- `x`: `float32`, shape `(n,)`, `1 <= n <= 4096`. All arithmetic in `float32`; counts can be held as `float32` (exact up to $2^{24}$).
- Return a tuple of two Python floats. Compared with relative tolerance $10^{-4}$, absolute $10^{-5}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda, float32

BLOCKS, TPB = 4, 64

def mean_var(x):
    """x: float32 (n,), n >= 1. Returns (mean, population variance) as Python floats."""
    raise NotImplementedError

@cuda.jit(device=True)
def merge(ca, ma, qa, cb, mb, qb):
    """Merge triples (count, mean, M2). Returns the merged (count, mean, M2)."""
    return ca + cb, ma, qa   # TODO

@cuda.jit
def stage1(x, pc, pm, pq):
    pass

@cuda.jit
def stage2(pc, pm, pq, out):
    pass
'''

SOLUTION = '''import numpy as np
from numba import cuda, float32

BLOCKS, TPB = 4, 64

def mean_var(x):
    d_x = cuda.to_device(x)
    pc, pm, pq = (cuda.device_array(BLOCKS, dtype=np.float32) for _ in range(3))
    d_out = cuda.device_array(2, dtype=np.float32)
    stage1[BLOCKS, TPB](d_x, pc, pm, pq)
    stage2[1, TPB](pc, pm, pq, d_out)
    mean, var = d_out.copy_to_host()
    return float(mean), float(var)

@cuda.jit(device=True)
def merge(ca, ma, qa, cb, mb, qb):
    """Chan et al. pairwise merge of (count, mean, M2) triples."""
    c = ca + cb
    if c == 0:
        return float32(0.0), float32(0.0), float32(0.0)   # both empty
    d = mb - ma
    return c, ma + d * (cb / c), qa + qb + d * d * (ca * cb / c)

@cuda.jit(device=True)
def block_merge(sc, sm, sq, tid):
    """Tree-reduce the triples in shared memory into slot 0. All threads must call it."""
    cuda.syncthreads()
    step = TPB // 2
    while step > 0:
        if tid < step:
            c, m, q = merge(sc[tid], sm[tid], sq[tid], sc[tid + step], sm[tid + step], sq[tid + step])
            sc[tid], sm[tid], sq[tid] = c, m, q
        cuda.syncthreads()
        step //= 2

@cuda.jit
def stage1(x, pc, pm, pq):
    tid = cuda.threadIdx.x
    sc = cuda.shared.array(TPB, float32)
    sm = cuda.shared.array(TPB, float32)
    sq = cuda.shared.array(TPB, float32)
    c, m, q = float32(0.0), float32(0.0), float32(0.0)
    for i in range(cuda.grid(1), x.shape[0], cuda.gridsize(1)):
        c, m, q = merge(c, m, q, float32(1.0), x[i], float32(0.0))   # Welford step
    sc[tid], sm[tid], sq[tid] = c, m, q
    block_merge(sc, sm, sq, tid)
    if tid == 0:
        b = cuda.blockIdx.x
        pc[b], pm[b], pq[b] = sc[0], sm[0], sq[0]

@cuda.jit
def stage2(pc, pm, pq, out):
    tid = cuda.threadIdx.x
    sc = cuda.shared.array(TPB, float32)
    sm = cuda.shared.array(TPB, float32)
    sq = cuda.shared.array(TPB, float32)
    if tid < pc.shape[0]:
        sc[tid], sm[tid], sq[tid] = pc[tid], pm[tid], pq[tid]
    else:
        sc[tid], sm[tid], sq[tid] = float32(0.0), float32(0.0), float32(0.0)
    block_merge(sc, sm, sq, tid)
    if tid == 0:
        out[0] = sm[0]
        out[1] = sq[0] / sc[0]          # population variance; n >= 1 so sc[0] > 0
'''

HINTS = [
    'Write `merge` first, straight from the formulas, with an early return of zeros when `ca + cb == 0`. Test it on paper: merging $(1, x, 0)$ into $(0, 0, 0)$ must give $(1, x, 0)$.',
    'Stage 1 per thread: start at `(0, 0, 0)` and call `merge(c, m, q, 1, x[i], 0)` for each grid-stride element. Then store the triple in three shared arrays and tree-reduce with the same `merge`.',
    'Stage 2 loads the 4 partials into slots 0-3 and zeros elsewhere, reduces, and writes `mean = sm[0]`, `var = sq[0] / sc[0]`. Copy the 2-element result back and return two floats.',
]

_r = np.random.default_rng(43)
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),),                          # (2.5, 1.25)
    (np.array([5.0], dtype=np.float32),),                                         # (5, 0)
    (np.full(300, 7.25, dtype=np.float32),),                                      # zero variance
    (np.array([-1.0, 1.0] * 500, dtype=np.float32),),                             # (0, 1)
    ((10000.0 + _r.standard_normal(2048)).astype(np.float32),),                   # E[x^2]-E[x]^2 fails here
    ((_r.standard_normal(1000) * 1e-3).astype(np.float32),),
    (_r.uniform(-50, 50, 4096).astype(np.float32),),
    ((3.0 * _r.standard_normal(777) - 2.0).astype(np.float32),),
]

EXPLANATION = r"""
## Why it works
A triple summarises a set of values exactly (up to rounding): its size, mean and $M_2 = \sum (x - \mu)^2$. The merge rule follows from expanding $\sum (x - \mu)^2$ over the union around the new mean; the cross term is $\delta^2 c_a c_b / c$. Since merging is associative, any reduction tree gives the same triple in exact arithmetic, so the grid-stride fold, the shared-memory tree and the second stage can all use it.

## Why it is stable
The naive formula subtracts two huge, nearly equal numbers. The merge only ever works with **deviations** ($\delta$, $M_2$), which stay on the scale of the spread. On test 5 (mean $10^4$, variance 1), the naive `float32` one-pass formula is off by far more than 100 %, while Welford/Chan agrees with NumPy's two-pass result to about $10^{-5}$.

## Cost
One pass over $4n$ bytes, a few FLOPs and one division per element: still bandwidth-bound. A two-pass algorithm (mean first, then $\sum (x - \mu)^2$) is equally stable but reads the data twice. The division per merge is the price of one pass; normalisation kernels (LayerNorm, BatchNorm, GroupNorm) pay it gladly because the row often sits in registers anyway.

## Common bugs
- Dividing by zero when two empty triples merge (idle threads, empty stage-2 slots).
- Updating `ma` before using it in the $M_2$ term: compute $\delta$ once, from the old means.
- Returning the sample variance ($M_2/(c-1)$) instead of the population variance, or the reverse; `torch.var` defaults to the unbiased one.

## Interview follow-ups
- LayerNorm computes mean and variance per row. How would you map this to one warp per row, and what changes in the merge when every lane holds the same count?
- How does BatchNorm combine running statistics across GPUs in data-parallel training, and why does it send $(c, \mu, M_2)$ (or sums) rather than per-GPU variances?
"""
