import numpy as np

META = dict(title='Argmax: reducing (value, index) pairs', tags=['GPU Programming'], diff='Medium',
            runtime='cuda', lessons=['ieng:13.04', 'ieng:13.03'],
            related=['max-of-array', 'argmin', 'dot-product-2'])

STATEMENT = r"""
Greedy decoding picks the token with the largest logit: an **argmax** over the vocabulary. On a GPU, argmax is a reduction whose elements are pairs $(v, i)$, combined by

$$\text{best}\big((v_1, i_1), (v_2, i_2)\big) = \begin{cases} (v_1, i_1) & v_1 > v_2 \text{ or } (v_1 = v_2 \text{ and } i_1 < i_2) \\ (v_2, i_2) & \text{otherwise} \end{cases}$$

The tie rule (smaller index wins) makes `best` associative and commutative, so the reduction order does not change the answer. It matches `np.argmax`, which returns the first occurrence.

Write `argmax(x)` with two kernels:

1. **Stage 1**, `BLOCKS = 8` blocks of `TPB = 64` threads: each thread folds its grid-stride elements into a local pair, the block tree-reduces 64 pairs held in **two** shared arrays (`float32` values, `int32` indices), and thread 0 writes the block's pair to `part_val[blockIdx.x]`, `part_idx[blockIdx.x]`.
2. **Stage 2**, one block of 64 threads: reduce the 8 partial pairs the same way and write the winning index to a 1-element `int32` device array.

## Inputs and output
- `x`: `float32`, shape `(n,)`, `1 <= n <= 4096`. No NaNs; `-inf` may appear.
- Return the index as a Python `int`, equal to `np.argmax(x)`.
- Threads with no elements start from the pair $(-\infty, n)$: it loses to every real element, including a real $-\infty$ (whose index is smaller than $n$).

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda, float32, int32

BLOCKS, TPB = 8, 64

def argmax(x):
    """x: float32 (n,), n >= 1. Returns the index of the first maximum as a Python int."""
    raise NotImplementedError

@cuda.jit(device=True)
def better(v1, i1, v2, i2):
    """True if pair (v1, i1) beats (v2, i2)."""
    return v1 > v2   # TODO: tie rule

@cuda.jit
def stage1(x, part_val, part_idx):
    pass

@cuda.jit
def stage2(part_val, part_idx, out):
    pass
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda, float32, int32

BLOCKS, TPB = 8, 64

def argmax(x):
    n = x.shape[0]
    d_x = cuda.to_device(x)
    d_val = cuda.device_array(BLOCKS, dtype=np.float32)
    d_idx = cuda.device_array(BLOCKS, dtype=np.int32)
    d_out = cuda.device_array(1, dtype=np.int32)
    stage1[BLOCKS, TPB](d_x, d_val, d_idx)
    stage2[1, TPB](d_val, d_idx, n, d_out)
    return int(d_out.copy_to_host()[0])

@cuda.jit(device=True)
def better(v1, i1, v2, i2):
    """True if (v1, i1) beats (v2, i2): larger value, ties to the smaller index."""
    return v1 > v2 or (v1 == v2 and i1 < i2)

@cuda.jit(device=True)
def block_argmax(vals, idxs, tid):
    """Tree-reduce the pairs in shared memory into slot 0. All threads must call it."""
    cuda.syncthreads()
    step = TPB // 2
    while step > 0:
        if tid < step and better(vals[tid + step], idxs[tid + step], vals[tid], idxs[tid]):
            vals[tid] = vals[tid + step]
            idxs[tid] = idxs[tid + step]
        cuda.syncthreads()
        step //= 2

@cuda.jit
def stage1(x, part_val, part_idx):
    tid = cuda.threadIdx.x
    vals = cuda.shared.array(TPB, float32)
    idxs = cuda.shared.array(TPB, int32)
    n = x.shape[0]
    bv, bi = -math.inf, n                       # identity pair: loses to every real element
    for i in range(cuda.grid(1), n, cuda.gridsize(1)):
        if better(x[i], i, bv, bi):
            bv, bi = x[i], i
    vals[tid] = bv
    idxs[tid] = bi
    block_argmax(vals, idxs, tid)
    if tid == 0:
        part_val[cuda.blockIdx.x] = vals[0]
        part_idx[cuda.blockIdx.x] = idxs[0]

@cuda.jit
def stage2(part_val, part_idx, n, out):
    tid = cuda.threadIdx.x
    vals = cuda.shared.array(TPB, float32)
    idxs = cuda.shared.array(TPB, int32)
    bv, bi = -math.inf, n
    for j in range(tid, part_val.shape[0], TPB):
        if better(part_val[j], part_idx[j], bv, bi):
            bv, bi = part_val[j], part_idx[j]
    vals[tid] = bv
    idxs[tid] = bi
    block_argmax(vals, idxs, tid)
    if tid == 0:
        out[0] = idxs[0]
'''

HINTS = [
    'Write `better(v1, i1, v2, i2)` first: `v1 > v2 or (v1 == v2 and i1 < i2)`. Every combine step, in registers and in shared memory, goes through it.',
    'Keep the pair in two parallel shared arrays. In the tree step, thread `tid < step` copies **both** `vals[tid + step]` and `idxs[tid + step]` into its slot when that pair is better.',
    'Start each thread at `(-math.inf, n)`, not `(-math.inf, 0)`: otherwise an idle thread claims index 0 and wins ties against a real `-inf`, or against the true answer when `x[0]` is not it. Stage 2 needs `n` too.',
]

_r = np.random.default_rng(33)
_tie = np.zeros(3000, dtype=np.float32)
_tie[[2900, 700, 1800]] = 5.0                 # three equal maxima: the first one (700) wins
_late = _r.standard_normal(4000).astype(np.float32)
_late[3999] = 10.0
TESTS = [
    (np.array([1.0, 3.0, 2.0, 3.0], dtype=np.float32),),     # tie: index 1
    (np.array([-4.0, -1.0, -7.0], dtype=np.float32),),
    (np.array([2.5], dtype=np.float32),),
    (np.full(700, -np.inf, dtype=np.float32),),              # every element is -inf: index 0
    (np.full(513, 1.0, dtype=np.float32),),                  # all equal, n > 512
    (_tie,),
    (_late,),
    (_r.standard_normal(2048).astype(np.float32),),
]

EXPLANATION = r"""
## Why it works
`better` defines a strict total order on pairs (value first, then index reversed), and taking the maximum under a total order is associative and commutative. So the grid-stride fold, the shared-memory tree and the second stage can group the pairs any way they like and still agree with a sequential left-to-right scan. Without the index tie-break, the tree would return whichever equal maximum happened to survive, which depends on the block layout.

## The identity pair
The identity must lose to **every** real element. $(-\infty, n)$ works: a real element has value $> -\infty$, or value $= -\infty$ and index $< n$. Using index 0 is the classic bug: an idle thread then beats a real $-\infty$ at index 3, and the all-`-inf` test returns the wrong index.

## Cost
Bandwidth-bound like any reduction: $4n$ bytes read. Shared memory per block doubles (value plus index), which is negligible here. For LLM decoding the vocabulary is ~100k entries per sequence; a batch of sequences maps naturally to one block (or more) per row, and greedy argmax is usually fused into the last kernel that produces the logits.

## Alternatives
- **Pack** the pair into one 64-bit integer so a single `atomicMax` does the job (see the argmin problem).
- Warp shuffles for the last 32 lanes, shuffling value and index separately.

## Common bugs
- Copying only the value in the tree step, so `idxs` no longer matches `vals`.
- `>=` instead of the explicit tie rule: then the later index wins at some steps and the earlier at others.
- Stage 2 reading uninitialised partials when `BLOCKS` and the partial array sizes disagree.

## Interview follow-ups
- How would you compute top-$k$ (say $k = 50$ for top-k sampling) instead of top-1?
- With NaNs allowed, `np.argmax` returns the first NaN. How would you change `better` to match?
"""
