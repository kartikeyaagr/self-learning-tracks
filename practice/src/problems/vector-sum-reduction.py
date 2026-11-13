import numpy as np

META = dict(title='Two-stage vector sum', tags=['GPU Programming'], diff='Easy', runtime='triton', tol=[1e-4, 1e-4],
            lessons=['ieng:13.03', 'ieng:13.04'], related=['vector-max-reduction', 'l2-vector-norm'])

STATEMENT = r"""
A GPU reduction can't be one loop: thousands of programs each reduce a block, and the partial results then have to be combined. Write the classic **two-stage** version.

Write `vector_sum(x)` that returns the sum of all elements of `x`.

## Inputs
- `x`: contiguous `float32` CUDA tensor, any shape, any number of elements (including 0).

## Output
- A 0-d `float32` CUDA tensor holding the sum. An empty input sums to `0.0`.

## Requirements
1. **Stage 1:** a kernel with `triton.cdiv(n, BLOCK)` programs. Program `pid` sums its block (masked lanes contribute 0) with `tl.sum` and writes one value to `partials[pid]`.
2. **Stage 2:** a kernel launched with **one** program that loops over `partials` in chunks and writes the final value.
3. No `torch.sum` on the host and no atomics, so the result is the same on every run.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a PyTorch reference that sums in float64; your float32 tree sum is graded with a small tolerance.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def vector_sum(x):
    """x: float32 CUDA tensor. Return a 0-d float32 tensor with the sum, using two Triton stages."""
    raise NotImplementedError


@triton.jit
def _partial_sum_kernel(x_ptr, partial_ptr, n, BLOCK: tl.constexpr):
    pass


@triton.jit
def _final_sum_kernel(partial_ptr, out_ptr, n_partials, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _partial_sum_kernel(x_ptr, partial_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)    # padding lanes add 0
    tl.store(partial_ptr + pid, tl.sum(x, axis=0))


@triton.jit
def _final_sum_kernel(partial_ptr, out_ptr, n_partials, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)               # lane-wise accumulator
    for start in range(0, n_partials, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(partial_ptr + offs, mask=offs < n_partials, other=0.0)
    tl.store(out_ptr, tl.sum(acc, axis=0))                  # one tree reduction at the end


def vector_sum(x):
    x = x.contiguous().view(-1)
    n = x.numel()
    out = torch.zeros((), dtype=torch.float32, device=x.device)
    if n == 0:
        return out
    BLOCK = 1024
    n_partials = triton.cdiv(n, BLOCK)
    partials = torch.empty(n_partials, dtype=torch.float32, device=x.device)
    _partial_sum_kernel[(n_partials,)](x, partials, n, BLOCK=BLOCK)
    _final_sum_kernel[(1,)](partials, out, n_partials, BLOCK=1024)
    return out
'''

REFERENCE = '''import torch

def vector_sum(x):
    return x.to(torch.float64).sum().to(torch.float32)
'''

HINTS = [
    'Stage 1 is an elementwise kernel that ends in `tl.sum(x, axis=0)`: load the block with `other=0.0` so masked lanes contribute nothing, then store one scalar at `partial_ptr + pid`.',
    'Stage 2 runs a single program. Keep a `tl.zeros([BLOCK], tl.float32)` accumulator, add one chunk of partials per loop iteration, and call `tl.sum` once after the loop.',
    'On the host, `n_partials = triton.cdiv(n, BLOCK)`; launch stage 1 with that grid and stage 2 with `(1,)`. Handle `n == 0` before launching anything.',
]

_r = np.random.default_rng(5)
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),),
    (np.array([[0.5, -0.5], [10.0, -2.5]], dtype=np.float32),),
    (np.array([42.0], dtype=np.float32),),
    (np.zeros(0, dtype=np.float32),),
    (np.ones(5000, dtype=np.float32),),
    (np.concatenate([np.full(3000, 1e3), np.full(3000, -1e3), [0.25]]).astype(np.float32),),
    (_r.normal(size=20000).astype(np.float32),),
    (_r.uniform(0, 1, size=(64, 33)).astype(np.float32),),
]

EXPLANATION = r"""
## Why two stages
Programs in one launch cannot wait for each other, so a single kernel can't know when all partial sums are ready. Ending the first kernel is the global barrier: the second launch starts only after every partial is written. With `BLOCK = 1024`, 20 000 elements become 20 partials, which one program finishes in one loop iteration. For $n$ up to a million there are about a thousand partials, still one cheap program.

## Inside a program
`tl.sum` is a tree reduction across the block: the compiler reduces within each thread, then across a warp with shuffles, then across warps through shared memory. You write one line.

## Accuracy
A tree sum has error growing like $O(\log n)\,\varepsilon$ instead of the $O(n)\,\varepsilon$ of a sequential loop, which is why GPU sums of float32 are usually *more* accurate than a naive CPU loop. Test 6 sums large values that cancel; the order in which they are added matters at float32 precision, so the result is graded against a float64 reference with tolerance.

## Alternatives
- `tl.atomic_add(out_ptr, block_sum)` in stage 1: one kernel, but the addition order varies between runs, so results are not bit-reproducible.
- A persistent kernel with a fixed number of programs that each loop over many blocks: fewer partials and better for small inputs.

## Interview follow-ups
- How would you reduce along the last axis of an `(M, N)` tensor instead (one program per row)?
- Why do training frameworks offer a "deterministic algorithms" flag, and what does it disable?
"""
