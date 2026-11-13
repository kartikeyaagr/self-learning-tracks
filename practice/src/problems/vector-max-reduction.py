import numpy as np

META = dict(title='Max and argmax reduction', tags=['GPU Programming'], diff='Easy', runtime='triton',
            lessons=['ieng:13.03', 'ieng:13.04'], related=['vector-sum-reduction', 'fused-row-wise-softmax'])

STATEMENT = r"""
Greedy decoding picks the token with the largest logit, which is a max **and** an argmax over a long vector. Write it as a two-stage Triton reduction.

Write `vector_max(x)` returning `(value, index)`.

## Inputs
- `x`: contiguous 1-D `float32` CUDA tensor with $n \ge 1$ finite elements (no NaN).

## Output
- `value`: 0-d `float32` CUDA tensor, the maximum of `x`.
- `index`: 0-d `int64` CUDA tensor, the position of the maximum. **Ties go to the smallest index.**

## Requirements
1. **Stage 1:** `triton.cdiv(n, BLOCK)` programs. Each finds the max of its block and the *global* index of that max, and writes them to `block_max[pid]` and `block_idx[pid]`.
2. **Stage 2:** a single program reduces the partials the same way.
3. Masked lanes must never win: load them as `-inf`.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def vector_max(x):
    """x: 1-D float32 CUDA tensor, n >= 1. Return (max 0-d float32, argmax 0-d int64), ties -> smallest index."""
    raise NotImplementedError


@triton.jit
def _block_argmax_kernel(x_ptr, max_ptr, idx_ptr, n, BLOCK: tl.constexpr):
    pass


@triton.jit
def _final_argmax_kernel(max_ptr, idx_ptr, out_val_ptr, out_idx_ptr, n_blocks, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _block_argmax_kernel(x_ptr, max_ptr, idx_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=float("-inf"))   # padding can never win
    best = tl.max(x, axis=0)
    # smallest offset holding the max: offsets are increasing, so this is the first occurrence
    first = tl.min(tl.where(x == best, offs, n), axis=0)
    tl.store(max_ptr + pid, best)
    tl.store(idx_ptr + pid, first.to(tl.int64))


@triton.jit
def _final_argmax_kernel(max_ptr, idx_ptr, out_val_ptr, out_idx_ptr, n_blocks, BLOCK: tl.constexpr):
    # n_blocks <= BLOCK (checked on the host), so all partials fit in one block
    offs = tl.arange(0, BLOCK)
    mask = offs < n_blocks
    m = tl.load(max_ptr + offs, mask=mask, other=float("-inf"))
    i = tl.load(idx_ptr + offs, mask=mask, other=0)
    best = tl.max(m, axis=0)
    big = tl.full([BLOCK], 2**62, dtype=tl.int64)
    first = tl.min(tl.where(mask & (m == best), i, big), axis=0)       # tie -> smallest global index
    tl.store(out_val_ptr, best)
    tl.store(out_idx_ptr, first)


def vector_max(x):
    x = x.contiguous()
    n = x.numel()
    BLOCK = 1024
    n_blocks = triton.cdiv(n, BLOCK)
    FINAL = triton.next_power_of_2(n_blocks)
    assert FINAL <= 65536, "add a third stage for inputs this large"
    block_max = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    block_idx = torch.empty(n_blocks, dtype=torch.int64, device=x.device)
    _block_argmax_kernel[(n_blocks,)](x, block_max, block_idx, n, BLOCK=BLOCK)
    val = torch.empty((), dtype=torch.float32, device=x.device)
    idx = torch.empty((), dtype=torch.int64, device=x.device)
    _final_argmax_kernel[(1,)](block_max, block_idx, val, idx, n_blocks, BLOCK=FINAL)
    return val, idx
'''

REFERENCE = '''import torch

def vector_max(x):
    val = x.max()
    idx = torch.nonzero(x == val)[0, 0]     # first occurrence
    return val, idx.to(torch.int64)
'''

HINTS = [
    'Load with `other=float("-inf")` so padding lanes never become the max. `tl.max(x, axis=0)` gives the block maximum.',
    'For the index, keep the global offsets `offs` around: `tl.min(tl.where(x == best, offs, n), axis=0)` is the first position holding the max.',
    'Stage 2 does the same over `block_max`, but takes the minimum of the stored *global* indices among the tied blocks. Because blocks are in order, that is the first occurrence overall.',
]

_r = np.random.default_rng(8)
_tie = _r.normal(size=3000).astype(np.float32)
_tie[[2100, 1500, 2999]] = 9.0                   # three tied maxima in different blocks
_neg = -np.abs(_r.normal(size=1500)).astype(np.float32) - 1
TESTS = [
    (np.array([3.0, 7.0, 1.0, 7.0], dtype=np.float32),),
    (np.array([-5.0, -2.0, -9.0], dtype=np.float32),),
    (np.array([0.5], dtype=np.float32),),
    (_tie,),
    (_neg,),
    (np.arange(4096, dtype=np.float32),),
    (np.arange(4096, dtype=np.float32)[::-1].copy(),),
    (_r.normal(size=10001).astype(np.float32),),
]

EXPLANATION = r"""
## Carrying the index through the reduction
A max reduction is easy; the index is what makes it interesting. Inside a block, `tl.where(x == best, offs, n)` keeps the offsets that hold the maximum and replaces the rest with a sentinel larger than any real offset, and `tl.min` picks the first one. Stage 2 repeats the trick on the partials, using their stored global indices. Since every partial is the first occurrence within its block and blocks are ordered, the minimum global index among tied blocks is the first occurrence overall (test 4 has ties in three blocks).

`tl.argmax(x, axis=0)` also exists and returns the first occurrence within a block, but the explicit `where`/`min` form is the one that generalises to the cross-block stage.

## Why `-inf` padding
With `other=0.0`, an all-negative input (test 5) would report 0 at a padding position. That is the most common bug in max reductions.

## Performance
Memory-bound, one pass over `x`. Stage 2 is a single tiny program. In a real decoder the vocabulary (32k to 256k) is reduced per batch row, with one program per row and a loop over the vocabulary.

## Interview follow-ups
- How would you extend this to top-k (e.g. k = 50 for top-k sampling)?
- If the input could contain NaN, what should the max be, and how does `torch.max` treat it?
"""
