import numpy as np

META = dict(title='Streaming row-wise log-sum-exp', tags=['GPU Programming', 'Language Models'], diff='Medium',
            runtime='triton', tol=[1e-5, 1e-5], lessons=['ieng:13.05', 'ieng:14.02'],
            related=['fused-row-wise-softmax', 'online-softmax', 'softmax-numerical-stability-and-log-sum-exp'])

STATEMENT = r"""
The log-sum-exp $\mathrm{lse}(x) = \log \sum_j e^{x_j}$ is the normaliser behind softmax and cross-entropy. For a 128k-token vocabulary a row does not fit in one block, so the kernel must **stream** over it: one read of each element, keeping a running maximum and a running sum that is rescaled whenever the maximum grows.

Write `logsumexp_rows(x)`.

## Inputs
- `x`: contiguous `float32` CUDA tensor of shape `(M, N)`, $M, N \ge 1$, with $N$ possibly much larger than the block.
- Entries are finite or `-inf`. A row may be **entirely** `-inf`.

## Output
- `float32` tensor of shape `(M,)` with $\mathrm{lse}$ of each row. An all-`-inf` row gives `-inf` (never NaN).

## Requirements
- One program per row, a loop over chunks of `BLOCK` columns, each element loaded once.
- Keep **lane-wise** running state `m` (max) and `s` (sum of $e^{x - m}$) of shape `[BLOCK]`, updated per chunk with
  $$m' = \max(m, x), \qquad s' = s\, e^{m - m'} + e^{x - m'}$$
  then combine the lanes after the loop: $M = \max_\ell m_\ell$, $S = \sum_\ell s_\ell\, e^{m_\ell - M}$, result $M + \log S$.
- Whenever a quantity like $-\infty - (-\infty)$ would appear, the corresponding term must count as 0.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def logsumexp_rows(x):
    """x: (M, N) float32 CUDA tensor, entries finite or -inf. Return (M,) float32 log-sum-exp per row."""
    raise NotImplementedError


@triton.jit
def _lse_kernel(x_ptr, out_ptr, N, stride_row, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _lse_kernel(x_ptr, out_ptr, N, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    base = x_ptr + row.to(tl.int64) * stride_row
    m = tl.full([BLOCK], float("-inf"), dtype=tl.float32)   # per-lane running max
    s = tl.zeros([BLOCK], dtype=tl.float32)                 # per-lane sum of exp(x - m)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < N, other=float("-inf"))
        m_new = tl.maximum(m, x)
        # a lane that has only seen -inf has m_new = -inf: keep its sum at 0 instead of exp(nan)
        alive = m_new > float("-inf")
        safe = tl.where(alive, m_new, 0.0)
        s = tl.where(alive, s * tl.exp(m - safe) + tl.exp(x - safe), 0.0)
        m = m_new
    big = tl.max(m, axis=0)
    # all-(-inf) row: shift by 0 instead, so total = 0 and the result is -inf + log(0) = -inf
    shift = tl.where(big > float("-inf"), big, 0.0)
    total = tl.sum(s * tl.exp(m - shift), axis=0)           # dead lanes: 0 * exp(-inf) = 0
    tl.store(out_ptr + row, big + tl.log(total))


def logsumexp_rows(x):
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = min(1024, triton.next_power_of_2(N))
    _lse_kernel[(M,)](x, out, N, x.stride(0), BLOCK=BLOCK)
    return out
'''

REFERENCE = '''import torch

def logsumexp_rows(x):
    return torch.logsumexp(x.to(torch.float64), dim=1).to(torch.float32)
'''

HINTS = [
    'Start each lane at `m = -inf`, `s = 0`. When a new chunk arrives, the old sum was measured relative to the old max, so multiply it by `exp(m - m_new)` before adding `exp(x - m_new)`.',
    'If a lane has seen only `-inf` so far, `m_new = -inf` and `x - m_new` is NaN. Compute `alive = m_new > -inf`, substitute 0 for the max in the exponents, and force `s = 0` on dead lanes with `tl.where`.',
    'After the loop: `big = tl.max(m)`; the result is `big + log(sum(s * exp(m - big)))`. For an all-`-inf` row, shift by 0 instead of `big`: the total is then 0 and `-inf + log(0)` is `-inf`.',
]

_r = np.random.default_rng(23)
_part = _r.normal(size=(3, 3000)).astype(np.float32)
_part[0, :2900] = -np.inf             # finite values only in the last chunk
_part[1, ::2] = -np.inf
_part[2, :] = -np.inf                 # fully masked row
TESTS = [
    (np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32),),
    (np.array([[1000.0, 1000.0, 1000.0], [-1000.0, -1000.0, -1000.0]], dtype=np.float32),),
    (np.array([[-np.inf, -np.inf], [-np.inf, 3.0]], dtype=np.float32),),
    (_part,),
    (np.array([[7.5]], dtype=np.float32),),
    (_r.normal(size=(4, 5000)).astype(np.float32) * 10,),
    (np.concatenate([np.full((2, 1024), -50.0), np.full((2, 1024), 50.0)], axis=1).astype(np.float32),),
    (_r.normal(size=(9, 129)).astype(np.float32),),
]

EXPLANATION = r"""
## Rescaling the running sum
Suppose the sum so far is $s = \sum e^{x - m}$ for the old max $m$. When a larger value raises the max to $m'$, every old term needs $e^{x - m'} = e^{x - m}\, e^{m - m'}$, so the whole sum is multiplied by $e^{m - m'} \le 1$. Nothing ever exceeds 1 before it is added, so nothing overflows, and each element is read exactly once. This is the same online-normaliser trick that FlashAttention uses.

## Lane-wise state
Keeping `m` and `s` per lane means the loop body has no cross-lane reductions at all, just elementwise `max`, `exp` and `where`. The single cross-lane combine happens once after the loop. The scalar alternative (reduce each chunk with `tl.max` and `tl.sum`, then update two scalars) is also correct, but it does two block reductions per chunk.

## The `-inf` cases
Masked logits are `-inf`. A lane that has seen nothing but `-inf` has $m = m' = -\infty$, and $m - m'$ is NaN. The `alive` guard keeps those lanes at $s = 0$. After the loop, a dead lane contributes $0 \cdot e^{-\infty} = 0$, and if every lane is dead the row result is `-inf` (test 4, row 3). Test 7 makes the max jump from -50 to 50 at a chunk boundary, which checks the rescaling.

## Interview follow-ups
- How does cross-entropy use the LSE to avoid materialising softmax probabilities over a huge vocabulary?
- How would you split a single 256k-wide row across several programs and merge their `(m, s)` pairs?
"""
