import numpy as np

META = dict(title='Online softmax for long rows', tags=['GPU Programming', 'Attention'], diff='Medium',
            runtime='triton', tol=[1e-5, 1e-7], lessons=['ieng:14.02', 'ieng:13.05'],
            related=['fused-row-wise-softmax', 'row-wise-logsumexp', 'flashattention-forward'])

STATEMENT = r"""
When a row is too long to sit in registers (a 128k vocabulary of logits, say), the fused one-block softmax no longer works. The **online softmax** computes the max and the normaliser in a single streaming pass by rescaling the running sum each time the max grows, and then a second pass writes the probabilities. That is two reads instead of the three of a naive chunked version (max, sum, write).

Write `online_softmax(x, temperature)` returning the softmax of `x / temperature` along each row.

## Inputs
- `x`: contiguous `float32` CUDA tensor of shape `(M, N)` with **finite** entries; $N$ can be far larger than the block.
- `temperature`: Python float, $T > 0$.

## Output
- `float32` tensor `(M, N)`: $p_{ij} = e^{(x_{ij} - m_i)/T} / d_i$, where $m_i = \max_j x_{ij}$ and $d_i = \sum_j e^{(x_{ij} - m_i)/T}$.

## Requirements
- One program per row with a fixed `BLOCK` (use 1024 or smaller). Do not size `BLOCK` from `N`.
- **Pass 1** keeps two scalars, the running max $m$ and normaliser $d$, and per chunk updates
  $$m' = \max\bigl(m, \max(\text{chunk})\bigr), \qquad d' = d\, e^{(m - m')/T} + \sum_{\text{chunk}} e^{(x - m')/T}$$
- **Pass 2** re-reads the row and stores $e^{(x - m)/T} / d$.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def online_softmax(x, temperature):
    """x: (M, N) float32 CUDA tensor, finite. Return softmax(x / temperature) per row, two passes per row."""
    raise NotImplementedError


@triton.jit
def _online_softmax_kernel(x_ptr, y_ptr, N, stride_row, inv_t, BLOCK: tl.constexpr):
    # pass 1: running (m, d); pass 2: write exp((x - m) * inv_t) / d
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _online_softmax_kernel(x_ptr, y_ptr, N, stride_row, inv_t, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    x_row = x_ptr + row.to(tl.int64) * stride_row
    y_row = y_ptr + row.to(tl.int64) * stride_row
    m = float("-inf")       # running max of the raw logits
    d = 0.0                 # running sum of exp((x - m) / T)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < N, other=float("-inf"))
        m_new = tl.maximum(m, tl.max(x, axis=0))           # finite: every chunk has a real entry
        # old terms were relative to m: rescale them to m_new; padding lanes give exp(-inf) = 0
        d = d * tl.exp((m - m_new) * inv_t) + tl.sum(tl.exp((x - m_new) * inv_t), axis=0)
        m = m_new
    inv_d = 1.0 / d
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0)
        tl.store(y_row + cols, tl.exp((x - m) * inv_t) * inv_d, mask=mask)


def online_softmax(x, temperature):
    assert temperature > 0
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = 1024
    _online_softmax_kernel[(M,)](x, y, N, x.stride(0), 1.0 / temperature, BLOCK=BLOCK)
    return y
'''

REFERENCE = '''import torch

def online_softmax(x, temperature):
    return torch.softmax(x.to(torch.float64) / temperature, dim=1).to(torch.float32)
'''

HINTS = [
    'Carry two scalars through the first loop: `m = float("-inf")` and `d = 0.0`. Load padding lanes as `-inf` so they never affect the max and contribute `exp(-inf) = 0`.',
    'Per chunk: `m_new = max(m, tl.max(x))`, then `d = d * exp((m - m_new) / T) + sum(exp((x - m_new) / T))`. On the first chunk `m = -inf`, so the old `d = 0` is multiplied by 0, which is fine.',
    'Precompute `inv_t = 1 / temperature` on the host and multiply. The second loop reloads each chunk and stores `exp((x - m) * inv_t) / d`.',
]

_r = np.random.default_rng(29)
_jump = np.concatenate([np.zeros(1024), np.full(1024, 40.0), np.zeros(10)]).astype(np.float32)[None, :]
TESTS = [
    (np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32), 1.0),
    (np.array([[1.0, 2.0, 3.0]], dtype=np.float32), 0.5),
    (_jump, 1.0),
    (_r.normal(size=(3, 5000)).astype(np.float32) * 4, 1.0),
    (_r.normal(size=(2, 3000)).astype(np.float32) * 4, 0.1),
    (np.full((2, 2049), 1000.0, dtype=np.float32), 1.0),
    (_r.normal(size=(5, 77)).astype(np.float32), 2.0),
    (np.array([[-3.0]], dtype=np.float32), 0.7),
]

EXPLANATION = r"""
## Why the rescale is exact
After some chunks, $d = \sum_{\text{seen}} e^{(x - m)/T}$. When the max rises to $m'$, each seen term should have been $e^{(x - m')/T} = e^{(x - m)/T} e^{(m - m')/T}$, so multiplying the whole sum by $e^{(m - m')/T}$ fixes all of them at once. By the end, $d$ is exactly the normaliser for the final max, and the running max guarantees no exponent ever exceeds 0. Test 3 jumps from 0 to 40 at a chunk boundary, which shrinks the earlier sum by $e^{-40}$.

## Memory traffic
| Version | Reads of `x` | Writes |
|---|---|---|
| naive chunked (max, sum, normalise) | 3 | 1 |
| online softmax | 2 | 1 |
| fused one-block (row fits in registers) | 1 | 1 |

For a memory-bound op that is a 25% saving over the naive version. FlashAttention takes it one step further: the second pass is never needed, because the output there is a weighted sum of values that can itself be rescaled on the fly.

## Temperature
Dividing by $T$ after subtracting the max (here, multiplying by $1/T$) is the same as dividing first and keeps the max at exponent 0. Low temperatures (test 5) make the distribution peaky and stress the stability.

## Common bugs
- Forgetting to rescale `d` when the max changes. The result is then right only when the max sits in the first chunk.
- Using the final `m` with a `d` that was not rescaled to it.
- Loading padding as 0 in pass 1: a row of negative logits then gets a wrong max.

## Interview follow-ups
- How do two partial results $(m_1, d_1)$ and $(m_2, d_2)$ from different programs combine? (Same formula, with $d_2$ also rescaled.)
- Extend pass 1 to also produce the log-sum-exp and use it for a fused cross-entropy loss.
"""
