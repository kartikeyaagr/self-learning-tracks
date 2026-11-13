import numpy as np

META = dict(title='FlashAttention forward kernel in Triton', tags=['GPU Programming', 'Attention', 'Inference'],
            diff='Hard', runtime='triton', tol=[1e-4, 1e-5], lessons=['ieng:14.03', 'ieng:14.02'],
            related=['online-softmax', 'flashattention-with-online-softmax', 'scaled-dot-product-attention-2'])

STATEMENT = r"""
Standard attention writes the $T \times T$ score matrix to memory, softmaxes it and reads it back. FlashAttention keeps a tile of queries in registers, streams key/value tiles past it, and folds each tile into a running softmax, so the scores never leave the chip.

Write `flash_attn_fwd(q, k, v, causal)`.

## Inputs
- `q`, `k`, `v`: contiguous `float32` CUDA tensors of shape `(B, H, T, D)`. $T \ge 1$ is arbitrary (not a multiple of the tile); $D \in \{16, 32, 64, 128\}$.
- `causal`: Python bool. If true, query $i$ attends only to keys $j \le i$.

## Output
A tuple `(o, lse)`:
- `o`: `float32` `(B, H, T, D)`, $o_i = \sum_j \mathrm{softmax}_j(s\, q_i \cdot k_j)\, v_j$ with $s = 1/\sqrt{D}$.
- `lse`: `float32` `(B, H, T)`, the natural-log log-sum-exp of each query row's scaled (and masked) scores, $\log\sum_j e^{s\, q_i\cdot k_j}$. The backward pass needs it.

## Requirements
- Grid `(triton.cdiv(T, BLOCK_M), B * H)`: program `(pid_m, bh)` owns query rows `pid_m*BLOCK_M + [0, BLOCK_M)` of head `bh`. Use `BLOCK_M = BLOCK_N = 64`, all `tl.constexpr`.
- Keep per row a running max `m`, running normaliser `l` and an unnormalised `(BLOCK_M, D)` accumulator, all in `float32`. For every key tile: scores, new max, rescale factor `alpha = exp(m - m_new)`, then update `l` and `acc`.
- Mask keys at positions $\ge T$ (score $-\infty$), and with `causal`, keys $j > i$. With `causal`, stop the loop after the tile that contains the diagonal: later tiles are fully masked.
- Use `tl.dot(..., input_precision="ieee")` so float32 inputs are not rounded to TF32.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import math
import torch
import triton
import triton.language as tl

def flash_attn_fwd(q, k, v, causal):
    """q, k, v: (B, H, T, D) float32 CUDA. Return (o (B, H, T, D), lse (B, H, T))."""
    raise NotImplementedError


@triton.jit
def _attn_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, T, sm_scale,
                     D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    pass
'''

SOLUTION = '''import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, T, sm_scale,
                     D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)
    base = bh.to(tl.int64) * T * D                    # start of this (batch, head) slice, contiguous (T, D)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < T

    q = tl.load(q_ptr + base + offs_m[:, None] * D + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    m = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, T)     # tiles past the diagonal are fully masked: skip them
    else:
        hi = T
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        col_ok = cols < T
        kt = tl.load(k_ptr + base + cols[None, :] * D + offs_d[:, None], mask=col_ok[None, :], other=0.0)  # (D, BN)
        v = tl.load(v_ptr + base + cols[:, None] * D + offs_d[None, :], mask=col_ok[:, None], other=0.0)   # (BN, D)
        s = tl.dot(q, kt, input_precision="ieee") * sm_scale                                               # (BM, BN)
        keep = col_ok[None, :]
        if CAUSAL:
            keep = keep & (cols[None, :] <= offs_m[:, None])
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))      # finite: key 0 is always visible in the first tile
        alpha = tl.exp(m - m_new)                     # rescale everything accumulated so far
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m = m_new

    o = acc / l[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * D + offs_d[None, :], o, mask=row_ok[:, None])
    tl.store(lse_ptr + bh.to(tl.int64) * T + offs_m, m + tl.log(l), mask=row_ok)


def flash_attn_fwd(q, k, v, causal):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, T, D = q.shape
    assert D in (16, 32, 64, 128) and k.shape == v.shape == q.shape
    o = torch.empty_like(q)
    lse = torch.empty((B, H, T), dtype=torch.float32, device=q.device)
    BLOCK_M = BLOCK_N = 64
    grid = (triton.cdiv(T, BLOCK_M), B * H)
    _attn_fwd_kernel[grid](q, k, v, o, lse, T, 1.0 / math.sqrt(D),
                           D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=bool(causal), num_warps=4)
    return o, lse
'''

REFERENCE = '''import math
import torch

def flash_attn_fwd(q, k, v, causal):
    q, k, v = q.double(), k.double(), v.double()
    T, D = q.shape[-2], q.shape[-1]
    s = q @ k.transpose(-1, -2) / math.sqrt(D)
    if causal:
        future = torch.ones(T, T, dtype=torch.bool).triu(1)
        s = s.masked_fill(future, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    o = torch.exp(s - lse[..., None]) @ v
    return o.float(), lse.float()
'''

HINTS = [
    'Load the `(BLOCK_M, D)` query tile once. Inside the loop load K already transposed, `k_ptr + base + cols[None, :] * D + offs_d[:, None]` gives a `(D, BLOCK_N)` tile, so `tl.dot(q, kt)` is the score tile.',
    'Online softmax per row: `m_new = max(m, rowmax(s))`, `alpha = exp(m - m_new)`, `p = exp(s - m_new[:, None])`, `l = l * alpha + rowsum(p)`, `acc = acc * alpha[:, None] + dot(p, v)`. Set masked scores to `-inf` first so they contribute `exp(-inf) = 0`.',
    'With `CAUSAL`, the loop bound is `min((pid_m + 1) * BLOCK_M, T)` and the in-tile mask is `cols[None, :] <= offs_m[:, None]`. At the end `o = acc / l[:, None]` and `lse = m + log(l)`; mask both stores with `offs_m < T`.',
]

_r = np.random.default_rng(71)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
_eye = np.zeros((1, 1, 3, 16), dtype=np.float32)
_eye[0, 0, [0, 1, 2], [0, 1, 2]] = 4.0
_v = np.zeros((1, 1, 3, 16), dtype=np.float32)
_v[0, 0, :, 0] = [1.0, 2.0, 3.0]
TESTS = [
    (_eye, _eye, _v, False),
    (_eye, _eye, _v, True),
    (_f(1, 1, 1, 16), _f(1, 1, 1, 16), _f(1, 1, 1, 16), True),
    (_f(1, 2, 64, 32), _f(1, 2, 64, 32), _f(1, 2, 64, 32), False),
    (_f(2, 1, 100, 16), _f(2, 1, 100, 16), _f(2, 1, 100, 16), True),
    (_f(1, 1, 130, 64), _f(1, 1, 130, 64), _f(1, 1, 130, 64), False),
    (_f(1, 3, 77, 32) * 3, _f(1, 3, 77, 32) * 3, _f(1, 3, 77, 32), True),
    (_f(1, 1, 65, 128), _f(1, 1, 65, 128), _f(1, 1, 65, 128), True),
]

EXPLANATION = r"""
## Why the running rescale is exact
After processing tiles $1..c$, row $i$ holds $m = \max$ of the scores seen, $l = \sum e^{s_j - m}$ and $\text{acc} = \sum e^{s_j - m} v_j$. When a new tile raises the max to $m'$, multiplying both $l$ and $\text{acc}$ by $e^{m - m'}$ re-expresses every old term relative to $m'$. After the last tile $\text{acc}/l = \sum_j \mathrm{softmax}_j\, v_j$ exactly, in any tile order. $\mathrm{lse} = m + \log l$ falls out for free.

## Memory traffic
Naive attention reads and writes $O(T^2)$ scores per head. Here each program reads its Q tile once and every K/V tile once, so HBM traffic is $O(T D \cdot T / \text{BLOCK\_M})$ and extra memory is $O(T)$ for `lse`. The FLOPs are unchanged (about $4 T^2 D$ per head); what changes is that the kernel becomes compute-bound instead of memory-bound. Causal masking halves the work by skipping tiles above the diagonal.

## Masks and edge cases
- Keys past $T$ get score $-\infty$; queries past $T$ are loaded as zeros and never stored. Tests 5 to 8 use $T = 100, 130, 77, 65$, so the last tile is partial.
- With `causal`, row $i$ always sees key 0 in the first tile, so $m$ becomes finite immediately and `alpha = exp(-inf - finite) = 0` only happens on the first step, where `l` and `acc` are already zero. A row that saw only $-\infty$ would compute $\exp(-\infty + \infty) = $ NaN, which is why fully masked rows need special care in variants like sliding-window attention.
- Test 1 uses one-hot queries and keys (score $4 \cdot 4 / 4 = 4$ on the diagonal, 0 elsewhere), and test 2 shows the causal version: row 0 can only see itself, so $o_0 = v_0$.

## Performance notes
Real kernels use `tl.exp2` with $\log_2 e$ folded into the scale (one fewer multiply), half-precision Q/K/V with float32 accumulation, pipelining (`num_stages`) and, for causal, splitting the loop into "full" tiles with no mask and the one diagonal tile.

## Common bugs
- Rescaling `l` but not `acc` (or the reverse).
- Masking with 0 instead of $-\infty$: a masked score of 0 still gets weight $e^{-m}$.
- Using `T` as the causal loop bound (correct but twice the work) or `pid_m * BLOCK_M` (skips the diagonal tile: wrong).
- Leaving TF32 on: about $10^{-3}$ relative error.

## Interview follow-ups
- What does the backward pass recompute instead of storing, and why does it need `lse`?
- For decoding ($T_q = 1$, long $T_k$), this grid has a single program per head. How does split-K ("flash-decoding") fix the occupancy problem, and how are the partial results merged?
"""
