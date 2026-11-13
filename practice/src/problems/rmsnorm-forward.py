import numpy as np

META = dict(title='Fused residual add and RMSNorm', tags=['GPU Programming', 'Transformers'], diff='Medium',
            runtime='triton', tol=[1e-4, 1e-5], lessons=['ieng:13.05', 'air-fanout-v1:05.04'],
            related=['layernorm-forward', 'l2-vector-norm', 'norm-gated-linear-transform'])

STATEMENT = r"""
In a pre-norm LLM block, every sublayer starts by adding the previous sublayer's output to the residual stream and normalising the sum with RMSNorm:

$$h = x + r, \qquad y_{ij} = \frac{h_{ij}}{\sqrt{\frac1N \sum_k h_{ik}^2 + \varepsilon}}\; w_j$$

Inference engines fuse both steps into one kernel that returns the updated residual stream $h$ (needed by the next residual add) and the normalised $y$ (fed to attention or the MLP).

Write `add_rms_norm(x, residual, w, eps)`.

## Inputs
- `x`, `residual`: contiguous `float32` CUDA tensors `(M, N)` with $1 \le N \le 8192$.
- `w`: `float32` `(N,)`. `eps`: Python float (e.g. `1e-6`).

## Output
- A tuple `(y, h)` of new `float32` `(M, N)` tensors. The inputs are not modified.

## Requirements
- One program per row, whole row in one block (`BLOCK = triton.next_power_of_2(N)`): read `x` and `residual` once, write `h` and `y` once.
- Note what RMSNorm leaves out compared with LayerNorm: no mean subtraction and no bias.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def add_rms_norm(x, residual, w, eps):
    """x, residual: (M, N), w: (N,), float32 CUDA. Return (rmsnorm(x + residual) * w, x + residual)."""
    raise NotImplementedError


@triton.jit
def _add_rms_norm_kernel(x_ptr, r_ptr, w_ptr, y_ptr, h_ptr, N, stride_row, eps, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _add_rms_norm_kernel(x_ptr, r_ptr, w_ptr, y_ptr, h_ptr, N, stride_row, eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    offs = row.to(tl.int64) * stride_row + cols
    h = tl.load(x_ptr + offs, mask=mask, other=0.0) + tl.load(r_ptr + offs, mask=mask, other=0.0)
    tl.store(h_ptr + offs, h, mask=mask)                     # updated residual stream
    inv_rms = tl.rsqrt(tl.sum(h * h, axis=0) / N + eps)      # padding lanes are 0 and add nothing
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + offs, h * inv_rms * w, mask=mask)


def add_rms_norm(x, residual, w, eps):
    x, residual = x.contiguous(), residual.contiguous()
    M, N = x.shape
    y, h = torch.empty_like(x), torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    _add_rms_norm_kernel[(M,)](x, residual, w, y, h, N, x.stride(0), eps, BLOCK=BLOCK, num_warps=num_warps)
    return y, h
'''

REFERENCE = '''import torch

def add_rms_norm(x, residual, w, eps):
    h = x + residual                                       # float32 add, exactly as the kernel does it
    h64 = h.to(torch.float64)
    y = h64 * torch.rsqrt(h64.pow(2).mean(dim=1, keepdim=True) + eps) * w.to(torch.float64)
    return y.to(torch.float32), h
'''

HINTS = [
    'Load both inputs with the same offsets `row * stride_row + cols` and add them. Store `h` right away: it is one of the outputs.',
    'The mean of squares is `tl.sum(h * h, axis=0) / N`. Padding lanes loaded as 0 contribute nothing, so no `where` is needed here (unlike LayerNorm).',
    '`inv_rms = tl.rsqrt(mean_sq + eps)`, then store `h * inv_rms * w`. There is no mean subtraction and no bias.',
]

_r = np.random.default_rng(53)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
TESTS = [
    (np.array([[1.0, 2.0, 2.0]], dtype=np.float32), np.array([[2.0, 0.0, 2.0]], dtype=np.float32),
     np.ones(3, dtype=np.float32), 1e-6),
    (np.array([[3.0, -4.0], [1.0, 1.0]], dtype=np.float32), np.zeros((2, 2), dtype=np.float32),
     np.array([1.0, 0.5], dtype=np.float32), 1e-6),
    (np.array([[1.0, -1.0, 0.5, 0.0]], dtype=np.float32), np.array([[-1.0, 1.0, -0.5, 0.0]], dtype=np.float32),
     np.ones(4, dtype=np.float32), 1e-6),
    (_f(4, 2048), _f(4, 2048), _f(2048), 1e-6),
    (_f(8, 4096) * 30, _f(8, 4096), _f(4096), 1e-5),
    (_f(5, 100), _f(5, 100), _f(100), 1e-6),
    (np.full((2, 1), 0.5, dtype=np.float32), np.full((2, 1), 0.25, dtype=np.float32),
     np.array([3.0], dtype=np.float32), 1e-6),
]

EXPLANATION = r"""
## Why fuse the residual add
Unfused, a decoder layer writes `h = x + r` (one read of each input, one write), then the norm reads `h` again and writes `y`. The fused kernel never re-reads `h`: it reads $8N$ bytes and writes $8N$ per row instead of reading $12N$ and writing $8N$. Every pre-norm sublayer has this pattern, twice per layer, which is why vLLM, TensorRT-LLM and similar engines ship a dedicated `fused_add_rms_norm` op.

## RMSNorm vs LayerNorm
RMSNorm skips the mean subtraction and the bias. That removes one reduction and one parameter vector, and models trained with it match LayerNorm's quality. The kernel is correspondingly simpler: padding lanes need no special masking in the reduction, because they are 0 and $0^2 = 0$.

## Edge cases
- **Zero row** (test 3: `x + r` cancels to 0): the mean square is 0 and `eps` keeps `rsqrt` finite, so `y = 0`.
- **$N = 1$** (test 7): $y = h / \sqrt{h^2 + \varepsilon}\cdot w \approx \operatorname{sign}(h)\, w$.
- **Large activations** (test 5): outlier-scale values are where float16 inputs would overflow the sum of squares, which is why real kernels accumulate in float32 even for bf16 tensors.

## Interview follow-ups
- Write the backward pass of RMSNorm. Which row statistic would you save from the forward?
- Some engines write `h` in place into `residual`. What does that save, and what does it break?
"""
