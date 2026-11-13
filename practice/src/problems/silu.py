import numpy as np

META = dict(title='SiLU forward and derivative in one kernel', tags=['GPU Programming', 'Activation Functions'],
            diff='Easy', runtime='triton', tol=[1e-5, 1e-6],
            lessons=['ieng:13.04', 'ieng:13.06'], related=['fused-multiply-add', 'fused-matmul-bias-relu'])

STATEMENT = r"""
SiLU (also called swish) is the activation inside the SwiGLU MLPs of most current LLMs:

$$\mathrm{silu}(x) = x\,\sigma(x), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}, \qquad \frac{d}{dx}\mathrm{silu}(x) = \sigma(x)\bigl(1 + x\,(1 - \sigma(x))\bigr)$$

Write `silu_fwd_bwd(x)` that computes **both** the activation and its derivative in a single Triton kernel: one read of `x`, two writes.

## Inputs
- `x`: contiguous `float32` CUDA tensor, any shape, at least one element.

## Output
- A tuple `(y, dy)` of two new `float32` CUDA tensors with the shape of `x`: `y = silu(x)` and `dy = silu'(x)`.

## Requirements
- Compute in `float32`. Large inputs of either sign (e.g. $\pm 100$) must give finite results: `silu(100) = 100`, `silu(-100) ≈ 0`, and the derivative tends to 1 and 0.
- Flat 1-D grid, `BLOCK` a `tl.constexpr` power of two, masked loads and stores.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def silu_fwd_bwd(x):
    """x: float32 CUDA tensor. Return (silu(x), d silu / dx) from one Triton kernel."""
    raise NotImplementedError


@triton.jit
def _silu_kernel(x_ptr, y_ptr, dy_ptr, n, BLOCK: tl.constexpr):
    # TODO: load a block of x, compute sigmoid once, store y and dy
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _silu_kernel(x_ptr, y_ptr, dy_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # exp(-x) overflows to inf for x < -88, and 1 / (1 + inf) = 0: still the right limit, no NaN
    s = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offs, x * s, mask=mask)
    tl.store(dy_ptr + offs, s * (1.0 + x * (1.0 - s)), mask=mask)   # reuses s: the fusion pays here


def silu_fwd_bwd(x):
    x = x.contiguous()
    y, dy = torch.empty_like(x), torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    _silu_kernel[(triton.cdiv(n, BLOCK),)](x, y, dy, n, BLOCK=BLOCK)
    return y, dy
'''

REFERENCE = '''import torch

def silu_fwd_bwd(x):
    x = x.to(torch.float32)
    s = torch.sigmoid(x)
    return x * s, s * (1 + x * (1 - s))
'''

HINTS = [
    'It is the same skeleton as an elementwise kernel: offsets, mask, one load, two stores to two output pointers.',
    'Compute `s = 1 / (1 + tl.exp(-x))` once and derive both outputs from it. Check what happens at `x = -100`: `exp(100)` is `inf` in float32, and `1 / (1 + inf)` is a clean 0.',
    '`dy = s * (1 + x * (1 - s))`. Allocate both outputs with `torch.empty_like(x)` and pass both pointers to the kernel.',
]

_r = np.random.default_rng(11)
TESTS = [
    (np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32),),
    (np.array([[3.0, -3.0], [0.5, -0.5]], dtype=np.float32),),
    (np.array([100.0, -100.0, 30.0, -30.0, 88.0, -88.0], dtype=np.float32),),
    (np.array([-1.2784645], dtype=np.float32),),
    (_r.normal(size=2049).astype(np.float32) * 4,),
    (_r.normal(size=(5, 7, 9)).astype(np.float32),),
    (np.linspace(-20, 20, 1000, dtype=np.float32),),
]

EXPLANATION = r"""
## Why fuse forward and derivative
A training step needs $\sigma(x)$ twice: once for the activation and once in the backward pass. Computing both from one load of `x` reads 4 bytes and writes 8 per element, instead of 8 read and 8 written for two separate kernels. Frameworks usually save `x` and recompute $\sigma$ in backward; this kernel shows the trade the other way.

## Numerical limits
For very negative `x`, `exp(-x)` overflows to `inf`, so `s = 0` exactly and `y = x * 0 = -0.0`. That is the correct limit and contains no NaN. A formulation like `x * exp(x) / (1 + exp(x))` would compute `inf / inf = NaN` for large positive `x`. Test 3 checks both ends. Test 4 is near the minimum of SiLU, where the derivative crosses zero.

## Performance
Memory-bound: about 12 bytes per element for a dozen flops. `tl.exp` lowers to the fast hardware `ex2` instruction plus a multiply, so arithmetic is free relative to DRAM traffic.

## Common bugs
- Computing the sigmoid twice (correct but slower) or with a formula that is NaN at the extremes.
- Writing both outputs to the same pointer.
- Forgetting `other=0.0` is harmless here, but loading masked lanes without a mask is not.

## Interview follow-ups
- Write the fused SwiGLU kernel: `silu(gate) * up` for two halves of one projection, and its backward.
- Why is `x * sigmoid(x)` not the same as GELU, and when is the tanh approximation of GELU used?
"""
