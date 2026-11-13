import numpy as np

META = dict(title='Masked tanh-GELU kernel for fp16 and fp32', tags=['GPU Programming', 'Activation Functions'], diff='Easy',
            runtime='triton', tol=[2e-3, 1e-5], lessons=['ieng:13.02', 'ieng:13.04'],
            related=['gelu', 'triton-row-wise-softmax', 'tiled-triton-row-sum'])

STATEMENT = r"""
The tanh approximation of GELU, used by GPT-2-style models:

$$\mathrm{gelu}(x) = \tfrac12\, x \left(1 + \tanh\!\bigl(\sqrt{2/\pi}\,(x + 0.044715\, x^3)\bigr)\right)$$

Write it as a Triton kernel that works for tensors of **any** shape and size, including sizes that are not a multiple of the block, and for both half and single precision.

Write `gelu_tanh(x)`.

## Inputs
- `x`: contiguous CUDA tensor of dtype `float16` or `float32`, any shape, possibly with 0 elements.

## Output
- A new tensor with the same shape and dtype as `x`.

## Requirements
- Treat the tensor as a flat array of `n = x.numel()` elements. A 1-D grid of `triton.cdiv(n, BLOCK)` programs, `BLOCK = 1024`; the last program must mask its out-of-range lanes on both load and store.
- Load, convert to `float32`, compute, and convert back to the output dtype on store. Do not compute in `float16`: $x^3$ overflows it for $|x| > 40$.
- $\tanh$ must not produce NaN for large $|x|$. Build it from `tl.exp` as $\tanh(u) = 1 - \dfrac{2}{e^{2u} + 1}$, which saturates cleanly to $\pm 1$.
- For `n == 0`, return an empty tensor without launching.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def gelu_tanh(x):
    """x: contiguous float16/float32 CUDA tensor. Return tanh-approximate GELU, same shape and dtype."""
    raise NotImplementedError


@triton.jit
def _gelu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl

SQRT_2_OVER_PI = 0.7978845608028654


@triton.jit
def _gelu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)     # int64: safe beyond 2**31 elements
    mask = offs < n                                          # the last block runs past the end
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    t = 1.0 - 2.0 / (tl.exp(2.0 * u) + 1.0)                  # tanh: exp -> inf gives 1, exp -> 0 gives -1
    y = 0.5 * x * (1.0 + t)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def gelu_tanh(x):
    assert x.is_contiguous() and x.dtype in (torch.float16, torch.float32)
    y = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return y
    BLOCK = 1024
    _gelu_kernel[(triton.cdiv(n, BLOCK),)](x, y, n, BLOCK=BLOCK)
    return y
'''

REFERENCE = '''import torch

def gelu_tanh(x):
    return torch.nn.functional.gelu(x.to(torch.float64), approximate='tanh').to(x.dtype)
'''

HINTS = [
    'Offsets are `pid * BLOCK + tl.arange(0, BLOCK)` and the mask is `offs < n`. Pass the same mask to `tl.load` and `tl.store`; a contiguous tensor of any shape is just a flat array of `numel()` elements.',
    'Upcast right after the load (`.to(tl.float32)`) and downcast at the store with `y.to(y_ptr.dtype.element_ty)`, so one kernel serves both dtypes.',
    'With $\\tanh(u) = 1 - 2/(e^{2u}+1)$: for $u = 300$, $e^{600} = \\infty$ in float32 and the result is exactly 1; for $u = -300$ it is $1 - 2 = -1$. The formula $(e^{2u}-1)/(e^{2u}+1)$ would give $\\infty/\\infty$ = NaN.',
]

_r = np.random.default_rng(2)
TESTS = [
    (np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),),
    (np.array([[0.5, -0.5], [3.0, -3.0]], dtype=np.float16),),
    ((_r.normal(size=3000) * 3).astype(np.float32),),
    ((_r.normal(size=(7, 333)) * 2).astype(np.float16),),
    (np.array([-60.0, -45.0, 45.0, 60.0, 300.0, -300.0], dtype=np.float32),),
    (np.array([-60.0, -45.0, 45.0, 60.0], dtype=np.float16),),
    (np.zeros((0, 4), dtype=np.float32),),
    (np.linspace(-6, 6, 1025, dtype=np.float32).reshape(1, 1025),),
]

EXPLANATION = r"""
## Why it works
Elementwise kernels need no communication between lanes: each program handles `BLOCK` consecutive elements, and the mask turns the tail of the last block into no-ops. Because a contiguous tensor's elements are laid out flat in memory, the same kernel handles every shape; only `numel()` matters. Test 8 has 1025 elements, so its second program has a single live lane.

## Precision
In `float16`, $x^3$ overflows to `inf` for $|x| > 40$ (the largest finite value is 65504), so the tanh argument becomes `inf` and the polynomial is lost. Computing in `float32` and casting once on store avoids this and costs nothing: the kernel is memory-bound, so arithmetic is free. Tests 5 and 6 probe the saturated region.

## Performance
Each element is read once (2 or 4 bytes) and written once. On a GPU with 2 TB/s, a 100M-element fp16 tensor takes about $400\text{ MB} / 2\text{ TB/s} = 0.2$ ms. Fusing GELU into the preceding matmul's epilogue removes even that round trip.

## Common bugs
- Missing mask on the store: the last program writes past the end of `y`, corrupting whatever follows it.
- Launching with an empty grid for `n == 0` (Triton raises on a zero-size grid).
- Writing $\tanh$ as $(e^{2u} - 1)/(e^{2u} + 1)$, which is NaN once $e^{2u}$ overflows.

## Interview follow-ups
- How would you write the backward kernel, and what would you save from the forward pass to make it cheap?
- When is the tanh approximation preferred to the exact erf form, and does it matter for a model trained with the other one?
"""
