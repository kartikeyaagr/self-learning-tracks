import numpy as np

META = dict(title='Inverted dropout kernel from a noise tensor', tags=['GPU Programming', 'Training'], diff='Easy',
            runtime='triton', tol=[1e-6, 1e-7], lessons=['ieng:13.04', 'ieng:13.02'],
            related=['dropout', 'fused-multiply-add', 'silu'])

STATEMENT = r"""
Dropout zeroes each activation with probability $p$ during training. **Inverted** dropout also scales the survivors by $\frac{1}{1-p}$ at training time, so the expected value of every output equals its input and inference needs no rescaling at all.

To keep the kernel testable, the randomness is handed to you: `u` holds one uniform sample per element. (A production kernel would draw it inside the kernel with `tl.rand(seed, offsets)`; see the follow-ups.)

Write `dropout(x, u, p)`.

## Inputs
- `x`: contiguous `float32` CUDA tensor, any shape (including 0 elements).
- `u`: contiguous `float32` CUDA tensor of the same shape, values in $[0, 1)$.
- `p`: Python float, $0 \le p \le 1$.

## Output
A new `float32` tensor of the shape of `x`:

$$y_i = \begin{cases} x_i \cdot \dfrac{1}{1-p} & \text{if } u_i \ge p \quad\text{(kept)}\\[4pt] 0 & \text{if } u_i < p \quad\text{(dropped)}\end{cases}$$

## Rules
- Ties: an element with $u_i = p$ exactly is **kept**.
- $p = 0$ returns a copy of `x`. $p = 1$ drops everything and returns zeros; compute the scale on the host as `1 / (1 - p)` if `p < 1` else `0.0`, so nothing divides by zero.
- Flat 1-D grid of `triton.cdiv(n, BLOCK)` programs, masked loads and stores; no launch for an empty tensor.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def dropout(x, u, p):
    """x, u: float32 CUDA tensors of one shape, u in [0, 1). Keep where u >= p, scale survivors by 1/(1-p)."""
    raise NotImplementedError


@triton.jit
def _dropout_kernel(x_ptr, u_ptr, out_ptr, n, p, scale, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _dropout_kernel(x_ptr, u_ptr, out_ptr, n, p, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    u = tl.load(u_ptr + offs, mask=mask)
    keep = u >= p                                   # ties (u == p) are kept
    tl.store(out_ptr + offs, tl.where(keep, x * scale, 0.0), mask=mask)


def dropout(x, u, p):
    assert x.shape == u.shape and 0.0 <= p <= 1.0
    x, u = x.contiguous(), u.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0     # p == 1 drops everything; never divide by zero
    BLOCK = 1024
    _dropout_kernel[(triton.cdiv(n, BLOCK),)](x, u, out, n, p, scale, BLOCK=BLOCK)
    return out
'''

REFERENCE = '''import torch

def dropout(x, u, p):
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    return torch.where(u >= p, x * scale, torch.zeros_like(x))
'''

HINTS = [
    'Same skeleton as any elementwise kernel: `offs = pid * BLOCK + tl.arange(0, BLOCK)`, `mask = offs < n`, load `x` and `u` with the mask.',
    'The keep decision is `u >= p` (so a tie keeps the element). `tl.where(keep, x * scale, 0.0)` gives the output in one expression.',
    'Compute `scale` on the host: `1 / (1 - p)` when `p < 1`, otherwise `0.0`. Pass `p` and `scale` as plain scalar kernel arguments.',
]

_r = np.random.default_rng(11)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
_u = lambda *s: _r.random(size=s).astype(np.float32)
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), np.array([0.9, 0.1, 0.5, 0.7], dtype=np.float32), 0.5),
    (np.array([[-2.0, 6.0], [1.5, 0.0]], dtype=np.float32), np.array([[0.3, 0.2], [0.8, 0.99]], dtype=np.float32), 0.25),
    (_f(37), _u(37), 0.0),
    (_f(37), _u(37), 1.0),
    (np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), 0.5),
    (_f(1025), _u(1025), 0.1),
    (_f(6, 333), _u(6, 333), 0.5),
    (_f(4096) * 10, _u(4096), 0.9),
]

EXPLANATION = r"""
## Why "inverted"
With keep probability $1-p$, $\mathbb E[y_i] = (1-p)\cdot\frac{x_i}{1-p} + p \cdot 0 = x_i$. Scaling at training time means the trained weights already expect activations of the right size, so `model.eval()` turns dropout into the identity: no extra multiply in the inference graph.

## The kernel
It is a masked elementwise kernel with a comparison and a select. Test 1 has a tie ($u = p = 0.5$ at index 2), which is kept and scaled to $6$. Test 4 ($p = 1$) must produce zeros without a division by zero, and test 3 ($p = 0$) must be an exact copy.

## Performance
Reading `x` and `u` and writing `y` is 12 bytes per element for a compare and a multiply: completely memory-bound. That is exactly why real kernels don't read a noise tensor. They generate it in registers with `tl.rand(seed, offsets)`, a counter-based Philox generator that maps `(seed, offset)` to a uniform number, and drop the `u` read, saving a third of the traffic. The backward pass reruns the same generator with the same seed to rebuild the mask instead of storing it.

## Common bugs
- Using `u > p`, which drops ties (test 1 catches it).
- Scaling the dropped elements too, or forgetting to scale survivors at all.
- Dividing by `1 - p` inside the kernel for `p = 1`.
- Comparing `u` against the *keep* probability instead of the drop probability.

## Interview follow-ups
- Rewrite the kernel with `tl.rand(seed, offs)`. How does the backward pass get the same mask without saving it, and why must the offsets be global element indices?
- How would you fuse dropout into the epilogue of the matmul that produces `x`, and what does that save?
"""
