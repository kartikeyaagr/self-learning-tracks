import numpy as np

META = dict(title='Row-wise L2 norm and normalization', tags=['GPU Programming', 'Embeddings'], diff='Easy',
            runtime='triton', tol=[1e-5, 1e-6],
            lessons=['ieng:13.04', 'air-fanout-v1:02.04'], related=['vector-norms', 'vector-sum-reduction', 'rmsnorm-forward'])

STATEMENT = r"""
Embedding search L2-normalises every vector so that a dot product becomes a cosine similarity. Write a kernel that computes each row's L2 norm and the normalised row together.

Write `l2_normalize(x, eps)`.

## Inputs
- `x`: contiguous `float32` CUDA tensor of shape `(M, D)`, $M, D \ge 1$. `D` may be larger than one block.
- `eps`: Python float, e.g. `1e-12`.

## Output
A tuple `(y, norms)`:
- `norms`: `float32` tensor of shape `(M,)`, $\lVert x_i \rVert_2 = \sqrt{\sum_j x_{ij}^2}$.
- `y`: `float32` tensor of shape `(M, D)`, $y_i = x_i / \max(\lVert x_i\rVert_2,\ \varepsilon)$. An all-zero row therefore maps to all zeros (the same rule as `torch.nn.functional.normalize`).

## Requirements
- One program per row. First loop over the row accumulating squares in a lane-wise `float32` accumulator, reduce it once, then loop again to scale and store.
- `BLOCK` is a `tl.constexpr` power of two; mask every load and store.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def l2_normalize(x, eps):
    """x: (M, D) float32 CUDA tensor. Return (x / max(||x_i||, eps), norms (M,))."""
    raise NotImplementedError


@triton.jit
def _l2_normalize_kernel(x_ptr, y_ptr, norm_ptr, D, stride_row, eps, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _l2_normalize_kernel(x_ptr, y_ptr, norm_ptr, D, stride_row, eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    x_row = x_ptr + row.to(tl.int64) * stride_row
    y_row = y_ptr + row.to(tl.int64) * stride_row
    # pass 1: sum of squares, one partial per lane, reduced once after the loop
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, D, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < D, other=0.0)
        acc += x * x
    norm = tl.sqrt(tl.sum(acc, axis=0))
    tl.store(norm_ptr + row, norm)
    inv = 1.0 / tl.maximum(norm, eps)
    # pass 2: re-read the row (it is usually still in L2 cache) and scale
    for start in range(0, D, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0)
        tl.store(y_row + cols, x * inv, mask=mask)


def l2_normalize(x, eps):
    x = x.contiguous()
    M, D = x.shape
    y = torch.empty_like(x)
    norms = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = min(1024, triton.next_power_of_2(D))
    _l2_normalize_kernel[(M,)](x, y, norms, D, x.stride(0), eps, BLOCK=BLOCK)
    return y, norms
'''

REFERENCE = '''import torch

def l2_normalize(x, eps):
    norms = x.to(torch.float64).pow(2).sum(dim=1).sqrt()
    y = x.to(torch.float64) / norms.clamp(min=eps)[:, None]
    return y.to(torch.float32), norms.to(torch.float32)
'''

HINTS = [
    'Program `row` starts at `x_ptr + row * stride_row`. Loop `for start in range(0, D, BLOCK)` and add `x * x` into `acc = tl.zeros([BLOCK], tl.float32)`; masked lanes load 0 and add nothing.',
    'After the loop, `norm = tl.sqrt(tl.sum(acc, axis=0))`. Store it, then compute `inv = 1.0 / tl.maximum(norm, eps)` once.',
    'Loop over the row a second time, load each chunk again and store `x * inv` with the same mask.',
]

_r = np.random.default_rng(13)
TESTS = [
    (np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32), 1e-12),
    (np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32), 1e-12),
    (np.array([[1e-8, 0.0, 0.0]], dtype=np.float32), 1e-6),
    (np.array([[-2.0]], dtype=np.float32), 1e-12),
    (_r.normal(size=(5, 768)).astype(np.float32), 1e-12),
    (_r.normal(size=(3, 2500)).astype(np.float32) * 10, 1e-12),
    (_r.normal(size=(64, 17)).astype(np.float32), 1e-12),
    (_r.uniform(-1e-3, 1e-3, size=(2, 1030)).astype(np.float32), 1e-12),
]

EXPLANATION = r"""
## Shape of the kernel
This is the template for every row-wise "reduce, then use the result" kernel (softmax, LayerNorm, RMSNorm): one program per row, a reduction loop, then an apply loop. Keeping a lane-wise accumulator and reducing once after the loop is cheaper than calling `tl.sum` every iteration, and it is also slightly more accurate because it sums in a tree at the end.

## The `eps` rule
Dividing by $\max(\lVert x\rVert, \varepsilon)$ rather than $\lVert x\rVert + \varepsilon$ leaves normal rows exactly unit-norm, while an all-zero row (test 2) becomes zeros instead of NaN. Test 3 has a norm smaller than `eps`, so the row is divided by `eps`, not by its norm.

## Cost
The row is read twice and written once: $12D$ bytes per row. When $D \le$ `BLOCK`, a single-loop version that keeps the row in registers reads it only once; when $D$ is large, the second read usually hits L2 cache.

## Common bugs
- Accumulating `x` instead of `x * x`, or forgetting the square root.
- Using the same pointer arithmetic for `y` with a different stride. Here `y` is a fresh contiguous tensor with the same stride as `x`.
- Overflow: float32 squares overflow above about $1.8 \times 10^{19}$. Embedding values never get there, but a general-purpose norm should scale by the max first.

## Interview follow-ups
- How would you make this a single read of the row when `D` fits in one block?
- Why do retrieval systems store normalised vectors and use inner-product search instead of cosine search?
"""
