import numpy as np

META = dict(title='Batched fp16 matmul with a fused ReLU epilogue', tags=['GPU Programming', 'Neural Networks'], diff='Hard',
            runtime='triton', tol=[2e-3, 2e-3], lessons=['ieng:13.03', 'ieng:13.06'],
            related=['fused-matmul-bias-relu', 'autotuned-matrix-multiplication', 'tiled-triton-row-sum'])

STATEMENT = r"""
A tiled matmul computes each `(BLOCK_M, BLOCK_N)` output tile by sliding over $K$ in steps of `BLOCK_K`, accumulating `tl.dot` products in `float32`. Applying ReLU **before** the tile is written (the epilogue) means the pre-activation never touches global memory. Here the operands are half precision and there is a batch of independent matrices.

Write `bmm_relu(a, b)`.

## Inputs
- `a`: `float16` CUDA tensor `(B, M, K)`; `b`: `float16` CUDA tensor `(B, K, N)`. Both contiguous, all sizes $\ge 1$ and not necessarily multiples of the tile sizes.

## Output
- `float16` tensor `(B, M, N)`: $\;y_{z} = \max(0,\ a_{z} b_{z})$ for each batch index $z$, with the products accumulated in `float32` and rounded to `float16` once, at the end.

## Requirements
- A 2-D launch grid: axis 0 enumerates output tiles of one matrix (`cdiv(M, BM) * cdiv(N, BN)`), axis 1 the batch index. Address each operand through its three strides (batch, row, column).
- Tile sizes of at least 16 in every dimension (`tl.dot` requires it). Mask the $M$, $N$ and $K$ edges, loading zeros outside, so partial tiles contribute nothing.
- Accumulate in a `float32` tile; apply ReLU and cast to `float16` in the epilogue; mask the store.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def bmm_relu(a, b):
    """a: (B, M, K) fp16, b: (B, K, N) fp16 CUDA tensors. Return relu(a @ b) as (B, M, N) fp16."""
    raise NotImplementedError


@triton.jit
def _bmm_relu_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                     s_ab, s_am, s_ak, s_bb, s_bk, s_bn, s_cb, s_cm, s_cn,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _bmm_relu_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                     s_ab, s_am, s_ak, s_bb, s_bk, s_bn, s_cb, s_cm, s_cn,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(axis=0)
    z = tl.program_id(axis=1).to(tl.int64)               # batch index
    n_tiles_n = tl.cdiv(N, BN)
    pm, pn = tile // n_tiles_n, tile % n_tiles_n
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_base = a_ptr + z * s_ab
    b_base = b_ptr + z * s_bb

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(a_base + rm[:, None] * s_am + kk[None, :] * s_ak,
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_base + kk[:, None] * s_bk + rn[None, :] * s_bn,
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)                               # fp16 x fp16 -> fp32 accumulate on tensor cores

    out = tl.maximum(acc, 0.0).to(tl.float16)              # epilogue: ReLU in fp32, one rounding
    c = c_ptr + z * s_cb + rm[:, None] * s_cm + rn[None, :] * s_cn
    tl.store(c, out, mask=(rm[:, None] < M) & (rn[None, :] < N))


def bmm_relu(a, b):
    assert a.dtype == b.dtype == torch.float16 and a.ndim == b.ndim == 3
    B, M, K = a.shape
    _, _, N = b.shape
    c = torch.empty((B, M, N), dtype=torch.float16, device=a.device)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), B)
    _bmm_relu_kernel[grid](a, b, c, M, N, K,
                           a.stride(0), a.stride(1), a.stride(2),
                           b.stride(0), b.stride(1), b.stride(2),
                           c.stride(0), c.stride(1), c.stride(2),
                           BM=BM, BN=BN, BK=BK)
    return c
'''

REFERENCE = '''import torch

def bmm_relu(a, b):
    return torch.relu(torch.bmm(a.to(torch.float64), b.to(torch.float64))).to(torch.float16)
'''

HINTS = [
    'Program `(tile, z)`: split `tile` into a row-tile and column-tile index with `// cdiv(N, BN)` and `% cdiv(N, BN)`; offset both operands by `z * batch_stride` before building the 2-D pointer grids.',
    'Inside the K loop, load an `(BM, BK)` tile of `a` and a `(BK, BN)` tile of `b`, each masked on both of its axes with `other=0.0`. Zeros past $K$ add nothing to the dot product, which is why the K edge is safe.',
    'Keep `acc` in float32, then `tl.maximum(acc, 0.0).to(tl.float16)` after the loop. Rounding each partial product to fp16 would lose precision and could overflow for large $K$.',
]

_r = np.random.default_rng(41)


def _ab(B, M, K, N, s=0.5):
    return ((_r.normal(size=(B, M, K)) * s).astype(np.float16), (_r.normal(size=(B, K, N)) * s).astype(np.float16))


TESTS = [
    (np.array([[[1.0, 2.0], [-1.0, 0.5]]], dtype=np.float16), np.array([[[1.0, -1.0], [1.0, 1.0]]], dtype=np.float16)),
    (np.array([[[2.0]], [[-3.0]]], dtype=np.float16), np.array([[[4.0]], [[5.0]]], dtype=np.float16)),
    _ab(2, 33, 17, 45),
    _ab(3, 64, 64, 64),
    _ab(1, 100, 300, 70, 0.2),
    _ab(4, 1, 129, 5),
    _ab(2, 130, 16, 1),
    _ab(1, 17, 1000, 33, 0.1),
]

EXPLANATION = r"""
## Why it works
Output tile $(m, n)$ of matrix $z$ is $\sum_{k\text{-tiles}} A_{z}[m, k] B_{z}[k, n]$. Each program keeps that partial sum in registers as a float32 tile while streaming $K$, so each element of $A$ and $B$ is loaded once per output tile rather than once per output element. ReLU is nonlinear, so it must be applied to the finished sum, after the loop. Doing it there costs one instruction per output instead of a separate kernel that would re-read and re-write the whole `(B, M, N)` result.

## Precision
fp16 products are exact in fp32, and the float32 accumulator keeps the sum accurate even for $K = 1000$ (test 8). The only rounding to fp16 happens once at the store. Accumulating in fp16 would lose about 3 decimal digits on long sums and overflow beyond 65504.

## Performance
Arithmetic intensity per tile is about $\frac{2\,BM\,BN\,BK}{2(BM + BN)BK}$ FLOPs per byte $= \frac{BM\,BN}{BM + BN}$: 32 for $64 \times 64$ tiles, enough to be compute-bound on tensor cores for large matrices. The batch axis of the grid gives parallelism even when each matrix is small (test 6: $M = 1$, a batched GEMV, is memory-bound regardless).

## Common bugs
- Masking only the M and N edges: the last K tile reads past the end of each row of $A$, which is memory of the **next** row or the next batch, silently adding garbage.
- Forgetting the batch stride on the output pointer, so every batch writes over matrix 0.
- Using `program_id(0)` ordering that walks rows of tiles: fine for correctness, but grouped ordering improves L2 reuse for large $M, N$.

## Interview follow-ups
- How would you fuse a bias and a residual add into the same epilogue, and what extra memory traffic does each add?
- For the backward pass, which mask do you need from the forward, and is it cheaper to store it (1 bit per element) or recompute it from the saved output?
"""
