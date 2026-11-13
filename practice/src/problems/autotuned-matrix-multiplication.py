import numpy as np

META = dict(title='Autotuned tiled matmul', tags=['GPU Programming', 'Linear Algebra'], diff='Medium', runtime='triton',
            tol=[1e-4, 1e-4], lessons=['ieng:13.03', 'air-fanout-v1:03.02'],
            related=['fused-matmul-bias-relu', 'grouped-program-id-matmul', 'block-pointer-matmul'])

STATEMENT = r"""
The tile sizes that make a matmul fast depend on the GPU and on the problem shape, so Triton lets you list candidate configurations and benchmark them on first use with `@triton.autotune`.

Write `matmul(a, b)` returning `a @ b` from an autotuned Triton kernel.

## Inputs
- `a`: `float32` CUDA tensor `(M, K)`; `b`: `float32` CUDA tensor `(K, N)`. Any sizes $\ge 1$, not necessarily multiples of the tile sizes. Use the tensors' strides; do not assume contiguity.

## Output
- `float32` tensor `(M, N)`.

## Requirements
- A 2-D grid over output tiles: program `(pid_m, pid_n)` computes `C[pid_m*BM : +BM, pid_n*BN : +BN]`, looping over `K` in steps of `BK` and accumulating in a `float32` tile with `tl.dot`.
- Decorate the kernel with `@triton.autotune(configs=[...], key=["M", "N", "K"])` and at least three `triton.Config`s that vary the block sizes, `num_warps` and `num_stages`. The grid must be a function of the chosen config (`lambda meta: ...`).
- All block sizes at least 16 (a `tl.dot` requirement). Mask loads with `other=0.0` on both the M/N and K edges, and mask the store.
- Pass `input_precision="ieee"` to `tl.dot` so float32 inputs are not rounded to TF32 (the tests compare against exact float32 maths).

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def matmul(a, b):
    """a: (M, K), b: (K, N) float32 CUDA tensors. Return a @ b, (M, N), from an autotuned Triton kernel."""
    raise NotImplementedError


# @triton.autotune(configs=[triton.Config({...}, num_warps=..., num_stages=...), ...], key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BM": 32, "BN": 32, "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],          # re-tune whenever the problem shape changes
)
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak   # (BM, BK) tile of A
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn   # (BK, BN) tile of B
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k_ok = (k0 + rk) < K
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & k_ok[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_ok[:, None] & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")      # zero padding adds nothing to the dot
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a, b):
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    _matmul_kernel[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
'''

REFERENCE = '''import torch

def matmul(a, b):
    return (a.to(torch.float64) @ b.to(torch.float64)).to(torch.float32)
'''

HINTS = [
    'Build pointer tiles with broadcasting: `a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak` is a `(BM, BK)` grid of addresses. Advance both tiles by `BK` along K after each iteration.',
    'Mask the A tile with `(rm < M)` on rows and `(k0 + rk < K)` on columns, the B tile with the K mask on rows and `(rn < N)` on columns, and use `other=0.0`: padded zeros contribute nothing to `tl.dot`.',
    'With `@triton.autotune`, the block sizes are not passed at the call site; the grid must read them from the chosen config: `grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))`.',
]

_r = np.random.default_rng(31)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)),
    (np.array([[1.0, 0.0, -1.0]], dtype=np.float32), np.array([[2.0], [3.0], [4.0]], dtype=np.float32)),
    (_f(1, 1), _f(1, 1)),
    (_f(64, 64), _f(64, 64)),
    (_f(37, 50), _f(50, 23)),
    (_f(130, 7), _f(7, 70)),
    (_f(5, 300), _f(300, 129)),
    (np.eye(48, dtype=np.float32), _f(48, 17)),
]

EXPLANATION = r"""
## Why tiles
A naive matmul reads each element of $A$ and $B$ $N$ and $M$ times from DRAM. A program that owns a `BM x BN` output tile loads a `BM x BK` slice of $A$ and a `BK x BN` slice of $B$ per step and reuses each loaded value `BN` (or `BM`) times inside `tl.dot`, which runs on tensor cores. Arithmetic intensity grows with the tile size, about $\frac{2\,BM\,BN\,BK}{4(BM\,BK + BK\,BN)}$ flops per byte, which is how a matmul moves from memory-bound to compute-bound.

## What autotuning chooses
- **Block sizes**: larger tiles mean more reuse but more registers and shared memory per program, and fewer programs to fill the GPU. Small shapes (tests 1 to 3) want small tiles.
- **`num_warps`**: how many warps cooperate on a tile.
- **`num_stages`**: how many K-steps are prefetched in a software pipeline, which hides load latency at the cost of shared memory.

The first call with a new `(M, N, K)` benchmarks every config and caches the winner, so keep the key coarse enough that real workloads don't re-tune constantly.

## Masks
K-masking matters as much as M/N masking: with $K = 7$ and `BK = 32` (test 6), unmasked loads would read 25 columns of garbage into the dot product. Zero padding from `other=0.0` makes those terms vanish.

## Common bugs
- Forgetting `input_precision="ieee"`: TF32 has a 10-bit mantissa, so results differ from float32 by about $10^{-3}$ relative.
- Advancing pointers by `BK` instead of `BK * stride`.
- Passing `BM=...` at the call site of an autotuned kernel (it conflicts with the configs).

## Interview follow-ups
- Why does the order in which programs are scheduled (grouped launch order) change L2 hit rate and speed?
- When is split-K worth it, and what does it cost?
"""
