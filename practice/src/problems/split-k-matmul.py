import numpy as np

META = dict(title='Split-K matmul for skinny outputs', tags=['GPU Programming', 'Linear Algebra', 'Inference'],
            diff='Hard', runtime='triton', tol=[1e-4, 1e-4], lessons=['ieng:13.03', 'ieng:12.07'],
            related=['grouped-program-id-matmul', 'autotuned-matrix-multiplication', 'gemv-matrix-vector-product'])

STATEMENT = r"""
When $M$ and $N$ are small but $K$ is large (a decode-time projection with a handful of tokens, say $16 \times 4096$ times $4096 \times 64$), an output-tiled matmul launches only a few programs and leaves most of the GPU idle, each program grinding through all of $K$. **Split-K** also cuts the reduction dimension into `split_k` slices, computes one partial product per slice in parallel, and sums the partials afterwards.

Write `split_k_matmul(a, b, split_k, BK)`.

## Inputs
- `a`: `float32` CUDA `(M, K)`; `b`: `float32` CUDA `(K, N)`; all sizes $\ge 1$. Use their strides.
- `split_k`: Python int $\ge 1$. `BK`: Python int, a power of two $\ge 16$ (the K step inside a program).

## How K is split
Every slice has the same length, rounded up to a multiple of `BK`:

$$\text{chunk} = \left\lceil \frac{K}{\text{split\_k} \cdot BK} \right\rceil \cdot BK, \qquad \text{slice } s = \bigl[\, s \cdot \text{chunk},\ \min((s+1)\cdot\text{chunk},\, K) \bigr)$$

Trailing slices can be empty (their partial is all zeros).

## Output
A tuple `(c, partial)`:
- `partial`: `float32` `(split_k, M, N)`; `partial[s] = a[:, slice_s] @ b[slice_s, :]`.
- `c`: `float32` `(M, N)` $= \sum_s$ `partial[s]` (that is, $a b$).

## Requirements
- A 3-D grid `(cdiv(M, 32), cdiv(N, 32), split_k)` with `BM = BN = 32`: program `(pid_m, pid_n, pid_k)` computes its output tile over slice `pid_k`, accumulating in float32 with `tl.dot(..., input_precision="ieee")`, and stores it to `partial[pid_k]`.
- Mask the K edge as well as M and N. Reduce the partials with a fixed-order sum (no atomics).

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def split_k_matmul(a, b, split_k, BK):
    """a (M, K), b (K, N) float32 CUDA. Return (c (M, N), partial (split_k, M, N))."""
    raise NotImplementedError


@triton.jit
def _split_k_kernel(a_ptr, b_ptr, part_ptr, M, N, K, chunk,
                    stride_am, stride_ak, stride_bk, stride_bn,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _split_k_kernel(a_ptr, b_ptr, part_ptr, M, N, K, chunk,
                    stride_am, stride_ak, stride_bk, stride_bn,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    k_start = pid_k * chunk                                  # chunk is a multiple of BK
    rk = k_start + tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(k_start, k_start + chunk, BK):          # only this program's slice of K
        k_ok = (k0 + tl.arange(0, BK)) < K                   # the last slice may run past K (or be empty)
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & k_ok[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_ok[:, None] & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    out = part_ptr + pid_k.to(tl.int64) * M * N + rm[:, None] * N + rn[None, :]
    tl.store(out, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def split_k_matmul(a, b, split_k, BK):
    M, K = a.shape
    N = b.shape[1]
    chunk = triton.cdiv(K, split_k * BK) * BK
    partial = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
    BM = BN = 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), split_k)
    _split_k_kernel[grid](a, b, partial, M, N, K, chunk,
                          a.stride(0), a.stride(1), b.stride(0), b.stride(1), BM=BM, BN=BN, BK=BK)
    return partial.sum(dim=0), partial                       # deterministic reduction over slices
'''

REFERENCE = '''import torch

def split_k_matmul(a, b, split_k, BK):
    M, K = a.shape
    N = b.shape[1]
    chunk = -(-K // (split_k * BK)) * BK
    a, b = a.double(), b.double()
    partial = torch.zeros((split_k, M, N), dtype=torch.float64)
    for s in range(split_k):
        lo, hi = s * chunk, min((s + 1) * chunk, K)
        if lo < hi:
            partial[s] = a[:, lo:hi] @ b[lo:hi, :]
    return partial.sum(dim=0).float(), partial.float()
'''

HINTS = [
    'Program `(pid_m, pid_n, pid_k)` is an ordinary tiled matmul whose K loop runs over `range(pid_k * chunk, pid_k * chunk + chunk, BK)` instead of `range(0, K, BK)`. Start the K offsets of both pointer tiles at `pid_k * chunk`.',
    'The K mask is `(k0 + tl.arange(0, BK)) < K`, with `other=0.0` on both loads. It handles the slice that runs past `K` and the slices that start past `K` (they load only zeros and store a zero tile).',
    'Store the tile at `part_ptr + pid_k * M * N + rm[:, None] * N + rn[None, :]` with an M/N mask, then return `partial.sum(dim=0), partial` from the host.',
]

_r = np.random.default_rng(53)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
TESTS = [
    (np.ones((2, 64), dtype=np.float32), np.arange(64, dtype=np.float32)[:, None] * np.ones((1, 3), dtype=np.float32), 2, 16),
    (np.array([[1.0, 2.0, 3.0]], dtype=np.float32), np.array([[1.0], [10.0], [100.0]], dtype=np.float32), 1, 16),
    (_f(4, 40), _f(40, 5), 4, 16),
    (_f(8, 1000), _f(1000, 8), 4, 32),
    (_f(16, 2048), _f(2048, 64), 8, 64),
    (_f(33, 300), _f(300, 40), 3, 32),
    (_f(1, 17), _f(17, 1), 5, 16),
    (_f(70, 129), _f(129, 33), 2, 16),
]

EXPLANATION = r"""
## Why split K
The number of programs in an output-tiled matmul is $\lceil M/BM\rceil \lceil N/BN\rceil$. For $M = 16$, $N = 64$ that's 2 programs on a GPU with 100+ SMs. Splitting K by 8 gives 16 programs, each doing an eighth of the work, so the latency drops by up to 8x until the GPU is full. The price: `split_k` partial tiles of $M \times N$ floats are written and read back, which is cheap exactly when $M N$ is small compared to $K(M+N)$.

## The split arithmetic
Rounding the chunk to a multiple of `BK` means each program's loop never straddles two slices. Test 1: $K = 64$, `split_k = 2`, `BK = 16` gives chunk 32, so `partial[0]` sums rows 0 to 31 of $b$ ($496$) and `partial[1]` rows 32 to 63 ($1520$). Test 3: $K = 40$, `split_k = 4`, `BK = 16` gives chunk 16, slices $[0,16), [16,32), [32,40)$ and an **empty** slice $[48, 64)$ whose partial must be all zeros. Test 7 ($K = 17$, `split_k = 5`) has slices $[0, 16)$ and $[16, 17)$ and three empty ones.

## Deterministic vs atomic reduction
Many implementations `tl.atomic_add` each partial tile into `c`. That saves the workspace and a pass, but float addition in an arbitrary order makes results differ in the last bits between runs, which hurts reproducibility and debugging. A fixed-order sum (here `partial.sum(0)`, or a tiny second kernel) is bitwise reproducible.

## Common bugs
- Starting the K pointers at 0 in every program (all slices compute the same thing).
- Masking only `k0 + rk < k_start + chunk` and forgetting `< K`.
- Writing partials with stride `N` but forgetting the `pid_k * M * N` slice offset, so slices overwrite each other.
- Choosing `split_k` so large that each slice is a single `BK` step: the partial traffic then dominates.

## Interview follow-ups
- Given $M$, $N$, $K$ and the SM count, how would you pick `split_k` automatically?
- Stream-K splits the total work (all tiles times all K steps) evenly across a fixed number of programs. What problem with split-K does that solve?
"""
