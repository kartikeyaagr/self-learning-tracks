import numpy as np

META = dict(title='Matmul with grouped program ordering for L2 reuse', tags=['GPU Programming', 'Linear Algebra'],
            diff='Hard', runtime='triton', tol=[1e-4, 1e-4], lessons=['ieng:13.03', 'ieng:12.06'],
            related=['autotuned-matrix-multiplication', 'split-k-matmul', 'block-pointer-matmul'])

STATEMENT = r"""
A matmul launched on a 1-D grid has to turn each program id into an output tile `(pid_m, pid_n)`. Row-major order (`pid_m = pid // num_pid_n`) runs a whole row of tiles before moving on, so every program in flight needs a different column panel of $B$ and the L2 cache thrashes. **Grouped ordering** walks `GROUP_M` tile-rows at a time, column by column, so programs that run together share both their $A$ rows and their $B$ columns.

With `num_pid_m = cdiv(M, BM)` and `num_pid_n = cdiv(N, BN)`, the mapping is:

```
group_size   = GROUP_M * num_pid_n                      # programs per group
group_id     = pid // group_size
first_pid_m  = group_id * GROUP_M
group_rows   = min(num_pid_m - first_pid_m, GROUP_M)    # the last group may be shorter
pid_m        = first_pid_m + (pid % group_size) % group_rows
pid_n        = (pid % group_size) // group_rows
```

Write `grouped_matmul(a, b, BM, BN, BK, GROUP_M)`.

## Inputs
- `a`: `float32` CUDA `(M, K)`, `b`: `float32` CUDA `(K, N)`, any sizes $\ge 1$. Use their strides.
- `BM`, `BN`, `BK`: Python ints, powers of two $\ge 16$. `GROUP_M`: Python int $\ge 1$ (it may exceed `num_pid_m`).

## Output
A tuple `(order, c)`:
- `order`: `int32` `(num_pid_m, num_pid_n)`; `order[pid_m, pid_n]` is the program id that computed that tile. Each program writes its own `pid` there, which makes the schedule visible to the tests.
- `c`: `float32` `(M, N)` $= a b$, accumulated in float32 with `tl.dot(..., input_precision="ieee")`.

## Requirements
- A 1-D grid of `num_pid_m * num_pid_n` programs using the mapping above. Mask every load (`other=0.0`) and the store of `c`.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a PyTorch reference (float64 matmul, Python mapping).
"""

STARTER = '''import torch
import triton
import triton.language as tl

def grouped_matmul(a, b, BM, BN, BK, GROUP_M):
    """a (M, K), b (K, N) float32 CUDA. Return (order, a @ b) where order[pid_m, pid_n] = pid that computed the tile."""
    raise NotImplementedError


@triton.jit
def _grouped_matmul_kernel(a_ptr, b_ptr, c_ptr, order_ptr, M, N, K,
                           stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    # TODO: pid -> (pid_m, pid_n) with grouped ordering, then the usual tiled matmul
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_matmul_kernel(a_ptr, b_ptr, c_ptr, order_ptr, M, N, K,
                           stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_M
    group_rows = tl.minimum(num_pid_m - first_pid_m, GROUP_M)    # last group can be short
    pid_m = first_pid_m + (pid % group_size) % group_rows          # go down the group's rows first...
    pid_n = (pid % group_size) // group_rows                       # ...then step to the next column
    tl.store(order_ptr + pid_m * num_pid_n + pid_n, pid)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k_ok = (k0 + rk) < K
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & k_ok[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_ok[:, None] & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def grouped_matmul(a, b, BM, BN, BK, GROUP_M):
    M, K = a.shape
    N = b.shape[1]
    num_pid_m, num_pid_n = triton.cdiv(M, BM), triton.cdiv(N, BN)
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    order = torch.empty((num_pid_m, num_pid_n), dtype=torch.int32, device=a.device)
    _grouped_matmul_kernel[(num_pid_m * num_pid_n,)](
        a, b, c, order, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, GROUP_M=GROUP_M)
    return order, c
'''

REFERENCE = '''import torch

def grouped_matmul(a, b, BM, BN, BK, GROUP_M):
    M, N = a.shape[0], b.shape[1]
    num_pid_m, num_pid_n = -(-M // BM), -(-N // BN)
    order = torch.empty((num_pid_m, num_pid_n), dtype=torch.int32)
    group_size = GROUP_M * num_pid_n
    for pid in range(num_pid_m * num_pid_n):
        first = (pid // group_size) * GROUP_M
        rows = min(num_pid_m - first, GROUP_M)
        order[first + (pid % group_size) % rows, (pid % group_size) // rows] = pid
    return order, (a.double() @ b.double()).float()
'''

HINTS = [
    'Count tiles with `tl.cdiv(M, BM)` and `tl.cdiv(N, BN)` inside the kernel. One group is `GROUP_M` tile-rows times all `num_pid_n` columns, so `group_id = pid // (GROUP_M * num_pid_n)`.',
    'The last group can have fewer than `GROUP_M` rows: `group_rows = tl.minimum(num_pid_m - first_pid_m, GROUP_M)`. Inside a group, consecutive pids go down the rows (`% group_rows`) before moving right (`// group_rows`).',
    'Store `pid` at `order_ptr + pid_m * num_pid_n + pid_n`, then run the ordinary masked tiled matmul for tile `(pid_m, pid_n)`. Launch `(num_pid_m * num_pid_n,)` programs.',
]

_r = np.random.default_rng(37)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
TESTS = [
    (np.ones((64, 1), dtype=np.float32), np.arange(48, dtype=np.float32)[None, :], 16, 16, 16, 2),
    (np.ones((48, 1), dtype=np.float32), np.ones((1, 32), dtype=np.float32), 16, 16, 16, 1),
    (_f(80, 33), _f(33, 50), 16, 16, 16, 3),
    (_f(20, 40), _f(40, 70), 16, 32, 16, 8),
    (_f(1, 1), _f(1, 1), 16, 16, 16, 4),
    (_f(130, 64), _f(64, 129), 32, 32, 32, 4),
    (_f(200, 17), _f(17, 40), 32, 16, 16, 2),
    (np.eye(96, dtype=np.float32), _f(96, 96), 32, 32, 32, 2),
]

EXPLANATION = r"""
## Why the order matters
At any moment the GPU runs a few hundred programs. In row-major order those programs cover, say, one full row of 32 output tiles: they share one panel of $A$ but need 32 different panels of $B$. In grouped order with `GROUP_M = 8`, the same number of programs covers an $8 \times 4$ block of tiles: 8 panels of $A$ and 4 of $B$, so far fewer distinct bytes must stay in L2 at once. On large matmuls that's typically a 5 to 15 percent speedup with no change to the arithmetic.

## Reading the order matrix
Test 1 has a $4 \times 3$ tile grid with `GROUP_M = 2`:

```
 0  2  4
 1  3  5
 6  8 10
 7  9 11
```

pids 0 to 5 sweep the first two tile-rows column by column. Test 2 (`GROUP_M = 1`) collapses to row-major order. Test 3 has 5 tile-rows with `GROUP_M = 3`, so the last group has only 2 rows and needs the `group_rows` fix; test 4 has `GROUP_M` larger than the number of tile-rows, so everything is one column-major group.

## Complexity
The arithmetic is the usual $2MNK$ FLOPs. The mapping costs a handful of integer operations per program, which is nothing next to the K loop.

## Common bugs
- Using `GROUP_M` instead of `group_rows` in the modulo and division: wrong (or duplicated) tiles in the last group.
- Swapping `%` and `//`, which gives a valid but different (row-first) order.
- Computing `num_pid_n` with floor division: the partial last column of tiles is never computed.

## Interview follow-ups
- Why is the swizzle applied along M and not along N, and when would you flip it?
- How does a persistent kernel (a fixed number of programs, each looping over many tiles) interact with this ordering?
"""
