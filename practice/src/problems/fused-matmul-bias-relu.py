import numpy as np

META = dict(title='Linear layer with fused bias and ReLU epilogue', tags=['GPU Programming', 'Neural Networks'],
            diff='Medium', runtime='triton', tol=[1e-4, 1e-4], lessons=['ieng:13.06', 'ieng:13.03'],
            related=['autotuned-matrix-multiplication', 'silu', 'block-pointer-matmul'])

STATEMENT = r"""
`torch.relu(x @ w.T + b)` launches a matmul, then a bias add, then a ReLU, writing and re-reading the `(M, N)` output twice in between. Fusing the bias and activation into the matmul's **epilogue** (the code after the K loop, while the tile is still in registers) removes both round trips.

Write `linear_relu(x, w, b)`.

## Inputs
- `x`: `float32` CUDA tensor `(M, K)`, the activations.
- `w`: `float32` CUDA tensor `(N, K)`, in `nn.Linear` layout (one row per output feature). Do **not** call `w.t().contiguous()`; read it through its strides.
- `b`: `float32` CUDA tensor `(N,)`.
- All sizes $\ge 1$, not necessarily multiples of the tile sizes.

## Output
- `float32` tensor `(M, N)`: $\max(0,\ x w^\top + b)$.

## Requirements
- 2-D grid over `(BM, BN)` output tiles, K loop with `tl.dot(..., input_precision="ieee")` into a `float32` accumulator, block sizes at least 16.
- Load `w` directly as `(BK, BN)` tiles: element `(k, n)` of the tile is `w[n, k]`.
- Add the bias (broadcast over rows) and apply ReLU **before** the single store.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def linear_relu(x, w, b):
    """x: (M, K), w: (N, K) nn.Linear layout, b: (N,), float32 CUDA. Return relu(x @ w.T + b), (M, N)."""
    raise NotImplementedError


@triton.jit
def _linear_relu_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K,
                        stride_xm, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _linear_relu_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K,
                        stride_xm, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk     # (BM, BK)
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn     # (BK, BN) view of w.T
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k_ok = (k0 + rk) < K
        xt = tl.load(x_ptrs, mask=(rm[:, None] < M) & k_ok[None, :], other=0.0)
        wt = tl.load(w_ptrs, mask=k_ok[:, None] & (rn[None, :] < N), other=0.0)
        acc += tl.dot(xt, wt, input_precision="ieee")
        x_ptrs += BK * stride_xk
        w_ptrs += BK * stride_wk
    # epilogue: the tile is still in registers, so bias and ReLU cost no extra memory traffic
    bias = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
    acc = tl.maximum(acc + bias[None, :], 0.0)
    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def linear_relu(x, w, b):
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K and b.shape == (N,)
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    BM, BN, BK = 32, 32, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _linear_relu_kernel[grid](x, w, b, out, M, N, K,
                              x.stride(0), x.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
                              BM=BM, BN=BN, BK=BK)
    return out
'''

REFERENCE = '''import torch

def linear_relu(x, w, b):
    z = x.to(torch.float64) @ w.to(torch.float64).T + b.to(torch.float64)
    return torch.relu(z).to(torch.float32)
'''

HINTS = [
    'It is a standard tiled matmul of `x` with `w.T`. The trick is the pointer tile for `w`: `w_ptr + rk[:, None]*stride_wk + rn[None, :]*stride_wn` reads `w[n, k]` at tile position `(k, n)`, i.e. a `(BK, BN)` tile of `w.T`, with no copy.',
    'Advance `w_ptrs` by `BK * stride_wk` per K step, and mask both the K edge and the N edge with `other=0.0`.',
    'After the loop: `bias = tl.load(b_ptr + rn, mask=rn < N, other=0.0)`, then `acc = tl.maximum(acc + bias[None, :], 0.0)`, then one masked store.',
]

_r = np.random.default_rng(43)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
TESTS = [
    (np.array([[1.0, -1.0]], dtype=np.float32), np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
     np.array([0.5, -1.0, 1.0], dtype=np.float32)),
    (np.array([[1.0, 2.0], [-3.0, 0.5]], dtype=np.float32), np.eye(2, dtype=np.float32),
     np.array([0.0, 0.0], dtype=np.float32)),
    (_f(1, 1), _f(1, 1), _f(1)),
    (_f(64, 32), _f(48, 32), _f(48)),
    (_f(37, 70), _f(19, 70), _f(19)),
    (_f(8, 5), _f(100, 5), np.full(100, -100.0, dtype=np.float32)),
    (_f(50, 129), _f(65, 129), _f(65) * 5),
    (_f(3, 200), _f(33, 200), np.full(33, 100.0, dtype=np.float32)),
]

EXPLANATION = r"""
## Epilogue fusion
After the K loop, the program holds a finished `(BM, BN)` output tile in registers. Any elementwise operation on it is nearly free: one bias load of `BN` values and a `max` per element. The unfused version writes `M x N` floats, reads them for the bias add, writes them again, reads them for ReLU and writes them a third time: four extra passes over the output. For skinny layers where the matmul itself is cheap, those passes can cost more than the matmul.

This is exactly what cuBLASLt epilogues, CUTLASS epilogue visitors and `torch.compile` (which generates Triton kernels like this one) do automatically.

## Reading `w` transposed for free
`nn.Linear` stores `w` as `(out_features, in_features)`, and the layer computes `x @ w.T`. A transposed operand is just a different pair of strides, so the kernel builds a `(BK, BN)` pointer tile with `stride_wk` along the tile's rows. Calling `w.t().contiguous()` would copy the whole weight matrix on every call.

## Tests worth noting
Test 6 has a large negative bias, so the whole output is clamped to exactly 0. Test 8 has a large positive bias, so ReLU never triggers. Both catch an epilogue applied in the wrong order or skipped.

## Interview follow-ups
- What changes for a GELU or SiLU epilogue, and for fusing a residual add (which needs another input tile)?
- Why is the backward pass of a fused ReLU easy to implement from the output alone?
"""
