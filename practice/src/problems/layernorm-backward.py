import numpy as np

META = dict(title='LayerNorm backward with deterministic weight gradients', tags=['GPU Programming', 'Transformers', 'Training'],
            diff='Hard', runtime='triton', tol=[1e-4, 1e-5], lessons=['ieng:13.05', 'ieng:13.03'],
            related=['layernorm-forward', 'layer-normalization', 'rmsnorm-forward'])

STATEMENT = r"""
The forward pass $y_{ij} = \hat x_{ij} w_j + b_j$, with $\hat x_{ij} = (x_{ij} - \mu_i)\, r_i$, saved the per-row mean $\mu_i$ and inverse standard deviation $r_i$. Given the upstream gradient $\partial L/\partial y = dy$, the backward pass needs three things:

$$dx_{ij} = r_i\Bigl(g_{ij} - \tfrac1N\textstyle\sum_k g_{ik} - \hat x_{ij}\,\tfrac1N\sum_k g_{ik}\hat x_{ik}\Bigr), \quad g_{ij} = w_j\, dy_{ij}$$

$$dw_j = \sum_i dy_{ij}\,\hat x_{ij}, \qquad db_j = \sum_i dy_{ij}$$

`dx` is a per-row computation, but `dw` and `db` are sums **over rows**, so different programs contribute to the same output column. That is the interesting part.

Write `layer_norm_bwd(x, w, mean, rstd, dy)`.

## Inputs
- `x`, `dy`: contiguous `float32` CUDA tensors `(M, N)`, $M \ge 1$, $1 \le N \le 8192$.
- `w`: `float32` `(N,)`. `mean`, `rstd`: `float32` `(M,)`, exactly as saved by the forward pass (use them; do not recompute).

## Output
A tuple `(dx, dw, db)` of `float32` tensors with shapes `(M, N)`, `(N,)`, `(N,)`.

## Requirements
- Launch `G = min(M, 64)` programs. Program `pid` handles rows `pid, pid + G, pid + 2G, ...`, keeping `dw` and `db` accumulators for its rows in registers (one block of `BLOCK = next_power_of_2(N)` columns).
- Each program writes its partial sums to row `pid` of `(G, N)` buffers; the host sums those over the first axis. No atomics: the result must be identical run to run.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def layer_norm_bwd(x, w, mean, rstd, dy):
    """x, dy: (M, N), w: (N,), mean, rstd: (M,), float32 CUDA. Return (dx, dw, db)."""
    raise NotImplementedError


@triton.jit
def _ln_bwd_kernel(x_ptr, dy_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr, dw_part_ptr, db_part_ptr,
                   M, N, stride_row, G, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _ln_bwd_kernel(x_ptr, dy_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr, dw_part_ptr, db_part_ptr,
                   M, N, stride_row, G, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    dw_acc = tl.zeros((BLOCK,), dtype=tl.float32)          # this program's share of dw and db
    db_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for row in range(pid, M, G):                           # rows pid, pid + G, ...
        offs = row.to(tl.int64) * stride_row + cols
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        xhat = tl.where(mask, (x - mean) * rstd, 0.0)      # padding lanes must not enter the sums
        g = w * dy                                         # already 0 on padding (w, dy loaded as 0)
        c1 = tl.sum(g * xhat, axis=0) / N
        c2 = tl.sum(g, axis=0) / N
        tl.store(dx_ptr + offs, (g - (xhat * c1 + c2)) * rstd, mask=mask)
        dw_acc += dy * xhat
        db_acc += dy
    tl.store(dw_part_ptr + pid * N + cols, dw_acc, mask=mask)
    tl.store(db_part_ptr + pid * N + cols, db_acc, mask=mask)


def layer_norm_bwd(x, w, mean, rstd, dy):
    x, dy = x.contiguous(), dy.contiguous()
    M, N = x.shape
    G = min(M, 64)
    dx = torch.empty_like(x)
    dw_part = torch.empty((G, N), dtype=torch.float32, device=x.device)
    db_part = torch.empty((G, N), dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    _ln_bwd_kernel[(G,)](x, dy, w, mean, rstd, dx, dw_part, db_part, M, N, x.stride(0), G,
                         BLOCK=BLOCK, num_warps=min(max(BLOCK // 256, 1), 8))
    return dx, dw_part.sum(0), db_part.sum(0)              # fixed-order reduction: deterministic
'''

REFERENCE = '''import torch

def layer_norm_bwd(x, w, mean, rstd, dy):
    x, w, dy = x.double(), w.double(), dy.double()
    mean, rstd = mean.double()[:, None], rstd.double()[:, None]
    xhat = (x - mean) * rstd
    g = w * dy
    dx = rstd * (g - g.mean(dim=1, keepdim=True) - xhat * (g * xhat).mean(dim=1, keepdim=True))
    dw = (dy * xhat).sum(dim=0)
    db = dy.sum(dim=0)
    return dx.float(), dw.float(), db.float()
'''

HINTS = [
    'Per row: rebuild $\\hat x = (x - \\mu) r$ from the saved statistics (zeroed on padding lanes), set $g = w \\cdot dy$, and take two row sums: $c_1 = \\frac1N\\sum g\\hat x$ and $c_2 = \\frac1N\\sum g$. Then $dx = (g - (\\hat x c_1 + c_2))\\, r$.',
    '`dw` and `db` are column sums over all rows. Give each program a register accumulator of length `BLOCK`, add `dy * xhat` and `dy` for every row it owns, and loop `for row in range(pid, M, G)`.',
    'At the end, store the accumulators to `dw_part[pid, :]` and `db_part[pid, :]`, launch with grid `(G,)`, and return `dw_part.sum(0)`, `db_part.sum(0)` from the host.',
]


def _case(seed, M, N, shift=0.0, scale=1.0):
    r = np.random.default_rng(seed)
    x = (r.normal(size=(M, N)) * scale + shift).astype(np.float32)
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=1)
    rstd = 1.0 / np.sqrt(((x64 - mean[:, None]) ** 2).mean(axis=1) + 1e-5)
    w = r.normal(size=N).astype(np.float32)
    dy = r.normal(size=(M, N)).astype(np.float32)
    return x, w, mean.astype(np.float32), rstd.astype(np.float32), dy


TESTS = [
    (np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), np.ones(4, dtype=np.float32),
     np.array([2.5], dtype=np.float32), np.array([0.894423], dtype=np.float32),
     np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)),
    (np.array([[0.0, 2.0], [1.0, 1.0]], dtype=np.float32), np.array([2.0, -1.0], dtype=np.float32),
     np.array([1.0, 1.0], dtype=np.float32), np.array([0.999995, 316.22778], dtype=np.float32),
     np.array([[1.0, 1.0], [0.5, -0.5]], dtype=np.float32)),
    _case(1, 3, 1),
    _case(2, 5, 37),
    _case(3, 8, 768),
    _case(4, 100, 64),
    _case(5, 3, 1000, shift=50.0, scale=4.0),
    _case(6, 70, 129),
]

EXPLANATION = r"""
## Where the formula comes from
With $g = \partial L / \partial \hat x = w \odot dy$, both $\mu$ and $r$ depend on every element of the row, so each $\hat x_{ij}$ depends on all $x_{ik}$. Differentiating through the mean and the variance and simplifying gives the projection form above: $dx$ is $g$ with its mean removed and its component along $\hat x$ removed, scaled by $r$. That's why $\sum_j dx_{ij} = 0$ and $\sum_j dx_{ij}\hat x_{ij} = 0$ for every row, a cheap sanity check.

## The cross-row reduction
`dx` has no cross-row dependency, so one program per row would be natural. But every row adds into all $N$ entries of `dw` and `db`. Three options:
1. **Atomics** into `dw`: simple but nondeterministic (float addition order changes) and contended.
2. **Per-row partials** `(M, N)`: deterministic but writes as much memory as `dx` itself.
3. **Per-program partials** `(G, N)` with $G \ll M$ (this solution): each program strides over rows and keeps its sums in registers, so the extra traffic is $G \cdot N$ floats and the final $G$-way sum is tiny and fixed-order.

Test 6 ($M = 100 > G = 64$) and test 8 ($M = 70$) make some programs own two rows; test 3 ($N = 1$) has $\hat x = 0$ everywhere, so `dx` and `dw` must be exactly 0.

## Performance
The kernel reads `x` and `dy` once and writes `dx` once: about $12MN$ bytes, memory-bound like the forward pass. $G$ should be a few times the number of SMs so the GPU is full while the partial buffer stays small.

## Common bugs
- Recomputing $\mu$ and $r$ instead of using the saved ones (slower, and can differ in the last bits from what the forward used).
- Forgetting to zero $\hat x$ on padding lanes: $(0 - \mu) r$ leaks into $c_1$.
- Writing `dw` with a plain store from every program: last writer wins.
- Dividing by `BLOCK` instead of `N` in $c_1$ and $c_2$.

## Interview follow-ups
- What changes for RMSNorm's backward (no mean, no bias)?
- How would you fuse the residual add that usually precedes LayerNorm into this kernel, so the gradient also flows to the residual branch without an extra pass?
"""
