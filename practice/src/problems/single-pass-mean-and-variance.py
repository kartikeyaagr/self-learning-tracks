import numpy as np

META = dict(title='Single-pass row mean and variance (Welford)', tags=['GPU Programming', 'Neural Networks'],
            diff='Medium', runtime='triton', tol=[1e-4, 1e-4],
            lessons=['ieng:13.03', 'ieng:13.04'],
            related=['layernorm-forward', 'sample-variance-and-standard-deviation', 'vector-sum-reduction'])

STATEMENT = r"""
Normalization layers need the mean and variance of every row. The textbook shortcut $\mathrm{Var} = \mathbb{E}[x^2] - \mathbb{E}[x]^2$ reads the row once but cancels catastrophically in float32 when the mean is large compared to the spread. Compute both statistics in **one pass** over the row, stably, by merging per-chunk statistics.

Write `row_mean_var(x)`.

## Inputs
- `x`: contiguous `float32` CUDA tensor of shape `(M, N)`, $M \ge 1$, $N \ge 1$. `N` can be larger than the block size.

## Output
- A tuple `(mean, var)` of `float32` CUDA tensors of shape `(M,)`: the row means and the **population** variances (divide by $N$).

## Algorithm
One program per row loops over the row in chunks of `BLOCK` columns, reading each element exactly once. For every chunk compute its count $n_b$, mean $\bar x_b$ and centred sum of squares $M_{2,b} = \sum (x - \bar x_b)^2$ (masked lanes excluded), then merge into the running $(n, \bar x, M_2)$:

$$\delta = \bar x_b - \bar x, \quad n' = n + n_b, \quad \bar x' = \bar x + \delta\,\frac{n_b}{n'}, \quad M_2' = M_2 + M_{2,b} + \delta^2\,\frac{n\,n_b}{n'}$$

At the end, `var = M2 / N`.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def row_mean_var(x):
    """x: (M, N) float32 CUDA tensor. Return (mean (M,), population var (M,)) in one pass per row."""
    raise NotImplementedError


@triton.jit
def _row_mean_var_kernel(x_ptr, mean_ptr, var_ptr, N, stride_row, BLOCK: tl.constexpr):
    # TODO: one program per row; loop over chunks and merge (count, mean, M2)
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _row_mean_var_kernel(x_ptr, mean_ptr, var_ptr, N, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    base = x_ptr + row.to(tl.int64) * stride_row
    cnt = 0.0
    mean = 0.0
    m2 = 0.0
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(base + cols, mask=mask, other=0.0)
        n_b = tl.sum(mask.to(tl.float32), axis=0)            # >= 1: every chunk starts inside the row
        mean_b = tl.sum(x, axis=0) / n_b
        d = tl.where(mask, x - mean_b, 0.0)                  # centre on the chunk mean before squaring
        m2_b = tl.sum(d * d, axis=0)
        # Chan et al. merge of (cnt, mean, m2) with (n_b, mean_b, m2_b)
        delta = mean_b - mean
        tot = cnt + n_b
        mean = mean + delta * (n_b / tot)
        m2 = m2 + m2_b + delta * delta * (cnt * n_b / tot)
        cnt = tot
    tl.store(mean_ptr + row, mean)
    tl.store(var_ptr + row, m2 / N)


def row_mean_var(x):
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    var = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = min(1024, triton.next_power_of_2(N))
    _row_mean_var_kernel[(M,)](x, mean, var, N, x.stride(0), BLOCK=BLOCK)
    return mean, var
'''

REFERENCE = '''import torch

def row_mean_var(x):
    x = x.to(torch.float64)
    mean = x.mean(dim=1)
    var = ((x - mean[:, None]) ** 2).mean(dim=1)
    return mean.to(torch.float32), var.to(torch.float32)
'''

HINTS = [
    'One program per row: `row = tl.program_id(0)`, row base pointer `x_ptr + row * stride_row`, then `for start in range(0, N, BLOCK)` over the columns. Scalars such as `cnt = 0.0` can be carried through the loop.',
    'Per chunk: `n_b = tl.sum(mask.to(tl.float32))`, `mean_b = tl.sum(x) / n_b`, and centre with `tl.where(mask, x - mean_b, 0.0)` before squaring, so padding lanes add nothing.',
    'Merge with `delta = mean_b - mean`, `tot = cnt + n_b`, `mean += delta * n_b / tot`, `m2 += m2_b + delta**2 * cnt * n_b / tot`. The first chunk has `cnt = 0`, so it simply becomes the running state.',
]

_r = np.random.default_rng(21)
TESTS = [
    (np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32),),
    (np.array([[0.0, 10.0], [-1.0, 1.0], [5.0, 7.0]], dtype=np.float32),),
    (np.array([[3.5]], dtype=np.float32),),
    ((1e4 + _r.normal(size=(3, 5000))).astype(np.float32),),
    (_r.normal(size=(4, 1025)).astype(np.float32),),
    (_r.normal(size=(6, 3000)).astype(np.float32) * 3 - 2,),
    (np.tile(np.array([1e3, -1e3], dtype=np.float32), (2, 1500)),),
    (_r.uniform(-1, 1, size=(17, 33)).astype(np.float32),),
]

EXPLANATION = r"""
## Why not $\mathbb{E}[x^2] - \mathbb{E}[x]^2$
In test 4 the values are about $10^4$ with variance 1. Then $\mathbb{E}[x^2] \approx 10^8$, and float32 has about 7 significant digits, so its rounding error alone is about 10, far more than the variance being measured. Subtracting two nearly equal large numbers leaves noise, sometimes even a negative variance. Centring each chunk on its own mean before squaring keeps all the numbers small.

## The merge formula
Two groups with counts $n_a, n_b$, means $\bar x_a, \bar x_b$ and centred sums of squares $M_{2,a}, M_{2,b}$ combine exactly into one group:
$$M_2 = M_{2,a} + M_{2,b} + \delta^2 \frac{n_a n_b}{n_a + n_b}, \qquad \delta = \bar x_b - \bar x_a$$
The extra term accounts for the two groups having different means. The same formula merges per-lane, per-warp or per-device statistics, which is how distributed BatchNorm (SyncBN) and Welford reductions in compilers work. Welford's original algorithm is the special case $n_b = 1$.

## Cost
One read of `x`, so the kernel is memory-bound at about $4N$ bytes per row. The extra arithmetic per chunk (a few block reductions and scalar updates) is hidden behind memory traffic.

## Common bugs
- Counting padding lanes: use the mask both for `n_b` and in the centred sum.
- Dividing by $N - 1$: population variance is what LayerNorm uses.
- Computing the row offset in int32: `row * stride_row` overflows past $2^{31}$ elements, hence `.to(tl.int64)`.

## Interview follow-ups
- How would you split one very long row across several programs and merge their partial statistics?
- Why does LayerNorm in fp16/bf16 training still compute statistics in fp32?
"""
