import numpy as np

META = dict(title='LayerNorm forward that saves its statistics', tags=['GPU Programming', 'Transformers'],
            diff='Medium', runtime='triton', tol=[1e-4, 1e-5], lessons=['ieng:13.05', 'ieng:02.03'],
            related=['layernorm-backward', 'rmsnorm-forward', 'single-pass-mean-and-variance', 'layer-norm'])

STATEMENT = r"""
LayerNorm normalises each row of a `(M, N)` activation tensor and applies a learned scale and shift:

$$\mu_i = \frac1N\sum_j x_{ij}, \qquad \sigma_i^2 = \frac1N\sum_j (x_{ij} - \mu_i)^2, \qquad r_i = \frac{1}{\sqrt{\sigma_i^2 + \varepsilon}}, \qquad y_{ij} = (x_{ij} - \mu_i)\, r_i\, w_j + b_j$$

A training kernel also **saves** $\mu$ and $r$ ("rstd") so that the backward pass doesn't recompute them.

Write `layer_norm_fwd(x, w, b, eps)`.

## Inputs
- `x`: contiguous `float32` CUDA tensor `(M, N)` with $1 \le N \le 8192$.
- `w`, `b`: `float32` `(N,)`. `eps`: Python float (e.g. `1e-5`).

## Output
A tuple `(y, mean, rstd)`:
- `y`: `float32` `(M, N)`.
- `mean`, `rstd`: `float32` `(M,)`, the $\mu_i$ and $r_i$ above (population variance, $\varepsilon$ inside the square root).

## Requirements
- One program per row, the whole row in one block: `BLOCK = triton.next_power_of_2(N)`. Read `x` once.
- Compute the variance from centred values, with padding lanes forced to 0 *after* centring.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def layer_norm_fwd(x, w, b, eps):
    """x: (M, N), w, b: (N,), float32 CUDA. Return (y (M, N), mean (M,), rstd (M,))."""
    raise NotImplementedError


@triton.jit
def _layer_norm_fwd_kernel(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rstd_ptr, N, stride_row, eps,
                           BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_kernel(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rstd_ptr, N, stride_row, eps,
                           BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    offs = row.to(tl.int64) * stride_row + cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)            # padding would otherwise add mean**2 to the variance
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + offs, xc * rstd * w + b, mask=mask)
    tl.store(mean_ptr + row, mean)                # saved for the backward pass
    tl.store(rstd_ptr + row, rstd)


def layer_norm_fwd(x, w, b, eps):
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)      # scale warps with the row length
    _layer_norm_fwd_kernel[(M,)](x, w, b, y, mean, rstd, N, x.stride(0), eps, BLOCK=BLOCK, num_warps=num_warps)
    return y, mean, rstd
'''

REFERENCE = '''import torch

def layer_norm_fwd(x, w, b, eps):
    x64 = x.to(torch.float64)
    mean = x64.mean(dim=1)
    var = ((x64 - mean[:, None]) ** 2).mean(dim=1)
    rstd = 1.0 / torch.sqrt(var + eps)
    y = (x64 - mean[:, None]) * rstd[:, None] * w.to(torch.float64) + b.to(torch.float64)
    return y.to(torch.float32), mean.to(torch.float32), rstd.to(torch.float32)
'''

HINTS = [
    'One program per row with `cols = tl.arange(0, BLOCK)` and `mask = cols < N`. Load the row once with `other=0.0`; then `mean = tl.sum(x) / N` is correct because padding adds 0.',
    'Centre with `xc = tl.where(mask, x - mean, 0.0)`. Without the `where`, every padding lane contributes `(0 - mean)**2` to the variance.',
    '`rstd = 1 / tl.sqrt(var + eps)`, `y = xc * rstd * w + b`. Store `mean` and `rstd` at `mean_ptr + row` and `rstd_ptr + row`.',
]

_r = np.random.default_rng(47)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
TESTS = [
    (np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), np.ones(4, dtype=np.float32),
     np.zeros(4, dtype=np.float32), 1e-5),
    (np.array([[2.0, 2.0, 2.0], [0.0, 3.0, 6.0]], dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32),
     np.array([0.5, 0.0, -0.5], dtype=np.float32), 1e-5),
    (np.array([[5.0], [-1.0]], dtype=np.float32), np.array([2.0], dtype=np.float32),
     np.array([1.0], dtype=np.float32), 1e-5),
    (_f(4, 768), _f(768), _f(768), 1e-5),
    (_f(3, 1000) * 5 + 100, _f(1000), _f(1000), 1e-6),
    (_f(16, 37), _f(37), _f(37), 1e-3),
    (_f(2, 4097), _f(4097), _f(4097), 1e-5),
]

EXPLANATION = r"""
## One read, one write
With the row in registers, the kernel reads `x`, `w` and `b` once and writes `y` once. The eager PyTorch decomposition (mean, subtract, square, mean, rsqrt, multiply, multiply, add) makes several full passes over the activations. Normalisation layers are memory-bound, so this fusion is where the speedup comes from: typically 2x to 4x over unfused eager code.

## Why save `mean` and `rstd`
The backward pass needs $\hat x = (x - \mu) r$. Storing two floats per row costs $8M$ bytes, tiny compared to $4MN$ for the activations, and saves recomputing both reductions. This is the same choice `torch.nn.functional.layer_norm` makes internally.

## Edge cases
- **Constant rows** (test 2, row 1): variance 0, so `rstd` $= 1/\sqrt{\varepsilon}$ and $y = b$. The `eps` keeps it finite.
- **$N = 1$** (test 3): every row is constant, so $y = b$ exactly.
- **Large mean** (test 5): centring before squaring keeps the variance accurate in float32.
- **Non-power-of-two widths** (tests 6, 7): padding is masked out of both reductions.

## Limits
Above roughly 8k columns the row no longer fits comfortably in registers, and you switch to a chunked loop (two or three passes, or the single-pass merge of chunk statistics).

## Interview follow-ups
- Derive the backward pass for `x`, `w` and `b`. Which reductions are per row and which are over the batch?
- RMSNorm drops the mean and the bias. What does that save, and why do LLMs prefer it?
"""
