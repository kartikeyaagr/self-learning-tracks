import numpy as np

META = dict(title='Cross-entropy over a large vocabulary, mean with ignore_index', tags=['GPU Programming', 'Language Models', 'Loss Functions'],
            diff='Medium', runtime='triton', tol=[1e-4, 1e-5], lessons=['ieng:13.05', 'ieng:02.04'],
            related=['online-softmax', 'row-wise-logsumexp', 'cross-entropy-logits'])

STATEMENT = r"""
The last layer of a language model produces a `(M, V)` matrix of logits, one row per token, with $V$ in the tens of thousands. The training loss is the mean token cross-entropy, skipping padding positions:

$$\ell_i = \log\sum_{j=0}^{V-1} e^{z_{ij}} \;-\; z_{i, t_i}, \qquad L = \frac{1}{|\mathcal V|}\sum_{i \in \mathcal V} \ell_i, \qquad \mathcal V = \{\, i : t_i \ne \texttt{ignore\_index} \,\}$$

Write `cross_entropy(logits, target, ignore_index)` with a Triton kernel that never materialises the softmax.

## Inputs
- `logits`: contiguous `float32` CUDA tensor `(M, V)`, $M \ge 1$, $V \ge 1$, finite (values can be large, e.g. $\pm 100$).
- `target`: `int64` CUDA tensor `(M,)`. Each entry is either in $[0, V)$ or equal to `ignore_index`.
- `ignore_index`: Python int (PyTorch's default is `-100`).

## Output
- A 0-d `float32` tensor $L$. If **every** target is ignored, return NaN (that is what `torch.nn.functional.cross_entropy` does: $0/0$).

## Requirements
- One program per row with a fixed `BLOCK` (at most 4096). Stream the row in chunks, keeping a running max $m$ and running sum $s$ of $e^{z - m}$ (the online log-sum-exp), so any $V$ works.
- The kernel writes a per-row loss (0 for ignored rows) and a per-row valid flag. The host divides `loss.sum()` by `valid.sum()`. Do not use atomics (see the explanation for why).
- Never index the row with an ignored target: clamp it to a safe column before loading.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from a float64 PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def cross_entropy(logits, target, ignore_index):
    """logits: (M, V) float32, target: (M,) int64, CUDA. Return the mean loss over non-ignored rows (0-d float32)."""
    raise NotImplementedError


@triton.jit
def _ce_kernel(logits_ptr, target_ptr, loss_ptr, valid_ptr, V, stride_row, ignore_index, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _ce_kernel(logits_ptr, target_ptr, loss_ptr, valid_ptr, V, stride_row, ignore_index, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    z_row = logits_ptr + row.to(tl.int64) * stride_row
    m = float("-inf")
    s = 0.0
    for start in range(0, V, BLOCK):                      # online log-sum-exp over the vocabulary
        cols = start + tl.arange(0, BLOCK)
        z = tl.load(z_row + cols, mask=cols < V, other=float("-inf"))
        m_new = tl.maximum(m, tl.max(z, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(z - m_new), axis=0)
        m = m_new
    lse = m + tl.log(s)
    t = tl.load(target_ptr + row)
    valid = t != ignore_index
    z_t = tl.load(z_row + tl.where(valid, t, 0))          # never dereference column ignore_index
    tl.store(loss_ptr + row, tl.where(valid, lse - z_t, 0.0))
    tl.store(valid_ptr + row, valid.to(tl.float32))


def cross_entropy(logits, target, ignore_index):
    logits = logits.contiguous()
    M, V = logits.shape
    loss = torch.empty(M, dtype=torch.float32, device=logits.device)
    valid = torch.empty(M, dtype=torch.float32, device=logits.device)
    BLOCK = min(max(triton.next_power_of_2(V), 16), 4096)
    _ce_kernel[(M,)](logits, target, loss, valid, V, logits.stride(0), ignore_index, BLOCK=BLOCK)
    return loss.sum() / valid.sum()                        # 0 / 0 = nan when every row is ignored
'''

REFERENCE = '''import torch

def cross_entropy(logits, target, ignore_index):
    z = logits.to(torch.float64)
    valid = target != ignore_index
    lse = torch.logsumexp(z, dim=1)
    z_t = z.gather(1, torch.where(valid, target, 0)[:, None])[:, 0]
    loss = torch.where(valid, lse - z_t, 0.0)
    return (loss.sum() / valid.sum()).to(torch.float32)
'''

HINTS = [
    'Per row, $\\ell = \\mathrm{lse}(z) - z_t$. Compute the log-sum-exp in one streaming pass: load each chunk with `other=-inf`, then `m_new = max(m, max(chunk))` and `s = s * exp(m - m_new) + sum(exp(chunk - m_new))`.',
    'After the loop `lse = m + log(s)`. Load the target, set `valid = t != ignore_index`, and read the target logit at `tl.where(valid, t, 0)` so an ignored row never reads column -100.',
    'Store `tl.where(valid, lse - z_t, 0.0)` and `valid` as a float per row, then return `loss.sum() / valid.sum()` on the host.',
]

_r = np.random.default_rng(23)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
_t = lambda m, v: _r.integers(0, v, size=m).astype(np.int64)
_big = _f(4, 5000) * 3
_tb = _t(4, 5000)
_tb[1] = -100
TESTS = [
    (np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32), np.array([0, 2], dtype=np.int64), -100),
    (np.array([[1.0, 3.0], [5.0, -1.0], [0.5, 0.5]], dtype=np.float32), np.array([1, -100, 0], dtype=np.int64), -100),
    (np.array([[4.0], [-7.0]], dtype=np.float32), np.array([0, 0], dtype=np.int64), -100),
    (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([-100, -100], dtype=np.int64), -100),
    (np.array([[100.0, -100.0, 50.0], [-80.0, 90.0, 89.0]], dtype=np.float32), np.array([2, 1], dtype=np.int64), -100),
    (_big, _tb, -100),
    (_f(8, 1000), _t(8, 1000), -100),
    (_f(3, 37), np.array([5, 0, 36], dtype=np.int64), 0),
]

EXPLANATION = r"""
## Never build the softmax
The eager formulation `log_softmax(z).gather(...)` writes a full `(M, V)` tensor of log-probabilities and reads it back. For $M = 8192$ tokens and $V = 128\,000$ that is 4 GB of float32 traffic you don't need. The fused kernel reads each logit exactly once and writes two floats per row.

## Online log-sum-exp
Keeping $(m, s)$ with $s = \sum e^{z - m}$ lets the max change between chunks: when it grows from $m$ to $m'$, every old term is multiplied by $e^{m - m'}$. Subtracting the max keeps every exponent $\le 0$, so test 5, with logits of $\pm 100$, doesn't overflow ($e^{100}$ is about $10^{43}$, beyond float32's $3.4 \times 10^{38}$).

## ignore_index and the mean
Padding rows contribute neither loss nor count. Test 2 checks that the divisor is 2, not 3; test 4 (everything ignored) must give NaN like PyTorch; test 8 shows that `ignore_index` can be any value, here the valid class 0, so the row whose target is 0 is skipped.

## Why no atomics
`tl.atomic_add` into a single scalar would work, but float addition is not associative and the order in which programs finish varies run to run, so the loss would change in its last bits between identical runs. Per-row outputs plus a host (or second-kernel) reduction are deterministic.

## Common bugs
- Reading `z[row, ignore_index]`: an out-of-bounds load with `-100`.
- Dividing by $M$ instead of the number of valid rows.
- Padding lanes loaded as 0 instead of `-inf`, which adds $e^{-m}$ terms for nonexistent classes.
- Sizing `BLOCK = next_power_of_2(V)` without a cap: a 128k block won't fit in registers.

## Interview follow-ups
- Write the backward kernel: $\partial L / \partial z_{ij} = (\mathrm{softmax}(z_i)_j - [j = t_i]) / |\mathcal V|$. Why can it overwrite the logits buffer in place, and how much memory does that save?
- How would you add label smoothing? (It needs $\sum_j z_{ij}$ as one more running scalar.)
"""
