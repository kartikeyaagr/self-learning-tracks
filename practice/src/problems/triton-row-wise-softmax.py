import numpy as np

META = dict(title='Length-masked scaled softmax in Triton', tags=['GPU Programming', 'Attention'], diff='Medium',
            runtime='triton', tol=[1e-5, 1e-6], lessons=['ieng:13.05', 'ieng:14.01'],
            related=['fused-row-wise-softmax', 'online-softmax', 'masked-triton-gelu-kernel'])

STATEMENT = r"""
Attention over a padded batch needs a softmax where row $i$ only covers its first $\ell_i$ columns; the rest are padding and must get probability exactly 0. Passing the lengths to the kernel avoids materialising a `-inf` mask tensor in memory.

Write `masked_softmax(x, lengths, scale)`.

## Inputs
- `x`: contiguous `float32` CUDA tensor `(M, N)`, finite entries, $1 \le N \le 8192$.
- `lengths`: `int64` CUDA tensor `(M,)` with $0 \le \ell_i \le N$.
- `scale`: positive Python float (e.g. $1/\sqrt{d}$), applied to the logits before the softmax.

## Output
- `float32` tensor `(M, N)`:
$$y_{ij} = \begin{cases} \dfrac{e^{s x_{ij} - m_i}}{\sum_{k < \ell_i} e^{s x_{ik} - m_i}} & j < \ell_i \\[1ex] 0 & j \ge \ell_i \end{cases}, \qquad m_i = \max_{k < \ell_i} s\,x_{ik}$$
- A row with $\ell_i = 0$ is all zeros (not NaN).

## Requirements
- One program per row, `BLOCK = triton.next_power_of_2(N)`; load the row's length inside the kernel with `tl.load(lengths_ptr + row)`.
- Two masks: `cols < N` (memory safety) and `cols < length` (which entries count). Padding must not affect the max or the sum.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def masked_softmax(x, lengths, scale):
    """x: (M, N) float32 CUDA, lengths: (M,) int64 CUDA. Row i is a softmax over its first lengths[i] columns."""
    raise NotImplementedError


@triton.jit
def _masked_softmax_kernel(x_ptr, len_ptr, y_ptr, N, stride, scale, BLOCK: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _masked_softmax_kernel(x_ptr, len_ptr, y_ptr, N, stride, scale, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    length = tl.load(len_ptr + row)
    valid = cols < length                                  # length <= N, so this also implies cols < N
    base = row.to(tl.int64) * stride
    x = tl.load(x_ptr + base + cols, mask=valid, other=float("-inf")) * scale
    m = tl.max(x, axis=0)
    m = tl.where(length > 0, m, 0.0)                       # empty row: avoid -inf - (-inf)
    num = tl.where(valid, tl.exp(x - m), 0.0)
    den = tl.sum(num, axis=0)
    den = tl.where(den > 0, den, 1.0)                      # empty row: 0 / 1 = 0
    tl.store(y_ptr + base + cols, num / den, mask=cols < N)   # write zeros over the padding too


def masked_softmax(x, lengths, scale):
    assert x.is_contiguous() and x.dtype == torch.float32
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    _masked_softmax_kernel[(M,)](x, lengths, y, N, x.stride(0), scale, BLOCK=BLOCK,
                                 num_warps=4 if BLOCK <= 2048 else 8)
    return y
'''

REFERENCE = '''import torch

def masked_softmax(x, lengths, scale):
    M, N = x.shape
    valid = torch.arange(N)[None, :] < lengths[:, None]
    z = (x.to(torch.float64) * scale).masked_fill(~valid, float('-inf'))
    p = torch.softmax(z, dim=1)
    return torch.where(valid, p, torch.zeros_like(p)).to(torch.float32)   # empty rows: NaN -> 0
'''

HINTS = [
    'Load with `mask=cols < length, other=float("-inf")`: padding then never wins the max, and `exp(-inf - m) = 0` keeps it out of the sum.',
    'An empty row has max `-inf`, and `-inf - (-inf)` is NaN. Replace the max by 0 when `length == 0` and the denominator by 1 when it is 0; the numerators are all 0 anyway.',
    'Store with the memory mask `cols < N`, not the length mask: the padded outputs must be written as 0, because `torch.empty_like` leaves garbage there.',
]

_r = np.random.default_rng(29)
TESTS = [
    (np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32), np.array([4, 2]), 1.0),
    (np.array([[0.0, 0.0, 0.0], [5.0, 1.0, 9.0]], dtype=np.float32), np.array([0, 1]), 0.5),
    ((_r.normal(size=(6, 50)) * 3).astype(np.float32), np.array([50, 1, 0, 25, 49, 13]), 0.125),
    ((_r.normal(size=(3, 1000)) * 10).astype(np.float32), np.array([1000, 777, 3]), 1.0),
    (np.array([[1000.0, 999.0, -1000.0, 5000.0]], dtype=np.float32), np.array([3]), 1.0),
    ((_r.normal(size=(4, 4097))).astype(np.float32), np.array([4097, 4096, 2049, 0]), 0.0883883),
    (np.array([[2.0]], dtype=np.float32), np.array([1]), 3.0),
]

EXPLANATION = r"""
## Why it works
A softmax over a prefix of the row is an ordinary softmax over the row with the suffix set to $-\infty$. Doing that with a load mask means the padding is never read from memory at all, and no mask tensor exists. The scale is applied after loading, before the max, which matches $\mathrm{softmax}(s\,x)$ exactly; scaling after the max would be wrong for no extra speed.

## Edge cases
- $\ell = 0$ (tests 2, 3, 6): the guards turn $0/0$ into $0/1$. A reference built from a `-inf` mask produces NaN for that row; real attention code must special-case it the same way (or never create empty rows).
- $\ell = 1$: the single entry gets probability 1, whatever its value.
- Large values beyond the length (test 5 has 5000 in the padding) must not raise the max: they are never loaded.

## Performance
One read and one write per element; the padding is written but not read. For attention prefill with $T \le 8192$ this kernel replaces three PyTorch ops (scale, masked fill, softmax) and the `(M, N)` boolean mask.

## Common bugs
- Storing with the length mask, which leaves uninitialised memory in the padded part of `y`.
- Comparing `cols < N` only and padding with `other=0.0`: zeros join the softmax and steal probability mass.
- Loading `lengths` as a Python list or on the CPU: the kernel needs a device pointer.

## Interview follow-ups
- How would you also apply a causal mask so that row $i$ of a $(T, T)$ score matrix covers columns $\le i$? (A second comparison `cols <= row % T`.)
- For $N$ larger than a block, which parts of the online-softmax loop change to respect per-row lengths? (The loop can stop at $\ell_i$ instead of $N$, saving reads.)
"""
