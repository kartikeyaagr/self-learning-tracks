import numpy as np

META = dict(title='Per-channel weight quantization for a linear layer', tags=['Quantization', 'Inference'], diff='Medium',
            runtime='torch', lessons=['ieng:11.02', 'ieng:11.04'],
            related=['int8-symmetric-quantize', 'group-wise-int4-quantization', 'block-wise-fp8-quantization'])

STATEMENT = r"""
One scale for a whole weight matrix lets a single large row wreck the precision of every other row. **Per-channel** quantization gives each output channel (each row of an `nn.Linear` weight) its own scale, and the matmul still runs on integers because a per-row scale can be applied **after** the product.

Write `quantize_per_channel(W, x, bits=8)` in PyTorch.

## Inputs
- `W`: float64 `(out, in)`, a linear layer's weight (`y = x W^T`). `x`: float64 `(N, in)` activations, kept in float.
- `bits`: int in $[2, 8]$. Let $q_{\max} = 2^{\text{bits} - 1} - 1$ (127 for 8 bits).

## Per-channel quantization
For each row $c$: $s_c = \max_j |W_{cj}| / q_{\max}$ (use $s_c = 1$ for an all-zero row) and
$$Q_{cj} = \mathrm{clamp}\bigl(\mathrm{round}(W_{cj} / s_c),\, -q_{\max},\, q_{\max}\bigr),$$
rounding half to even (`torch.round`).

## Quantized matmul
Compute $y_q = (x\, Q^\top) \odot s$, i.e. multiply by the **integer** matrix first, then scale column $c$ of the result by $s_c$. (Do the product in float64; the point is where the scale goes.)

## Error report
Also build the **per-tensor** baseline: one scale $s = \max_{c,j} |W_{cj}| / q_{\max}$ (or 1 if `W` is all zero), the same rounding and clamping, and $y_t = (x\, Q_t^\top)\, s$. Report the max absolute error of each against the float result $y = x W^\top$.

## Output
A tuple `(Q, scales, y_q, err_channel, err_tensor)`: `Q` int8 `(out, in)`, `scales` float64 `(out,)`, `y_q` float64 `(N, out)`, and two Python floats.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def quantize_per_channel(W, x, bits=8):
    """Return (Q int8 (out,in), scales (out,), y_q (N,out), err_channel float, err_tensor float)."""
    raise NotImplementedError
'''

SOLUTION = '''import torch

def _quant(W, scale, qmax):
    return torch.clamp(torch.round(W / scale), -qmax, qmax)

def quantize_per_channel(W, x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1)                                     # (out,) one range per output channel
    scales = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    Q = _quant(W, scales[:, None], qmax)
    y_q = (x @ Q.T) * scales                                       # integer GEMM, then one multiply per column

    t_amax = W.abs().max()
    t_scale = t_amax / qmax if t_amax > 0 else torch.ones((), dtype=W.dtype)
    y_t = (x @ _quant(W, t_scale, qmax).T) * t_scale

    y = x @ W.T
    return (Q.to(torch.int8), scales, y_q,
            (y_q - y).abs().max().item(), (y_t - y).abs().max().item())
'''

HINTS = [
    'Reduce `W.abs()` over the input axis (`dim=1`) to get one max per row, then guard zero rows with `torch.where` before dividing.',
    'Broadcast the scales as `(out, 1)` when quantizing. In the matmul, `x @ Q.T` has shape `(N, out)`, so the `(out,)` scale vector broadcasts over its columns.',
    'Cast to `int8` only after rounding and clamping. The per-tensor baseline is the same code with a single scalar scale.',
]

_r = np.random.default_rng(51)
_W = _r.normal(size=(6, 10)) * np.array([0.01, 0.1, 1.0, 5.0, 0.5, 0.05])[:, None]
TESTS = [
    (np.array([[1.0, -0.5], [0.02, 0.01]]), np.array([[1.0, 1.0]])),
    (np.array([[2.0, -1.0, 0.0], [0.0, 0.0, 0.0]]), np.array([[1.0, 2.0, 3.0], [0.5, 0.0, -1.0]]), 4),
    (_W, _r.normal(size=(5, 10))),
    (_W, _r.normal(size=(3, 10)), 4),
    (_r.normal(size=(8, 16)), _r.normal(size=(4, 16)), 2),
    (np.array([[0.5, -127.0, 63.5], [1.5, 2.5, -3.0]]), np.eye(3)),
    (np.zeros((2, 4)), _r.normal(size=(2, 4))),
    (_r.standard_t(3, size=(16, 32)), _r.normal(size=(6, 32)), 8),
]

EXPLANATION = r"""
## Why the scale can move after the matmul
$y_{nc} = \sum_j x_{nj} W_{cj} \approx \sum_j x_{nj}\, s_c Q_{cj} = s_c \sum_j x_{nj} Q_{cj}$. The scale depends only on the output channel $c$, so it factors out of the sum over $j$. That is why per-**output**-channel scales are free: the GEMM runs on integers and one multiply per output column dequantizes it. A per-**input**-channel scale would sit inside the sum and can't be pulled out (SmoothQuant moves it into the activations instead).

## Precision
Each weight is off by at most $s_c / 2$. With one global scale, the rows with small weights (test 3 mixes rows a few hundred times apart in size) round to a handful of levels or to zero; per-channel scales give every row the full $[-q_{\max}, q_{\max}]$ range. The error gap grows as bits shrink.

## Common bugs
- Taking the max over the wrong axis, which gives per-input-channel scales of shape `(in,)`.
- Casting to `int8` before clamping, which wraps large values around instead of saturating. Here every value is already in range by construction, but the same code with a smaller manually chosen scale would overflow.
- Using $2^{\text{bits}-1}$ as $q_{\max}$: 128 does not fit in int8, and it breaks the symmetric range.

## Interview follow-ups
- Why are activations usually quantized per token rather than per channel? (A per-token scale also factors out of the sum, on the row side; a per-channel activation scale would not.)
- What is the memory saving of int8 per-channel weights for a 7B model, including scales? (About half of fp16: the scales add one number per row, negligible.)
"""
