import numpy as np

META = dict(title='RMSNorm in half precision, the way LLMs do it', tags=['Transformers', 'Neural Networks'], diff='Easy',
            runtime='torch', tol=[2e-3, 1e-5], lessons=['air-fanout-v1:05.04', 'ieng:02.03'],
            related=['norm-gated-linear-transform', 'layer-norm', 'parameter-matched-swiglu-block'])

STATEMENT = r"""
RMSNorm rescales each vector by its root-mean-square, with no mean subtraction and no bias:

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_{k=1}^{d} x_k^2 + \varepsilon}} \odot g$$

LLMs run in `float16` or `bfloat16`, but the sum of squares is computed in `float32`: in `float16`, any value above about 256 squares past the largest finite number (65504) and the norm becomes `inf`.

Write `rmsnorm(x, weight, eps=1e-6, offset=False)` in PyTorch.

## Inputs
- `x`: tensor `(..., d)` of dtype `float16`, `float32` or `float64`. Normalise along the last axis.
- `weight`: `(d,)`, same dtype as `x`.
- `offset`: if `True`, the gain is $1 + g$ instead of $g$ (a convention used by some model families, so that a zero-initialised weight means identity).

## Rules
1. Compute in `float32` when `x` is `float16`; otherwise compute in `x`'s own dtype. Cast `weight` to the compute dtype too.
2. Apply the gain in the compute dtype, then cast the result back to `x.dtype`.
3. $\varepsilon$ goes **inside** the square root.

## Output
A tensor of the same shape and dtype as `x`. Use tensor ops only.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def rmsnorm(x, weight, eps=1e-6, offset=False):
    """Normalise x along its last axis by its RMS, times the gain. Same shape and dtype as x."""
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)   # TODO: overflows in float16, ignores offset
    return x / rms * weight
'''

SOLUTION = '''import torch

def rmsnorm(x, weight, eps=1e-6, offset=False):
    compute = torch.float32 if x.dtype == torch.float16 else x.dtype
    h = x.to(compute)                                    # upcast BEFORE squaring
    inv_rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    g = weight.to(compute)
    if offset:
        g = 1.0 + g                                      # zero-initialised weight = identity scale
    return (h * inv_rms * g).to(x.dtype)
'''

HINTS = [
    'Pick the compute dtype first: `torch.float32` for `float16` input, the input dtype otherwise. Cast `x` before you square anything.',
    '`torch.rsqrt(mean(h**2, dim=-1, keepdim=True) + eps)` gives $1/\\mathrm{rms}$ with the right shape to broadcast over the last axis.',
    'For `offset=True` the gain is `1 + weight`, computed in the compute dtype. Cast the final product back with `.to(x.dtype)`.',
]

_r = np.random.default_rng(3)
TESTS = [
    (np.array([[3.0, 4.0], [1.0, -1.0]]), np.array([1.0, 2.0])),
    (np.array([[3.0, 4.0]]), np.array([0.0, 0.5]), 1e-6, True),
    (np.array([[300.0, -400.0, 250.0, 100.0]], dtype=np.float16), np.ones(4, dtype=np.float16)),
    ((_r.normal(size=(2, 3, 16)) * 500).astype(np.float16), (1 + 0.1 * _r.normal(size=16)).astype(np.float16)),
    (_r.normal(size=(4, 8)).astype(np.float32), _r.normal(size=8).astype(np.float32), 1e-5, True),
    (np.zeros((2, 4)), np.ones(4), 1e-6),
    (np.full((1, 3), 1e-4), np.ones(3), 1e-6),
    (_r.normal(size=(3, 5, 7)), _r.normal(size=7), 1e-6, False),
]

EXPLANATION = r"""
## Why it works
Dividing by $\sqrt{\overline{x^2}}$ makes every output vector have unit RMS before the gain, whatever the scale of the residual stream. The square root is taken of a quantity that is at least $\varepsilon$, so all-zero vectors (test 6) give 0, not NaN.

## Precision
- Test 3 squares 400 in `float16`: $160000 > 65504$, so a naive kernel gets `inf` and outputs zeros. Upcasting first fixes it.
- `eps` matters for tiny vectors: in test 7 the mean square is $10^{-8}$, so $\varepsilon = 10^{-6}$ dominates and the output is about $0.1$, not 1. Models are trained with a specific $\varepsilon$; using a different one at inference shifts outputs.
- `bfloat16` has the range of `float32`, so it does not overflow here, but its 8-bit mantissa makes the reduction inaccurate; implementations upcast it as well.

## Complexity
$O(n)$ for $n$ elements: one reduction per row and one elementwise pass. It is memory-bound, so fused kernels read `x` once and write `y` once.

## Common bugs
- Putting $\varepsilon$ outside the square root (a different function, noticeably so for small vectors).
- Multiplying by the gain in half precision before the cast, which can round differently from the reference implementation.
- Normalising over the wrong axis for `(B, T, d)` inputs.

## Interview follow-ups
- Why did LLMs move from LayerNorm to RMSNorm? (Cheaper, one reduction instead of two, and mean-centering turned out not to matter for quality.)
- What does QK-norm do, and why does it stabilise training at large learning rates?
"""
