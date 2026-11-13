import numpy as np

META = dict(title='Batched multi-channel conv2d', tags=['Convolutional Networks'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:04.02'], related=['dilated-2d-convolution', '2d-transposed-convolution', 'max-pool-2d'])

STATEMENT = r"""
A convolution layer maps `C_in` input channels to `C_out` output channels. Each output channel has its own `C_in x kh x kw` filter, and every output pixel is a dot product between that filter and the input patch under it, plus a bias. Deep-learning "convolution" is really **cross-correlation**: the kernel is not flipped.

$$y[n, o, i, j] = b_o + \sum_{c=0}^{C_{in}-1} \sum_{a=0}^{k_h-1} \sum_{d=0}^{k_w-1} w[o, c, a, d]\; \tilde x[n, c,\, i s + a,\, j s + d]$$

where $\tilde x$ is the input zero-padded by `padding` pixels on each spatial side.

Write `conv2d(x, w, b=None, stride=1, padding=0)`.

## Input
- `x`: float array `(N, C_in, H, W)`.
- `w`: float array `(C_out, C_in, kh, kw)`. `b`: `(C_out,)` or `None` (no bias).
- `stride`, `padding`: ints, `stride >= 1`, `padding >= 0`. Assume the padded input is at least as large as the kernel.

## Output
`float64` array `(N, C_out, H_out, W_out)` with $H_{out} = \lfloor (H + 2p - k_h)/s \rfloor + 1$ and the same for $W$.
"""

STARTER = '''import numpy as np

def conv2d(x, w, b=None, stride=1, padding=0):
    """x: (N, C_in, H, W), w: (C_out, C_in, kh, kw), b: (C_out,) or None. Return (N, C_out, H_out, W_out)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def conv2d(x, w, b=None, stride=1, padding=0):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    p = padding
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    kh, kw = w.shape[2:]
    # Patches: (N, C_in, H_out, W_out, kh, kw), a strided view (no copy until einsum).
    patches = sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]
    y = np.einsum('ncijab,ocab->noij', patches, w)
    if b is not None:
        y = y + np.asarray(b, dtype=np.float64)[None, :, None, None]
    return y
'''

HINTS = [
    'Pad first. Then each output pixel needs the `C_in x kh x kw` patch starting at row `i*stride`, column `j*stride`.',
    'Build all patches at once with `sliding_window_view(xp, (kh, kw), axis=(2, 3))` and keep every `stride`-th window position.',
    'Contract channels and kernel offsets in one go: `np.einsum("ncijab,ocab->noij", patches, w)`, then add `b[None, :, None, None]`.',
]

_r = np.random.default_rng(21)
TESTS = [
    (np.arange(9, dtype=float).reshape(1, 1, 3, 3), np.ones((1, 1, 2, 2))),
    (np.arange(16, dtype=float).reshape(1, 1, 4, 4), np.array([[[[1.0, 0.0], [0.0, -1.0]]]]), np.array([0.5]), 2),
    (np.ones((1, 2, 2, 2)), np.ones((3, 2, 3, 3)), np.array([1.0, 2.0, 3.0]), 1, 1),
    (_r.normal(size=(2, 3, 6, 5)), _r.normal(size=(4, 3, 3, 3)), _r.normal(size=4), 1, 1),
    (_r.normal(size=(1, 2, 7, 7)), _r.normal(size=(2, 2, 3, 2)), None, 2, 0),
    (_r.normal(size=(3, 1, 5, 5)), _r.normal(size=(1, 1, 1, 1)), np.array([0.0])),
    (_r.normal(size=(1, 3, 8, 6)), _r.normal(size=(2, 3, 5, 3)), _r.normal(size=2), 3, 2),
]

EXPLANATION = r"""
## Convolution as a big matrix product
Gathering patches turns convolution into a contraction over three indices $(c, a, d)$. The classic "im2col" trick reshapes the patches to a matrix of shape $(N H_{out} W_{out},\; C_{in} k_h k_w)$ and the weights to $(C_{in} k_h k_w,\; C_{out})$, so the whole layer is one GEMM. `einsum` expresses the same contraction without the reshapes.

## Complexity
$O(N \cdot C_{out} \cdot H_{out} W_{out} \cdot C_{in} k_h k_w)$ multiply-adds. im2col's patch matrix costs $k_h k_w$ times the input's memory once materialised; the strided view itself is free.

## Common bugs
- Flipping the kernel (`w[..., ::-1, ::-1]`): that's mathematical convolution, not what layers compute.
- Applying the stride by computing every position and then subsampling the **output** incorrectly (e.g. `y[..., ::s]` on the wrong axes), or subsampling before padding.
- Broadcasting the bias over the wrong axis: it belongs to the output channel axis 1.
- Getting `w`'s axis order wrong: PyTorch stores `(C_out, C_in, kh, kw)`; TensorFlow stores `(kh, kw, C_in, C_out)`.

## Interview follow-ups
- Count the parameters and FLOPs of this layer for given shapes. How do depthwise-separable convolutions reduce them?
- Implement the backward pass for `w`. (It's a correlation of the input with the output gradient.)
"""
