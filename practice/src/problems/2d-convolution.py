import numpy as np

META = dict(title='2D convolution with stride, padding and dilation', tags=['Convolutional Networks', 'Computer Vision'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:04.02'],
            related=['pooling-layers', 'transposed-convolution', 'depthwise-separable-convolution'])

STATEMENT = r"""
Implement the forward pass of a convolution layer the way deep-learning frameworks define it (strictly, a cross-correlation: the kernel is **not** flipped):

$$y_{n,f,i,j} = b_f + \sum_{c}\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1} W_{f,c,u,v}\; \tilde x_{n,\,c,\; i s + u d,\; j s + v d}$$

where $\tilde x$ is the input zero-padded by $p$ on every side, $s$ is the stride and $d$ the dilation.

Write `conv2d(x, W, b, stride=1, padding=0, dilation=1)`.

## Shapes
- `x`: `(N, C, H, W_in)`; `W`: `(F, C, k_h, k_w)`; `b`: `(F,)`.
- Output `(N, F, H_out, W_out)` with
$$H_\text{out} = \left\lfloor \frac{H + 2p - d(k_h - 1) - 1}{s} \right\rfloor + 1$$
and likewise for the width. Every test gives $H_\text{out}, W_\text{out} \ge 1$.

## Requirements
Loops over the kernel offsets $(u, v)$ are fine; avoid loops over output pixels.
"""

STARTER = '''import numpy as np

def conv2d(x, W, b, stride=1, padding=0, dilation=1):
    """x: (N, C, H, W); W: (F, C, kh, kw); b: (F,). Return (N, F, H_out, W_out)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def conv2d(x, W, b, stride=1, padding=0, dilation=1):
    x, W = np.asarray(x, dtype=float), np.asarray(W, dtype=float)
    N, C, H, Wd = x.shape
    F, _, kh, kw = W.shape
    s, p, d = stride, padding, dilation
    Ho = (H + 2 * p - d * (kh - 1) - 1) // s + 1
    Wo = (Wd + 2 * p - d * (kw - 1) - 1) // s + 1
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    y = np.zeros((N, F, Ho, Wo))
    for u in range(kh):
        for v in range(kw):
            # input pixels touched by kernel tap (u, v) for every output position at once
            patch = xp[:, :, u * d: u * d + s * (Ho - 1) + 1: s, v * d: v * d + s * (Wo - 1) + 1: s]   # (N, C, Ho, Wo)
            y += np.einsum('nchw,fc->nfhw', patch, W[:, :, u, v])
    return y + np.asarray(b, dtype=float)[None, :, None, None]
'''

HINTS = [
    'Compute the output size first, and pad once with `np.pad(x, ((0,0),(0,0),(p,p),(p,p)))`.',
    'Loop over the $k_h k_w$ kernel taps instead of the output pixels. For tap $(u, v)$ the pixels it reads form a strided slice of the padded input starting at $(u d, v d)$ with step $s$.',
    'That slice is `xp[:, :, u*d : u*d + s*(Ho-1) + 1 : s, v*d : v*d + s*(Wo-1) + 1 : s]`, shape `(N, C, Ho, Wo)`. Contract its channel axis with `W[:, :, u, v]` (`einsum("nchw,fc->nfhw")`) and accumulate.',
]

_r = np.random.default_rng(31)
TESTS = [
    (np.arange(9.0).reshape(1, 1, 3, 3), np.array([[[[1.0, 0.0], [0.0, -1.0]]]]), np.array([0.5])),
    (np.ones((1, 1, 2, 2)), np.ones((1, 1, 3, 3)), np.zeros(1), 1, 1),
    (_r.normal(size=(2, 3, 6, 5)), _r.normal(size=(4, 3, 3, 3)), _r.normal(size=4)),
    (_r.normal(size=(2, 3, 7, 7)), _r.normal(size=(2, 3, 3, 3)), _r.normal(size=2), 2, 1),
    (_r.normal(size=(1, 2, 8, 9)), _r.normal(size=(3, 2, 3, 2)), _r.normal(size=3), 1, 2, 2),
    (_r.normal(size=(1, 1, 5, 5)), _r.normal(size=(2, 1, 1, 1)), np.zeros(2)),
    (_r.normal(size=(3, 2, 9, 6)), _r.normal(size=(2, 2, 4, 2)), _r.normal(size=2), 3, 2, 1),
    (_r.normal(size=(1, 3, 4, 4)), _r.normal(size=(5, 3, 4, 4)), _r.normal(size=5)),
]

EXPLANATION = r"""
## Loop over taps, not pixels
The naive code has six nested loops. Swapping the order so the outer loop runs over the $k_h k_w$ kernel taps turns the inner work into one strided slice and one small matrix product over channels. The same trick, taken further, is **im2col**: lay all patches out as columns of a big matrix and do a single GEMM, which is how CPUs and GPUs actually run convolutions.

## The output-size formula
A dilated kernel spans $d(k-1) + 1$ pixels. The first window starts at 0 of the padded input and the last must end inside it, so the number of starts is $\lfloor (H + 2p - d(k-1) - 1)/s \rfloor + 1$. Floor division silently drops pixels that don't fit a full stride (test 7).

## Complexity
$O(N F C k_h k_w H_\text{out} W_\text{out})$ multiply-adds; parameters $F C k_h k_w + F$.

## Common bugs
- Flipping the kernel (that's mathematical convolution; frameworks don't).
- Applying dilation to the stride or the padding instead of the tap offsets.
- An off-by-one in the slice end, giving `Ho + 1` rows for some strides.

## Interview follow-ups
- Derive $\partial L/\partial x$: it is a transposed convolution of the upstream gradient with the same weights.
- Receptive field of three stacked 3x3 stride-1 convs? (7x7, with fewer parameters than one 7x7 layer: $27C^2$ vs $49C^2$.)
"""
