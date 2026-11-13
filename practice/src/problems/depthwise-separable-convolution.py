import numpy as np

META = dict(title='Depthwise separable convolution and its savings', tags=['Convolutional Networks', 'Computer Vision'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:04.02'],
            related=['2d-convolution', 'squeeze-and-excitation-block', 'inception-module'])

STATEMENT = r"""
MobileNet-style networks factor a $k\times k$ convolution into two cheap steps:
1. **Depthwise**: each input channel is convolved with its own single $k\times k$ filter (no mixing across channels).
2. **Pointwise**: a $1\times1$ convolution mixes the channels.

Write `depthwise_separable(x, Wd, bd, Wp, bp, stride=1, padding=0)` that returns `(y, separable_params, standard_params)`.

## Shapes
- `x`: `(N, C, H, W)`.
- `Wd`: `(C, 1, k, k)` depthwise filters (PyTorch `groups=C` layout), `bd`: `(C,)`.
- `Wp`: `(F, C, 1, 1)` pointwise weights, `bp`: `(F,)`.

## Computation
- Depthwise: $z_{n,c,i,j} = \mathrm{bd}_c + \sum_{u,v} \mathrm{Wd}_{c,0,u,v}\, \tilde x_{n,c,\,is+u,\,js+v}$ with $\tilde x$ zero-padded by `padding`. Output size $\lfloor (H + 2p - k)/s \rfloor + 1$.
- Pointwise: $y_{n,f,i,j} = \mathrm{bp}_f + \sum_c \mathrm{Wp}_{f,c,0,0}\, z_{n,c,i,j}$.
- No activation in between (real networks insert BatchNorm and ReLU there; leave them out).

## Output
- `y`: float `(N, F, H_out, W_out)`.
- `separable_params`: number of weights and biases in the two layers, an `int`.
- `standard_params`: weights and biases of an ordinary conv with the same $C$, $F$ and $k$ (`F*C*k*k + F`), an `int`.
"""

STARTER = '''import numpy as np

def depthwise_separable(x, Wd, bd, Wp, bp, stride=1, padding=0):
    """Return (y (N, F, Ho, Wo), separable_params, standard_params)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def depthwise_separable(x, Wd, bd, Wp, bp, stride=1, padding=0):
    x, Wd, Wp = (np.asarray(a, dtype=float) for a in (x, Wd, Wp))
    N, C, H, W = x.shape
    k, s, p = Wd.shape[-1], stride, padding
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    z = np.zeros((N, C, Ho, Wo))
    for u in range(k):
        for v in range(k):
            # each channel only ever meets its own filter tap: a broadcast multiply, no channel sum
            z += xp[:, :, u: u + s * (Ho - 1) + 1: s, v: v + s * (Wo - 1) + 1: s] * Wd[None, :, 0, u, v, None, None]
    z += np.asarray(bd, dtype=float)[None, :, None, None]
    y = np.einsum('nchw,fc->nfhw', z, Wp[:, :, 0, 0]) + np.asarray(bp, dtype=float)[None, :, None, None]
    F = Wp.shape[0]
    return y, int(Wd.size + C + Wp.size + F), int(F * C * k * k + F)
'''

HINTS = [
    'The depthwise step is a convolution where channel $c$ of the output only reads channel $c$ of the input: like a normal conv, but with no sum over channels.',
    'Loop over the $k^2$ taps; for each, take the strided slice of the padded input `(N, C, Ho, Wo)` and multiply by `Wd[:, 0, u, v]` reshaped to `(1, C, 1, 1)`.',
    'The pointwise step is a matrix product over channels: `einsum("nchw,fc->nfhw", z, Wp[:, :, 0, 0])`. Parameters: `C*k*k + C` for depthwise and `F*C + F` for pointwise.',
]

_r = np.random.default_rng(43)
def _p(C, F, k):
    return _r.normal(size=(C, 1, k, k)), _r.normal(size=C), _r.normal(size=(F, C, 1, 1)), _r.normal(size=F)
TESTS = [
    (np.arange(8.0).reshape(1, 2, 2, 2), np.ones((2, 1, 2, 2)), np.zeros(2), np.array([[[[1.0]], [[-1.0]]]]), np.zeros(1)),
    (np.ones((1, 1, 3, 3)), np.ones((1, 1, 3, 3)), np.zeros(1), np.array([[[[2.0]]], [[[0.5]]]]), np.array([0.0, 1.0]), 1, 1),
    (_r.normal(size=(2, 3, 6, 6)), *_p(3, 5, 3)),
    (_r.normal(size=(2, 3, 6, 6)), *_p(3, 5, 3), 1, 1),
    (_r.normal(size=(1, 4, 9, 7)), *_p(4, 2, 3), 2, 1),
    (_r.normal(size=(1, 2, 5, 5)), *_p(2, 3, 5), 1, 2),
    (_r.normal(size=(2, 6, 8, 8)), *_p(6, 4, 1), 2),
]

EXPLANATION = r"""
## The factorisation
A standard conv does spatial filtering and channel mixing in one $F \times C \times k \times k$ tensor. Separating them costs $Ck^2 + CF$ weights instead of $FCk^2$, a ratio of
$$\frac{Ck^2 + CF}{FCk^2} = \frac{1}{F} + \frac{1}{k^2},$$
so with $k = 3$ and $F$ in the hundreds it is roughly $8$ to $9\times$ cheaper, in both parameters and multiply-adds. The price is expressiveness: the separable version can only represent kernels that are a sum of channel-wise filters followed by a linear mix.

## Implementation
Depthwise conv is a convolution with `groups = C`, which in NumPy is just a broadcast multiply per tap (no `einsum` needed). Pointwise conv is a per-pixel matrix product.

## Complexity
Depthwise $O(N C k^2 H_o W_o)$, pointwise $O(N C F H_o W_o)$. On GPUs the depthwise part is memory-bound (very little arithmetic per byte), so the FLOP savings don't fully translate into speed.

## Common bugs
- Summing over channels in the depthwise step (that's a normal conv with `F = C`).
- Indexing `Wd[c, 0]` as if it were `(1, C, k, k)`.
- Forgetting either bias in the parameter count.

## Interview follow-ups
- What does MobileNetV2's inverted residual add (expand with 1x1, depthwise, project with linear 1x1)?
- Why is the ratio of separable to standard cost independent of the image size?
"""
