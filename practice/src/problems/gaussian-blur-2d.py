import numpy as np

META = dict(title='Separable Gaussian blur', tags=['Computer Vision', 'Convolutional Networks'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:04.02'], related=['sobel-edge-detection', 'multichannel-2d-convolution'])

STATEMENT = r"""
A 2-D Gaussian kernel is the outer product of two 1-D Gaussians, so blurring can be done as a 1-D pass down the rows followed by a 1-D pass along the columns. That's $2(2r+1)$ multiply-adds per pixel instead of $(2r+1)^2$.

Write `gaussian_blur(img, sigma, radius=None)`.

## The 1-D kernel
- If `radius` is `None`, use `r = ceil(3 * sigma)`; otherwise `r = radius`.
- Taps $k = -r, \dots, r$ with weights $g_k = \exp\!\left(-\frac{k^2}{2\sigma^2}\right)$, then divide by their sum so the kernel sums to 1.

## Blurring
1. Filter along axis 0 (rows): `out[i] = sum_k g_k * img[i + k]`.
2. Filter that result along axis 1 (columns) the same way.

Out-of-range indices take the value of the nearest border pixel (edge replication, as in `np.pad(..., mode="edge")`). The kernel is symmetric, so convolution and correlation agree.

## Input / output
- `img`: array `(H, W)` or `(H, W, C)`; channels are blurred independently and never mixed.
- `sigma`: positive float. `radius`: `None` or an int `>= 0` (`0` returns the image unchanged, as floats).
- Return a `float64` array with the same shape as `img`.
"""

STARTER = '''import numpy as np

def gaussian_blur(img, sigma, radius=None):
    """img: (H, W) or (H, W, C). Separable Gaussian blur with edge padding; float64, same shape."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

def _blur_axis(x, g, axis):
    r = len(g) // 2
    pad = [(0, 0)] * x.ndim
    pad[axis] = (r, r)
    P = np.pad(x, pad, mode='edge')
    n = x.shape[axis]
    # Sum of shifted copies: tap k reads P[k : k + n] along `axis`.
    return sum(w * np.take(P, np.arange(k, k + n), axis=axis) for k, w in enumerate(g))

def gaussian_blur(img, sigma, radius=None):
    x = np.asarray(img, dtype=np.float64)
    r = math.ceil(3 * sigma) if radius is None else radius
    k = np.arange(-r, r + 1)
    g = np.exp(-k**2 / (2 * sigma**2))
    g /= g.sum()
    return _blur_axis(_blur_axis(x, g, 0), g, 1)
'''

HINTS = [
    'Build the 1-D kernel first, normalise it, and write one helper that filters along a single axis.',
    'For one axis: pad by `r` with `mode="edge"` on that axis only, then add up `2r + 1` shifted slices of the padded array, each times its weight.',
    'Call the helper with `axis=0` and then with `axis=1`; a channel axis just rides along because you never pad or shift it.',
]

_r = np.random.default_rng(17)
TESTS = [
    (np.array([[0.0, 0.0, 9.0, 0.0, 0.0]]), 1.0, 1),
    (np.full((3, 3), 2.0), 0.8),
    (np.eye(5), 1.0),
    (np.arange(12, dtype=np.uint8).reshape(2, 2, 3), 0.5, 2),
    (_r.normal(size=(7, 6)), 1.3),
    (_r.uniform(0, 255, size=(5, 8, 3)), 2.0, 3),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), 1.0, 0),
    (np.array([[4.0]]), 2.0),
]

EXPLANATION = r"""
## Why separability holds
$\exp\!\left(-\frac{a^2 + b^2}{2\sigma^2}\right) = \exp\!\left(-\frac{a^2}{2\sigma^2}\right)\exp\!\left(-\frac{b^2}{2\sigma^2}\right)$, so the 2-D kernel is $g g^\top$ and its correlation with the image equals filtering with $g$ along one axis, then the other. Edge replication is applied per axis, which matches 2-D edge padding exactly because replicate padding of a padded array is still replicate padding.

## Why normalise and why $3\sigma$
Normalising keeps a constant image constant (test 2). A Gaussian has about 99.7% of its mass within $3\sigma$, so truncating there changes little; normalisation redistributes the clipped tail.

## Complexity
$O(HW(2r+1))$ for the two passes, versus $O(HW(2r+1)^2)$ for a direct 2-D kernel.

## Common bugs
- Forgetting to normalise after truncation: the image dims or brightens slightly.
- Padding with zeros: borders darken.
- Padding every axis (including channels), which leaks one channel into another.

## Interview follow-ups
- Why do you blur before downsampling? (To remove frequencies the smaller grid can't represent and avoid aliasing.)
- Which other kernels are separable? (Any rank-1 kernel: box filters, Sobel. Check with the SVD of the kernel.)
"""
