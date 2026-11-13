import numpy as np

META = dict(title='Dilated convolution with same padding', tags=['Convolutional Networks'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:04.02'], related=['multichannel-2d-convolution', '2d-transposed-convolution'])

STATEMENT = r"""
A dilated (atrous) convolution spreads the kernel taps `d` pixels apart. A `3 x 3` kernel with dilation 2 sees a `5 x 5` area with the same 9 weights, so the receptive field grows without extra parameters or downsampling. Segmentation networks stack them to see context at full resolution.

$$y[o, i, j] = b_o + \sum_{c} \sum_{a=0}^{k-1} \sum_{e=0}^{k-1} w[o, c, a, e]\; \tilde x[c,\, i + a d,\, j + e d]$$

with $\tilde x$ the input zero-padded by $p$ on each spatial side, and stride 1.

Write `dilated_conv2d(x, w, dilation, padding=0, b=None)`.

## Input
- `x`: float array `(C_in, H, W)`; `w`: float array `(C_out, C_in, k, k)` (square kernel); `b`: `(C_out,)` or `None`.
- `dilation`: int `d >= 1`.
- `padding`: an int `p >= 0`, or the string `"same"`, meaning `p = d * (k - 1) // 2`. `"same"` is only used with odd `k`, so the output then has the input's size.

## Output
`float64` array `(C_out, H_out, W_out)` with $H_{out} = H + 2p - d(k - 1)$ (same for $W$). Assume both are at least 1.
"""

STARTER = '''import numpy as np

def dilated_conv2d(x, w, dilation, padding=0, b=None):
    """x: (C_in, H, W), w: (C_out, C_in, k, k). Stride-1 dilated cross-correlation. Return (C_out, H_out, W_out)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dilated_conv2d(x, w, dilation, padding=0, b=None):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    k, d = w.shape[-1], dilation
    p = d * (k - 1) // 2 if padding == 'same' else padding
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    Ho = xp.shape[1] - d * (k - 1)
    Wo = xp.shape[2] - d * (k - 1)
    y = np.zeros((w.shape[0], Ho, Wo))
    for a in range(k):                       # loop over the k*k taps, vectorise over pixels and channels
        for e in range(k):
            patch = xp[:, a * d:a * d + Ho, e * d:e * d + Wo]          # (C_in, Ho, Wo)
            y += np.einsum('oc,chw->ohw', w[:, :, a, e], patch)
    if b is not None:
        y += np.asarray(b, dtype=np.float64)[:, None, None]
    return y
'''

HINTS = [
    'Dilation only changes **where** each tap reads from: tap `(a, e)` reads the input shifted by `(a*d, e*d)`.',
    'Loop over the `k*k` taps instead of the pixels. For each tap, slice a `(C_in, H_out, W_out)` block of the padded input and mix channels with that tap\'s `(C_out, C_in)` weights.',
    'Resolve `"same"` to `p = d * (k - 1) // 2`, pad, set `H_out = H + 2p - d*(k-1)`, and accumulate `np.einsum("oc,chw->ohw", w[:, :, a, e], xp[:, a*d:a*d+H_out, e*d:e*d+W_out])`.',
]

_r = np.random.default_rng(22)
TESTS = [
    (np.arange(25, dtype=float).reshape(1, 5, 5), np.ones((1, 1, 2, 2)), 2),
    (np.arange(9, dtype=float).reshape(1, 3, 3), np.ones((1, 1, 3, 3)), 1, 'same'),
    (np.arange(16, dtype=float).reshape(1, 4, 4), np.array([[[[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]]]), 2, 'same'),
    (_r.normal(size=(2, 9, 8)), _r.normal(size=(3, 2, 3, 3)), 3, 'same', _r.normal(size=3)),
    (_r.normal(size=(3, 7, 7)), _r.normal(size=(2, 3, 3, 3)), 2, 0),
    (_r.normal(size=(1, 6, 6)), _r.normal(size=(1, 1, 2, 2)), 1, 1, np.array([0.25])),
    (_r.normal(size=(2, 5, 6)), _r.normal(size=(2, 2, 3, 3)), 2, 1),
]

EXPLANATION = r"""
## Receptive field
A $k \times k$ kernel with dilation $d$ spans $d(k-1) + 1$ pixels per side. Stacking 3x3 layers with dilations $1, 2, 4, \dots$ grows the receptive field exponentially with depth while every layer keeps the full resolution, which is the idea behind dilated segmentation backbones and WaveNet's 1-D version.

## Same padding
Output size is $H + 2p - d(k-1)$; setting $p = d(k-1)/2$ makes it $H$. That only works when $d(k-1)$ is even, which is guaranteed for odd $k$.

## Complexity
$O(C_{out} C_{in} k^2 H_{out} W_{out})$, the same as an undilated convolution with the same number of weights.

## Common bugs
- Using `k` instead of the effective size $d(k-1)+1$ when computing the output size or the same-padding.
- Implementing dilation by inserting zeros into the kernel and running an ordinary convolution: correct but $d^2$ times more work.
- The "gridding" artefact: stacking layers with the same dilation $d > 1$ means each output only ever sees a sparse lattice of inputs. Mixing dilation rates avoids it.

## Interview follow-ups
- Express a dilated convolution as an ordinary convolution on a space-to-batch rearranged input. (Split the image into $d^2$ subsampled grids, convolve each normally, interleave the results.)
- How does dilation differ from a strided convolution followed by upsampling? (No information is thrown away; the map stays full resolution.)
"""
