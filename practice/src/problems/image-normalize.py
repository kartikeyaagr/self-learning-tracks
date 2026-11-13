import numpy as np

META = dict(title='Normalise an image into a CHW tensor', tags=['Computer Vision', 'Data Processing'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:03.03'], related=['per-channel-mean-and-std', 'rgb-to-grayscale', 'normalize-columns'])

STATEMENT = r"""
Images come off disk as `uint8` in **height, width, channel** order. Most CNNs want floats in **channel, height, width** order, standardised with the per-channel mean and standard deviation of the training set (measured on the 0-1 scale). That preprocessing is:

$$\text{out}[c, i, j] = \frac{\text{img}[i, j, c] / 255 - \mu_c}{\sigma_c}$$

Write `normalize_image(img, mean, std)`.

## Input
- `img`: `uint8` array of shape `(H, W, C)`, or `(H, W)` for a grayscale image (treat it as `C = 1`).
- `mean`, `std`: sequences of `C` floats (lists or arrays), with every `std` entry positive.

## Output
A `float64` array of shape `(C, H, W)`. A grayscale input therefore returns shape `(1, H, W)`.

## Example
`img = [[[255, 0, 51]]]` (one pixel), `mean = [0.5, 0.5, 0.5]`, `std = [0.5, 0.5, 0.5]` gives `[[[1.0]], [[-1.0]], [[-0.6]]]`.
"""

STARTER = '''import numpy as np

def normalize_image(img, mean, std):
    """img: uint8 (H, W, C) or (H, W). Return float64 (C, H, W), standardised per channel."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def normalize_image(img, mean, std):
    x = np.asarray(img, dtype=np.float64) / 255.0
    if x.ndim == 2:                       # grayscale: add a channel axis
        x = x[:, :, None]
    x = (x - np.asarray(mean, dtype=float)) / np.asarray(std, dtype=float)   # broadcasts over the last axis
    return x.transpose(2, 0, 1)           # HWC -> CHW
'''

HINTS = [
    'Do three things in order: scale to [0, 1], standardise each channel, then reorder the axes.',
    'While the image is still `(H, W, C)`, a length-`C` mean broadcasts against the last axis for free. Normalise before you transpose.',
    '`x = img.astype(float) / 255`; add `x[:, :, None]` for 2-D input; `((x - mean) / std).transpose(2, 0, 1)`.',
]

_r = np.random.default_rng(12)
TESTS = [
    (np.array([[[255, 0, 51]]], dtype=np.uint8), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    (np.array([[0, 255], [128, 64]], dtype=np.uint8), [0.5], [0.25]),
    (_r.integers(0, 256, size=(3, 4, 3), dtype=np.uint8), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    (np.zeros((2, 2, 3), dtype=np.uint8), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
    (np.full((1, 3, 2), 255, dtype=np.uint8), [1.0, 0.0], [2.0, 4.0]),
    (_r.integers(0, 256, size=(5, 2, 4), dtype=np.uint8), [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]),
]

EXPLANATION = r"""
## Why this order of operations
Broadcasting lines shapes up from the right. In HWC layout the channel axis is last, so a `(C,)` mean and std broadcast without any reshaping. After the transpose to CHW you'd need `mean[:, None, None]` instead. Both are correct; doing the arithmetic first is just less fiddly.

## Why standardise at all
Zero-mean, unit-variance inputs keep early activations in a sensible range and make the network's first layer behave the same across datasets with different brightness. The statistics must be the **training set's**, applied unchanged at test time, and on the same 0-1 scale they were measured on.

## Complexity
$O(HWC)$. `transpose` returns a view, so the reorder is free; call `np.ascontiguousarray` if a downstream kernel needs contiguous memory.

## Common bugs
- Doing arithmetic in `uint8` (`img - 128` wraps around).
- Applying 0-1 statistics to 0-255 pixels, or forgetting the `/255` entirely.
- `reshape(C, H, W)` instead of `transpose`: the shape is right but the pixels are scrambled.

## Interview follow-ups
- Write the inverse to display a normalised tensor. (`(x * std[:, None, None] + mean[:, None, None]) * 255`, clip, cast.)
- Why can't the normalisation be folded away at inference time? (It can: it's affine, so it merges into the first convolution's weights and bias.)
"""
