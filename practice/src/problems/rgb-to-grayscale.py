import numpy as np

META = dict(title='RGB and RGBA to grayscale', tags=['Computer Vision'], diff='Easy', runtime='numpy',
            lessons=[], related=['image-normalize', 'per-channel-mean-and-std'])

STATEMENT = r"""
A grayscale image keeps one number per pixel: its **luma**, a weighted sum of the colour channels. The weights reflect how sensitive the eye is to each primary, so green counts most and blue least. This problem uses the ITU-R BT.601 weights:

$$Y = 0.299\,R + 0.587\,G + 0.114\,B$$

Write `to_grayscale(img)`.

## Input
- `img`: a channels-last array of shape `(..., H, W, C)`, so a single image `(H, W, C)` or a batch `(N, H, W, C)` both work.
- `C` is 3 (RGB) or 4 (RGBA). For RGBA the alpha channel is **ignored**.
- Any numeric dtype, including `uint8`.

## Output
A `float64` array of shape `(..., H, W)`: the channel axis is removed. Do not rescale or round: a `uint8` input gives luma on the same 0-255 scale.

## Example
A single pure-green pixel `[[[0, 255, 0]]]` becomes `[[149.685]]`.
"""

STARTER = '''import numpy as np

def to_grayscale(img):
    """img: (..., H, W, 3 or 4). Return float64 luma of shape (..., H, W)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def to_grayscale(img):
    img = np.asarray(img, dtype=np.float64)       # cast first: uint8 arithmetic would overflow
    weights = np.array([0.299, 0.587, 0.114])
    return img[..., :3] @ weights                 # drop alpha, contract the channel axis
'''

HINTS = [
    'Grayscale is a dot product between each pixel\'s colour vector and a fixed weight vector.',
    'Slice `img[..., :3]` to keep only R, G, B whatever the number of leading axes, then contract the last axis with the weights.',
    'Convert to `float64` before multiplying, then `img[..., :3] @ np.array([0.299, 0.587, 0.114])` gives shape `(..., H, W)`.',
]

_r = np.random.default_rng(11)
TESTS = [
    (np.array([[[0, 255, 0]]], dtype=np.uint8),),
    (np.array([[[255, 0, 0], [0, 0, 255]], [[255, 255, 255], [0, 0, 0]]], dtype=np.uint8),),
    (np.array([[[10, 20, 30, 0], [10, 20, 30, 255]]], dtype=np.uint8),),
    (np.array([[[0.5, 0.5, 0.5]]]),),
    (_r.integers(0, 256, size=(2, 3, 4, 3), dtype=np.uint8),),
    (_r.uniform(0, 1, size=(5, 4, 4)),),
    (np.full((2, 2, 3), 200, dtype=np.uint8),),
]

EXPLANATION = r"""
## Why a dot product
Each output pixel is $\mathbf w^\top \mathbf c$ with $\mathbf w = (0.299, 0.587, 0.114)$ and $\mathbf c$ the pixel's RGB vector. `@` against a 1-D vector contracts the last axis, so it handles one image or a batch without reshaping. The weights sum to 1, so a gray pixel $(v, v, v)$ maps to $v$ (test 7).

## Complexity
$O(HWC)$ time, one pass; the output is $1/C$ the size of the input.

## Common bugs
- Multiplying a `uint8` array by integers, or summing channels in `uint8`, wraps around at 256.
- Using a plain mean of the channels. It's a valid grayscale, but not luma: pure green and pure blue would look equally bright.
- Forgetting RGBA: `img @ weights` fails with a shape error on 4 channels, and slicing `img[:, :, :3]` breaks for batches.
- Weights for linear vs gamma-encoded light differ (BT.709 uses 0.2126, 0.7152, 0.0722 on linear values). Ask which convention a pipeline expects.

## Interview follow-ups
- How would you convert to grayscale but keep 3 channels so a pretrained RGB model still accepts it? (Repeat the luma along a new last axis.)
- Your images are BGR (OpenCV order). What changes? (Reverse the weights or the channel axis.)
"""
