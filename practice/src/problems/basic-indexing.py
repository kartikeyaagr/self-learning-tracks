import numpy as np

META = dict(title='Centre crop with a stride', tags=['Computer Vision', 'Data Processing'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:03.05'], related=['reshaping-arrays', 'zeros-and-ones', 'mutation-trap'])

STATEMENT = r"""
Write `center_crop(img, h, w, stride=1)` that cuts an `h x w` window from the middle of an image and then keeps every `stride`-th pixel in both directions. Use slicing only.

## Input
- `img`: array of shape `(H, W)` (grayscale) or `(H, W, C)` (channels last), any dtype.
- `h, w`: crop size, Python ints with `1 <= h <= H` and `1 <= w <= W`.
- `stride`: Python int `>= 1`.

## Where the window starts
The crop starts at row `top = (H - h) // 2` and column `left = (W - w) // 2`. When the leftover space is odd, the extra row (or column) is therefore dropped from the **bottom** (or right).

## Output
The rows `top, top + stride, ...` below `top + h`, crossed with the columns `left, left + stride, ...` below `left + w`. All channels are kept. The shape is `(ceil(h / stride), ceil(w / stride))` with `C` appended for colour images, and the dtype matches `img`.

## Example
`img = np.arange(25).reshape(5, 5)`, `h = w = 3`, `stride = 1` gives `[[6, 7, 8], [11, 12, 13], [16, 17, 18]]`.
"""

STARTER = '''import numpy as np

def center_crop(img, h, w, stride=1):
    """img: (H, W) or (H, W, C). Centred h x w window, keeping every stride-th row and column."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def center_crop(img, h, w, stride=1):
    img = np.asarray(img)
    top, left = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    # One slice per spatial axis; any trailing channel axis is left untouched.
    return img[top:top + h:stride, left:left + w:stride]
'''

HINTS = [
    'A slice `a[start:stop:step]` works on every axis at once: `img[r0:r1:s, c0:c1:s]`. Axes you do not mention (the channels) are kept whole.',
    'Compute `top = (H - h) // 2` and `left = (W - w) // 2` from `img.shape[0]` and `img.shape[1]`.',
    'The answer is `img[top:top + h:stride, left:left + w:stride]`. The stop is `top + h` (exclusive), and the step goes last.',
]

_r = np.random.default_rng(8)
TESTS = [
    (np.arange(25).reshape(5, 5), 3, 3),
    (np.arange(24).reshape(4, 6), 2, 3, 2),
    (np.arange(6).reshape(2, 3), 2, 3),
    (np.arange(30).reshape(5, 6), 1, 1),
    (np.arange(36).reshape(6, 6), 4, 5),
    (np.arange(48).reshape(4, 4, 3), 2, 2),
    (_r.integers(0, 256, size=(9, 11, 3)), 7, 6, 3),
    (_r.normal(size=(10, 7)), 9, 7, 4),
]

EXPLANATION = r"""
## Why it works
Basic slicing (`start:stop:step`) on each axis selects an evenly spaced, rectangular sub-grid, so NumPy can return a **view** that simply points into the original buffer with new offsets and strides. No pixels are copied, which is why cropping huge images is essentially free. Trailing axes that you do not index are kept in full, so the same expression handles grayscale and channels-last colour images.

The integer division in `(H - h) // 2` decides the tie rule: with an odd leftover, the extra row is left at the bottom.

## Complexity
$O(1)$ to build the view. Touching its values later costs $O(hw/\text{stride}^2)$.

## Common bugs
- Writing `img[top:h, left:w]`, treating the crop size as an end index.
- Using `round((H - h) / 2)`: Python rounds halves to even, so the window shifts depending on the size.
- Forgetting the result is a view. Writing into the crop also writes into `img`. Call `.copy()` if you need independence (see the Mutation Trap problem).

## Interview follow-ups
- How would you take a **random** crop instead, for data augmentation? (Draw `top` from `rng.integers(0, H - h + 1)`, and the same for `left`.)
- Why is `img[::2, ::2]` a crude way to downsample, and what does averaging each `2 x 2` block fix? (It aliases high frequencies; block averaging is a low-pass filter.)
"""
