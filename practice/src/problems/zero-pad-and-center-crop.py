import numpy as np

META = dict(title='Centre pad-or-crop to a fixed size', tags=['Computer Vision', 'Data Processing'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:03.05'], related=['basic-indexing', 'bilinear-resize'])

STATEMENT = r"""
A batch needs every image at the same size. A common recipe that never distorts the aspect ratio: on each axis, **crop** the centre if the image is too big and **pad** symmetrically with a constant if it is too small. The two axes are handled independently, so a wide, short image can be cropped horizontally and padded vertically at once.

Write `pad_or_crop(img, out_h, out_w, fill=0)`.

## Input
- `img`: array of shape `(H, W)` or `(H, W, C)`, any dtype.
- `out_h, out_w`: target size, Python ints `>= 1`.
- `fill`: the constant used for padding.

## Rules per axis (shown for rows; columns are the same with `W`, `out_w`)
- `H > out_h`: keep rows `top, ..., top + out_h - 1` with `top = (H - out_h) // 2`.
- `H < out_h`: add `before = (out_h - H) // 2` rows of `fill` above and `out_h - H - before` below, so an odd leftover puts the extra row at the **bottom**.
- `H == out_h`: leave the axis alone.

## Output
An array of shape `(out_h, out_w)` or `(out_h, out_w, C)` with the same dtype as `img`. Channels are never padded or cropped.

## Example
`img = [[1, 2, 3]]`, `out_h = 2`, `out_w = 2`: the columns are cropped from `left = (3 - 2) // 2 = 0`, keeping `[1, 2]`, and the single missing row goes at the bottom (`before = 0`, `after = 1`), giving `[[1, 2], [0, 0]]`.
"""

STARTER = '''import numpy as np

def pad_or_crop(img, out_h, out_w, fill=0):
    """img: (H, W) or (H, W, C). Centre-crop or pad each spatial axis to (out_h, out_w)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pad_or_crop(img, out_h, out_w, fill=0):
    img = np.asarray(img)
    for axis, out in ((0, out_h), (1, out_w)):
        n = img.shape[axis]
        if n > out:                                  # crop the centre
            start = (n - out) // 2
            img = np.take(img, np.arange(start, start + out), axis=axis)
        elif n < out:                                # pad, extra element after
            before = (out - n) // 2
            widths = [(0, 0)] * img.ndim
            widths[axis] = (before, out - n - before)
            img = np.pad(img, widths, constant_values=fill)
    return img
'''

HINTS = [
    'Treat rows and columns separately: each axis is either cropped, padded or left alone.',
    'Cropping is a slice starting at `(n - out) // 2`; padding is `np.pad` with a per-axis `(before, after)` pair and `(0, 0)` for every other axis, including channels.',
    'Loop over `(axis, out)` in `((0, out_h), (1, out_w))`; use `np.take(img, np.arange(start, start + out), axis=axis)` to crop and `np.pad(img, widths, constant_values=fill)` to pad.',
]

_r = np.random.default_rng(14)
TESTS = [
    (np.array([[1, 2, 3]]), 2, 2),
    (np.arange(16).reshape(4, 4), 2, 3),
    (np.arange(6).reshape(2, 3), 5, 6, -1),
    (np.arange(12, dtype=np.uint8).reshape(2, 2, 3), 3, 1, 9),
    (_r.normal(size=(5, 8, 2)), 8, 5),
    (np.arange(9).reshape(3, 3), 3, 3),
    (_r.integers(0, 256, size=(7, 2), dtype=np.uint8), 4, 7, 128),
]

EXPLANATION = r"""
## One rule, applied per axis
Padding and cropping are inverses in spirit: both centre the content and put the odd pixel on the bottom/right. Writing one loop over the two spatial axes keeps the conventions identical and makes mixed cases (crop one axis, pad the other) fall out for free. `np.pad` with `(0, 0)` on the channel axis leaves colour images intact.

## Complexity
$O(\text{out\_h} \cdot \text{out\_w} \cdot C)$ for the output, plus an intermediate of similar size.

## Common bugs
- Putting the extra padded row on top for odd leftovers while the crop drops the extra row at the bottom. Pick one convention and test both parities.
- Padding the channel axis by passing a single `(before, after)` pair to `np.pad`, which pads **every** axis.
- Using negative padding or slices with negative stops when the image is smaller than the target.

## Interview follow-ups
- Letterboxing for detectors: resize so the longer side fits, then pad. Which box coordinates must be shifted and by how much?
- Why might reflect padding be better than zeros for a segmentation model? (No artificial edge at the border.)
"""
