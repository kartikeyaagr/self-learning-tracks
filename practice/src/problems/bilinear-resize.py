import numpy as np

META = dict(title='Bilinear resize with half-pixel centres', tags=['Computer Vision'], diff='Medium', runtime='numpy',
            lessons=[], related=['zero-pad-and-center-crop', 'roi-align', 'fpn-top-down-fusion'])

STATEMENT = r"""
Resizing an image means asking, for every output pixel, "where does this land in the input?" and blending the four input pixels around that point. The result depends on the coordinate convention, so this problem pins it down: pixels are unit squares and a pixel's value lives at its **centre** (the convention of most deep-learning frameworks when corners are not aligned).

Write `resize_bilinear(img, out_h, out_w)`.

## Input
- `img`: float or integer array of shape `(H, W)` or `(H, W, C)`.
- `out_h, out_w`: Python ints `>= 1`.

## Source coordinate (rows; columns are identical with `W`, `out_w`)
For output row `y`, with `scale = H / out_h` computed as a float:

1. `s = (y + 0.5) * scale - 0.5`, then clamp below: `s = max(s, 0.0)`.
2. `y0 = min(floor(s), H - 1)`, `y1 = min(y0 + 1, H - 1)`, and the weight `t = s - y0`.

The value along that axis is `(1 - t) * img[y0] + t * img[y1]`. Doing this on both axes gives the usual four-pixel blend
$$(1-t)(1-u)\,I_{y_0 x_0} + (1-t)\,u\,I_{y_0 x_1} + t\,(1-u)\,I_{y_1 x_0} + t\,u\,I_{y_1 x_1}$$
where `u` is the column weight. Near the bottom/right edge `y1 == y0`, so the edge pixel is simply repeated.

## Output
A `float64` array of shape `(out_h, out_w)` or `(out_h, out_w, C)`. Channels are interpolated independently.
"""

STARTER = '''import numpy as np

def resize_bilinear(img, out_h, out_w):
    """img: (H, W) or (H, W, C). Return float64 (out_h, out_w[, C]) using half-pixel centres."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _axis(n_in, n_out):
    """Integer neighbours (i0, i1) and blend weight t for every output index along one axis."""
    s = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
    s = np.maximum(s, 0.0)
    i0 = np.minimum(np.floor(s).astype(int), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, s - i0

def resize_bilinear(img, out_h, out_w):
    img = np.asarray(img, dtype=np.float64)
    y0, y1, t = _axis(img.shape[0], out_h)
    x0, x1, u = _axis(img.shape[1], out_w)
    extra = (None,) * (img.ndim - 2)              # broadcast weights over a channel axis if present
    t = t[(slice(None), None) + extra]
    u = u[(None, slice(None)) + extra]
    rows0, rows1 = img[y0], img[y1]               # (out_h, W[, C]) : interpolate rows first ...
    top = (1 - u) * rows0[:, x0] + u * rows0[:, x1]
    bot = (1 - u) * rows1[:, x0] + u * rows1[:, x1]
    return (1 - t) * top + t * bot                # ... then blend the two row results
'''

HINTS = [
    'The two axes are independent: compute the neighbour indices and weights for rows once and for columns once, as 1-D arrays.',
    'Use fancy indexing: `img[y0][:, x0]` gathers the top-left neighbour of every output pixel at once. Do the same for the other three corners.',
    'For one axis: `s = np.maximum((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0)`, `i0 = min(floor(s), n_in-1)`, `i1 = min(i0+1, n_in-1)`, `t = s - i0`. Reshape the weights to broadcast over channels.',
]

_r = np.random.default_rng(15)
TESTS = [
    (np.array([[0.0, 4.0]]), 1, 4),
    (np.array([[0.0, 1.0], [2.0, 3.0]]), 4, 4),
    (np.arange(16, dtype=float).reshape(4, 4), 2, 2),
    (np.arange(12, dtype=np.uint8).reshape(2, 2, 3), 3, 5),
    (_r.normal(size=(5, 7)), 5, 7),
    (_r.uniform(0, 255, size=(6, 4, 2)), 9, 3),
    (np.array([[7.0]]), 3, 2),
    (_r.normal(size=(10, 3)), 4, 8),
]

EXPLANATION = r"""
## Where the formula comes from
With half-pixel centres, output pixel $y$ covers the interval $[y, y+1)$ in output units, whose centre $y + 0.5$ maps to $(y + 0.5)\cdot\text{scale}$ in input units. Subtracting 0.5 converts that back to an input **index** (input pixel $k$ has its centre at $k + 0.5$). The clamps handle points that fall outside the outermost pixel centres, which happens at every border when upsampling.

## Why it is separable
Bilinear interpolation is linear interpolation along rows, then along columns. Computing index and weight vectors per axis and gathering with fancy indexing makes the whole resize vectorised: $O(\text{out\_h}\cdot\text{out\_w}\cdot C)$ time, no Python loops over pixels.

## Common bugs
- Using `s = y * scale` (the "asymmetric" convention). The output shifts by half a pixel and resizing to the same size is no longer the identity (test 5 checks it is).
- `align_corners=True` uses `s = y * (H - 1) / (out_h - 1)`, which maps corner to corner. It's a different, also common, convention; frameworks disagree on defaults, which is a classic source of off-by-half bugs when porting models.
- Clamping `s` at the top but not the integer index at the bottom, so `y1` reads past the end.
- Downsampling by large factors with bilinear aliases badly, because only 4 input pixels contribute. Anti-aliased resizers blur first or use a wider kernel.

## Interview follow-ups
- Implement `align_corners=True`. What happens when `out_h == 1`?
- How do you resize the box annotations consistently with the image? (Scale coordinates by `out/in`; with half-pixel centres no extra offset is needed for continuous box corners.)
"""
