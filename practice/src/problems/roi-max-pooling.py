import numpy as np

META = dict(title='RoI max pooling with quantised bins', tags=['Object Detection', 'Convolutional Networks'], diff='Medium', runtime='numpy',
            related=['roi-align', 'max-pool-2d'])

STATEMENT = r"""
Write `roi_max_pool(feat, rois, out_size, scale)` in the style of Fast R-CNN.

- `feat` has shape `(C, H, W)`. Each RoI is `(x1, y1, x2, y2)` in image coordinates.
- Map each box to the feature map with `round(coord * scale)` for all four corners, then clamp to the map. The box is inclusive, so its width is `x2 - x1 + 1` (minimum 1), and likewise for height.
- Split the box into an `out_size x out_size` grid. Bin `i` along an axis of length `L` starting at `s` covers `[s + floor(i*L/out), s + ceil((i+1)*L/out))`, clipped to the map.
- Each output cell is the max over its bin, or 0 if the bin is empty.

Return an array of shape `(len(rois), C, out_size, out_size)`.
"""

STARTER = '''import numpy as np

def roi_max_pool(feat, rois, out_size, scale):
    """Return (R, C, out_size, out_size) pooled features."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def roi_max_pool(feat, rois, out_size, scale):
    F = np.asarray(feat, dtype=float)
    C, H, W = F.shape
    out = np.zeros((len(rois), C, out_size, out_size))
    for r, (x1, y1, x2, y2) in enumerate(rois):
        xs = min(max(int(round(x1 * scale)), 0), W - 1)
        ys = min(max(int(round(y1 * scale)), 0), H - 1)
        xe = min(max(int(round(x2 * scale)), 0), W - 1)
        ye = min(max(int(round(y2 * scale)), 0), H - 1)
        bw, bh = max(xe - xs + 1, 1), max(ye - ys + 1, 1)
        for i in range(out_size):
            h0 = ys + int(np.floor(i * bh / out_size))
            h1 = min(ys + int(np.ceil((i + 1) * bh / out_size)), H)
            for j in range(out_size):
                w0 = xs + int(np.floor(j * bw / out_size))
                w1 = min(xs + int(np.ceil((j + 1) * bw / out_size)), W)
                if h1 > h0 and w1 > w0:
                    out[r, :, i, j] = F[:, h0:h1, w0:w1].max(axis=(1, 2))
    return out
'''

HINTS = [
    'Quantise the box corners first, then work entirely in integer feature-map coordinates.',
    'Bins may overlap by one cell when the box length is not divisible by out_size; that is expected.',
    'Take the max over axes (1, 2) of a slice so all channels are pooled at once.',
]

_r = np.random.default_rng(5)
_f = _r.normal(size=(2, 8, 8))
TESTS = [
    (np.arange(16.0).reshape(1, 4, 4), [(0, 0, 3, 3)], 2, 1.0),
    (_f, [(0, 0, 15, 15)], 2, 0.5),
    (_f, [(2, 4, 10, 9), (0, 0, 3, 3)], 3, 0.5),
    (_f, [(5, 5, 5, 5)], 2, 1.0),
    (_f, [(-4, 1, 30, 6)], 4, 0.5),
    (np.arange(25.0).reshape(1, 5, 5), [(1, 1, 3, 3)], 2, 1.0),
]

EXPLANATION = r"""
## What RoI pooling does
A detector proposes boxes of arbitrary size, but the classification head needs a fixed-size input. RoI pooling cuts each box into a fixed grid and keeps the maximum activation per cell, so every box becomes a `C x k x k` tensor.

## Where it loses precision
The corners are rounded to integer cells and the bins are rounded again. Each rounding shifts the pooled region by up to half a cell, which at stride 16 can be 8 pixels in the image. That misalignment hurts small objects and pixel-level tasks such as masks, which is why Mask R-CNN replaced it with RoIAlign: no rounding, bilinear sampling at fractional positions.

## Interview follow-ups
- How does the gradient flow through a max over a bin?
- Why does misalignment matter more for segmentation than for box classification?
"""
