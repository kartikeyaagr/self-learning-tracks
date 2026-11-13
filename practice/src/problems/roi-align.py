import numpy as np

META = dict(title='RoIAlign with bilinear sampling', tags=['Object Detection', 'Computer Vision'], diff='Hard', runtime='numpy',
            lessons=[], related=['bilinear-resize', 'adaptive-average-pool-2d', 'fpn-top-down-fusion'])

STATEMENT = r"""
Two-stage detectors crop a fixed-size feature grid for every proposal box. **RoIAlign** does this without rounding: it divides the box into bins, samples a few points per bin with bilinear interpolation, and averages them.

Write `roi_align(feat, rois, out_h, out_w, spatial_scale, sampling_ratio)`.

## Input
- `feat`: float array `(C, H, W)`. Feature pixel `(r, c)` has its value at continuous position `(r, c)` after the half-pixel shift below.
- `rois`: float array `(K, 4)` of `[x1, y1, x2, y2]` in **image** coordinates, with `x1 <= x2` and `y1 <= y2`.
- `spatial_scale`: image-to-feature scale (e.g. `1/16`). `sampling_ratio`: int `n >= 1` points per bin along each axis.

## Sample points (for one RoI)
1. `x_start = x1 * spatial_scale - 0.5`, `y_start = y1 * spatial_scale - 0.5`, `roi_w = (x2 - x1) * spatial_scale`, `roi_h = (y2 - y1) * spatial_scale`.
2. `bin_w = roi_w / out_w`, `bin_h = roi_h / out_h`.
3. Bin `(p, q)` samples at `y = y_start + p * bin_h + (iy + 0.5) * bin_h / n` and `x = x_start + q * bin_w + (ix + 0.5) * bin_w / n` for `iy, ix` in `0..n-1`.

## Bilinear value at (y, x)
- If `y < -1` or `y > H` or `x < -1` or `x > W`, the value is `0`.
- Otherwise (per axis, shown for `y`): set `y = max(y, 0)`, `y0 = floor(y)`. If `y0 >= H - 1`, set `y0 = y1 = H - 1` and `y = y0`; else `y1 = y0 + 1`. The weight is `ly = y - y0`.
- Value $= (1-l_y)(1-l_x) f_{y_0 x_0} + (1-l_y) l_x f_{y_0 x_1} + l_y (1-l_x) f_{y_1 x_0} + l_y l_x f_{y_1 x_1}$.

## Output
`out[k, c, p, q]` is the mean of the `n * n` sample values of bin `(p, q)` of RoI `k`. Return a `float64` array `(K, C, out_h, out_w)`.
"""

STARTER = '''import numpy as np

def roi_align(feat, rois, out_h, out_w, spatial_scale, sampling_ratio):
    """feat: (C, H, W), rois: (K, 4) image-space boxes. Return (K, C, out_h, out_w)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _axis(coords, n):
    """Neighbour indices, weights and validity for sample coordinates along one axis of length n."""
    valid = (coords >= -1.0) & (coords <= n)
    c = np.maximum(coords, 0.0)
    lo = np.floor(c).astype(int)
    edge = lo >= n - 1
    lo = np.where(edge, n - 1, lo)
    hi = np.where(edge, n - 1, lo + 1)
    c = np.where(edge, lo, c)
    return lo, hi, c - lo, valid

def roi_align(feat, rois, out_h, out_w, spatial_scale, sampling_ratio):
    feat = np.asarray(feat, dtype=np.float64)
    C, H, W = feat.shape
    n = sampling_ratio
    out = np.zeros((len(rois), C, out_h, out_w))
    for k, (x1, y1, x2, y2) in enumerate(np.asarray(rois, dtype=np.float64)):
        xs0, ys0 = x1 * spatial_scale - 0.5, y1 * spatial_scale - 0.5
        bw = (x2 - x1) * spatial_scale / out_w
        bh = (y2 - y1) * spatial_scale / out_h
        # Sample coordinates, ordered (bin, sample) so a reshape groups each bin's samples together.
        ys = (ys0 + np.arange(out_h)[:, None] * bh + (np.arange(n)[None, :] + 0.5) * bh / n).ravel()
        xs = (xs0 + np.arange(out_w)[:, None] * bw + (np.arange(n)[None, :] + 0.5) * bw / n).ravel()
        y0, y1_, ly, vy = _axis(ys, H)
        x0, x1_, lx, vx = _axis(xs, W)
        ly, lx = ly[:, None], lx[None, :]
        f = lambda r, c: feat[:, r][:, :, c]                 # (C, len(ys), len(xs))
        val = ((1 - ly) * (1 - lx) * f(y0, x0) + (1 - ly) * lx * f(y0, x1_)
               + ly * (1 - lx) * f(y1_, x0) + ly * lx * f(y1_, x1_))
        val = val * (vy[:, None] & vx[None, :])            # zero outside the extended border
        out[k] = val.reshape(C, out_h, n, out_w, n).mean(axis=(2, 4))
    return out
'''

HINTS = [
    'The sample grid of one RoI is separable: every sample is a pair (one of `out_h * n` y-values, one of `out_w * n` x-values). Compute those two 1-D coordinate lists first.',
    'Write a per-axis helper that returns `(lo, hi, weight, valid)` for an array of coordinates, applying the clamp and edge rules from the statement. Then the four corners are gathered with `feat[:, rows][:, :, cols]`.',
    'Order the coordinates as `(bin, sample)` (e.g. `start + p * bin + (i + 0.5) * bin / n`, raveled), multiply by the validity mask `vy[:, None] & vx[None, :]`, and average with `.reshape(C, out_h, n, out_w, n).mean(axis=(2, 4))`.',
]

_r = np.random.default_rng(27)
_feat = _r.normal(size=(3, 6, 7))
TESTS = [
    (np.arange(16, dtype=float).reshape(1, 4, 4), np.array([[0.5, 0.5, 2.5, 2.5]]), 1, 1, 1.0, 1),
    (np.arange(16, dtype=float).reshape(1, 4, 4), np.array([[0.0, 0.0, 4.0, 4.0]]), 2, 2, 1.0, 2),
    (np.arange(12, dtype=float).reshape(1, 3, 4), np.array([[0.0, 0.0, 16.0, 12.0]]), 3, 4, 0.25, 1),
    (_feat, np.array([[1.3, 2.2, 20.7, 15.1], [0.0, 0.0, 27.9, 23.6], [10.2, 4.4, 10.9, 5.3]]), 2, 3, 0.25, 2),
    (_feat, np.array([[-10.3, -7.7, 9.1, 30.6], [20.2, 1.1, 34.7, 13.9]]), 3, 3, 0.25, 3),
    (_r.normal(size=(2, 5, 5)), np.array([[3.3, 3.1, 3.3, 3.1]]), 2, 2, 1.0, 2),
    (_r.normal(size=(4, 8, 6)), _r.uniform(0, 5, (4, 2)) @ np.array([[1.0, 0, 1, 0], [0, 1.0, 0, 1]]) + np.array([0, 0, 3.7, 2.9]), 2, 2, 1.0, 2),
]

EXPLANATION = r"""
## Why no rounding
The older RoIPool snapped box corners and bin edges to integer feature pixels. At stride 16 a rounding of half a feature pixel is 8 image pixels, which hurts mask and keypoint accuracy. RoIAlign keeps every coordinate continuous and reads values by interpolation, so the pooled features move smoothly with the box.

## The half-pixel shift
Subtracting 0.5 aligns the continuous box with pixel **centres**: an RoI covering image pixels $[0, 16)$ at scale $1/16$ maps to the span $[-0.5, 0.5)$ around feature pixel 0's centre. Without it the pooled grid is offset by half a feature pixel (the "aligned=False" behaviour of early implementations).

## Border rules
Samples within one pixel outside the map are clamped to the edge (so a box touching the border still sees real values), and samples further out contribute 0 but still count in the average. Test 5 has RoIs reaching past both sides.

## Complexity
$O(K \cdot C \cdot \text{out\_h}\,\text{out\_w}\, n^2)$. The separable sample grid means gathering costs one fancy index per corner instead of a loop over samples.

## Common bugs
- Dividing by the number of **valid** samples instead of $n^2$.
- Applying the clamp `y = max(y, 0)` before the `y < -1` validity test.
- Treating `(x, y)` as `(row, col)`. Boxes are x-first; arrays are row-first.
- Ignoring the degenerate RoI (test 6): zero width gives every sample the same point.

## Interview follow-ups
- What sampling ratio do implementations use when it's "adaptive"? (`ceil(roi_h / out_h)` per RoI, so large boxes get more samples.)
- Write the backward pass: each sample scatters its upstream gradient to its four neighbours with the bilinear weights.
"""
