import numpy as np

META = dict(title='Anchor boxes over a feature map', tags=['Object Detection'], diff='Medium', runtime='numpy',
            lessons=[], related=['box-encode-and-decode', 'iou-matrix', 'fpn-top-down-fusion'])

STATEMENT = r"""
Anchor-based detectors (Faster R-CNN, RetinaNet, SSD) place a fixed set of reference boxes at every cell of a feature map and learn to adjust them. Each anchor has an area set by a **size** and a shape set by an **aspect ratio** $r = h / w$.

Write `generate_anchors(feat_h, feat_w, stride, sizes, ratios)`.

## Geometry
- Feature cell `(i, j)` (row `i`, column `j`) has its centre at image coordinates $c_x = (j + 0.5)\cdot\text{stride}$, $c_y = (i + 0.5)\cdot\text{stride}$.
- For a size $s$ and ratio $r$ the anchor is $w = s / \sqrt{r}$ wide and $h = s \sqrt{r}$ tall, so its area is $s^2$ for every ratio.
- The box is `[cx - w/2, cy - h/2, cx + w/2, cy + h/2]`.

## Ordering
Cells in row-major order (all of row 0 left to right, then row 1, ...). Within a cell, loop over `sizes` in the given order, and for each size over `ratios` in the given order. With `A = len(sizes) * len(ratios)` the anchors of cell `(i, j)` occupy rows `(i * feat_w + j) * A` to `(i * feat_w + j + 1) * A - 1`.

## Input / output
- `feat_h, feat_w`: ints `>= 1`; `stride`: positive number; `sizes`, `ratios`: non-empty lists of positive floats.
- Return a `float64` array of shape `(feat_h * feat_w * A, 4)`.
"""

STARTER = '''import numpy as np

def generate_anchors(feat_h, feat_w, stride, sizes, ratios):
    """Return (feat_h * feat_w * len(sizes) * len(ratios), 4) anchors as [x1, y1, x2, y2]."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def generate_anchors(feat_h, feat_w, stride, sizes, ratios):
    s = np.asarray(sizes, dtype=float)[:, None]           # (S, 1)
    r = np.sqrt(np.asarray(ratios, dtype=float))[None, :]  # (1, R)
    w = (s / r).ravel()                                   # (A,), sizes outer, ratios inner
    h = (s * r).ravel()
    base = np.stack([-w / 2, -h / 2, w / 2, h / 2], axis=1)   # (A, 4) anchors centred at the origin

    cy, cx = np.meshgrid((np.arange(feat_h) + 0.5) * stride,
                         (np.arange(feat_w) + 0.5) * stride, indexing='ij')
    centres = np.stack([cx, cy, cx, cy], axis=-1).reshape(-1, 1, 4)   # (cells, 1, 4), row-major
    return (centres + base[None]).reshape(-1, 4)
'''

HINTS = [
    'Split the problem in two: the `A` box shapes (the same at every cell) and the cell centres. Every anchor is a centre plus a shape offset.',
    'Build an `(A, 4)` array of offsets `[-w/2, -h/2, w/2, h/2]` with sizes as the outer loop, and a `(cells, 4)` array of `[cx, cy, cx, cy]`. Broadcasting `(cells, 1, 4) + (1, A, 4)` gives every combination.',
    '`np.meshgrid(ys, xs, indexing="ij")` gives row-major cells; `w = (sizes[:, None] / np.sqrt(ratios)[None, :]).ravel()` keeps sizes outer and ratios inner. Reshape the sum to `(-1, 4)`.',
]

TESTS = [
    (1, 1, 16, [32.0], [1.0]),
    (1, 2, 10, [10.0], [0.25, 1.0, 4.0]),
    (2, 2, 8, [16.0, 32.0], [0.5, 2.0]),
    (3, 1, 4.0, [4.0, 8.0, 16.0], [1.0]),
    (2, 3, 32, [128.0, 256.0], [0.5, 1.0, 2.0]),
    (4, 4, 2, [3.0], [0.7]),
]

EXPLANATION = r"""
## Why area-preserving ratios
With $w = s/\sqrt r$ and $h = s\sqrt r$, the product $wh = s^2$ doesn't depend on $r$. So `sizes` controls scale and `ratios` controls shape independently, and the anchor set covers a grid of scales times shapes.

## Shape offsets plus centres
All cells share the same `A` shapes, so the generator only needs an `(A, 4)` template and an `(H W, 4)` array of centres. Broadcasting their sum produces all $H W A$ anchors with no Python loops, and the reshape order (cells outer, shapes inner) is exactly the order a detection head produces its per-anchor outputs in when it predicts `A` values per cell.

## Complexity
$O(H W A)$ time and memory: the output itself.

## Common bugs
- Mixing up $r = h/w$ and $r = w/h$, which swaps tall and wide anchors.
- Centring at `j * stride` instead of `(j + 0.5) * stride`. Both appear in real code; mismatching the convention used during training shifts every box by half a stride.
- `np.meshgrid` defaults to `indexing="xy"`, which transposes the grid and puts the anchors in column-major order.
- Making the ratio loop outer: shapes still look right, but the anchor order no longer matches the head's channel layout.

## Interview follow-ups
- Given ground-truth boxes, assign each anchor a label: positive if its best IoU is above 0.7, negative below 0.3, ignored in between, plus force each ground truth's best anchor to be positive.
- How do anchor-free detectors (FCOS, CenterNet) avoid all this? (Predict distances from each cell centre to the box sides, or a centre heatmap.)
"""
