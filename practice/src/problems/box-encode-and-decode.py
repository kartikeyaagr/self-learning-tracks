import numpy as np

META = dict(title='Encode and decode box regression targets', tags=['Object Detection'], diff='Medium', runtime='numpy',
            lessons=[], related=['anchor-box-generation', 'iou-matrix', 'generalized-iou-loss'])

STATEMENT = r"""
Detectors don't regress raw coordinates. They predict **deltas** relative to an anchor: a centre shift measured in anchor widths and a log-scale change of size. That makes the targets roughly scale-invariant.

For an anchor with centre $(x_a, y_a)$ and size $(w_a, h_a)$, and a box with centre $(x, y)$ and size $(w, h)$, with weights $(\omega_x, \omega_y, \omega_w, \omega_h)$:

$$d_x = \omega_x \frac{x - x_a}{w_a}, \quad d_y = \omega_y \frac{y - y_a}{h_a}, \quad d_w = \omega_w \log\frac{w}{w_a}, \quad d_h = \omega_h \log\frac{h}{h_a}$$

Boxes are `[x1, y1, x2, y2]` with width `x2 - x1`, height `y2 - y1` and centre `x1 + width / 2` (continuous coordinates, no `+1`).

Write `box_coder(anchors, boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0))` that returns a tuple `(encoded, decoded)`:
- `encoded`: the deltas that turn each `anchors[k]` into `boxes[k]`, shape `(N, 4)` in the order `[dx, dy, dw, dh]`.
- `decoded`: the boxes obtained by applying each `deltas[k]` to `anchors[k]`, shape `(N, 4)` as `[x1, y1, x2, y2]`.

## Decoding details
1. Divide each delta by its weight.
2. Clamp the size deltas from above: $d_w, d_h \leftarrow \min(d_w, \log(1000/16))$, so a wild prediction can't overflow `exp`.
3. $x = d_x w_a + x_a$, $w = e^{d_w} w_a$ (same for $y, h$), then convert back to corners.

## Input
`anchors`, `boxes`, `deltas`: float arrays `(N, 4)`, `N >= 1`. Anchors and boxes have positive width and height.
"""

STARTER = '''import numpy as np

def box_coder(anchors, boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    """Return (encoded (N, 4) deltas for boxes w.r.t. anchors, decoded (N, 4) boxes from deltas)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

CLAMP = np.log(1000.0 / 16)

def _cxcywh(b):
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return b[:, 0] + 0.5 * w, b[:, 1] + 0.5 * h, w, h

def box_coder(anchors, boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    anchors, boxes, deltas = (np.asarray(a, dtype=np.float64) for a in (anchors, boxes, deltas))
    wx, wy, ww, wh = weights
    ax, ay, aw, ah = _cxcywh(anchors)

    bx, by, bw, bh = _cxcywh(boxes)
    encoded = np.stack([wx * (bx - ax) / aw, wy * (by - ay) / ah,
                        ww * np.log(bw / aw), wh * np.log(bh / ah)], axis=1)

    dx, dy = deltas[:, 0] / wx, deltas[:, 1] / wy
    dw = np.minimum(deltas[:, 2] / ww, CLAMP)          # clamp before exp
    dh = np.minimum(deltas[:, 3] / wh, CLAMP)
    cx, cy = dx * aw + ax, dy * ah + ay
    w, h = np.exp(dw) * aw, np.exp(dh) * ah
    decoded = np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=1)
    return encoded, decoded
'''

HINTS = [
    'Both directions are easiest in centre-size form `(cx, cy, w, h)`. Write a small helper that converts corners to it.',
    'Encoding: shifts are divided by the **anchor** size, sizes become log-ratios, then everything is multiplied by its weight. Decoding undoes each step in reverse order.',
    'Decode as `dw = min(deltas[:, 2] / ww, log(1000/16))`, `w = exp(dw) * aw`, `cx = dx * aw + ax`, then corners `cx -/+ w/2`. Return `(encoded, decoded)` as a tuple.',
]

_r = np.random.default_rng(24)
_a = np.hstack([_r.uniform(0, 50, (5, 2)), np.zeros((5, 2))]); _a[:, 2:] = _a[:, :2] + _r.uniform(5, 40, (5, 2))
_b = np.hstack([_r.uniform(0, 50, (5, 2)), np.zeros((5, 2))]); _b[:, 2:] = _b[:, :2] + _r.uniform(5, 40, (5, 2))
TESTS = [
    (np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([[2.0, 0.0, 12.0, 20.0]]), np.zeros((1, 4))),
    (np.array([[10.0, 10.0, 20.0, 30.0]]), np.array([[10.0, 10.0, 20.0, 30.0]]), np.array([[0.5, -0.5, 0.0, np.log(2.0)]])),
    (np.array([[0.0, 0.0, 4.0, 4.0]]), np.array([[1.0, 1.0, 3.0, 3.0]]), np.array([[0.0, 0.0, 10.0, -1.0]])),
    (_a, _b, _r.normal(size=(5, 4)), (10.0, 10.0, 5.0, 5.0)),
    (_a, _b, _r.normal(size=(5, 4)) * 3),
    (_a[:2], _a[:2], np.zeros((2, 4)), (10.0, 10.0, 5.0, 5.0)),
    (np.array([[5.0, 5.0, 6.0, 9.0]]), np.array([[0.0, 4.0, 2.0, 5.0]]), np.array([[-2.0, 1.0, 40.0, 40.0]]), (2.0, 2.0, 2.0, 2.0)),
]

EXPLANATION = r"""
## Why this parameterisation
Dividing the shift by the anchor size makes a "half an anchor to the right" target mean the same thing for a 16-pixel and a 512-pixel anchor. Log-ratios make size changes symmetric (doubling and halving are $\pm\log 2$) and keep decoded sizes positive. The weights (often $10, 10, 5, 5$) rescale the targets to roughly unit variance so the regression loss is balanced against the classification loss.

## Encode and decode are inverses
Without the clamp, `decode(anchor, encode(anchor, box)) == box` up to rounding (test 6 checks the zero-delta case). The clamp at $\log(1000/16) \approx 4.14$ caps a box at about 62 times its anchor's size; test 3 and test 7 exercise it.

## Complexity
$O(N)$, all vectorised.

## Common bugs
- Normalising the shift by the box's own size instead of the anchor's.
- Applying the weights in decode by multiplying instead of dividing.
- Clamping after `exp`: the overflow has already happened.
- Mixing the `+1` pixel convention from old codebases with continuous coordinates.

## Interview follow-ups
- Why do some detectors switch from this L1/smooth-L1 delta loss to an IoU-based loss on the decoded boxes? (The delta loss doesn't directly optimise the evaluation metric and treats the four coordinates as independent.)
- How would you encode a box relative to a **point** (anchor-free) instead of an anchor box?
"""
