import numpy as np

META = dict(title='From RPN outputs to region proposals: decode, clip, filter, NMS', tags=['Object Detection', 'Computer Vision'],
            diff='Hard', runtime='numpy', lessons=['dlcore:04.02'],
            related=['rpn-anchor-target-assignment', 'box-encode-and-decode', 'non-maximum-suppression', 'anchor-box-generation'])

STATEMENT = r"""
At inference the region proposal network's raw outputs become a short list of candidate boxes for the second stage. Implement the proposal layer for one image.

Write `rpn_proposals(anchors, logits, deltas, img_hw, pre_nms, nms_thr, post_nms, min_size)` returning `(boxes, scores)`.

## Steps
1. **Decode** each anchor `(x1, y1, x2, y2)` with its deltas $(d_x, d_y, d_w, d_h)$: with anchor centre $x_a, y_a$ and size $w_a, h_a$, clamp $d_w, d_h$ to at most $\log(1000/16)$, then $x = x_a + d_xw_a$, $y = y_a + d_yh_a$, $w = w_ae^{d_w}$, $h = h_ae^{d_h}$, and convert back to corners.
2. **Clip** to the image: $x \in [0, W]$, $y \in [0, H]$ with `img_hw = (H, W)`.
3. **Score**: $\sigma(\text{logit})$. Keep the `pre_nms` highest-scoring boxes (stable descending order, so ties keep the lower index first).
4. **Filter** out boxes whose width or height is $<$ `min_size`.
5. **NMS**: walk the boxes in descending score order and drop any box whose IoU with an already-kept box is $>$ `nms_thr`.
6. Keep the first `post_nms`.

Return `boxes` `(K, 4)` and `scores` `(K,)` in descending score order.
"""

STARTER = '''import numpy as np

def rpn_proposals(anchors, logits, deltas, img_hw, pre_nms, nms_thr, post_nms, min_size):
    """Return (boxes (K, 4), scores (K,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def rpn_proposals(anchors, logits, deltas, img_hw, pre_nms, nms_thr, post_nms, min_size):
    a, d = np.asarray(anchors, dtype=float), np.asarray(deltas, dtype=float)
    H, W = img_hw
    aw, ah = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
    ax, ay = a[:, 0] + aw / 2, a[:, 1] + ah / 2
    dw, dh = np.minimum(d[:, 2], np.log(1000 / 16)), np.minimum(d[:, 3], np.log(1000 / 16))
    x, y, w, h = ax + d[:, 0] * aw, ay + d[:, 1] * ah, aw * np.exp(dw), ah * np.exp(dh)
    b = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
    s = 1 / (1 + np.exp(-np.asarray(logits, dtype=float)))
    order = np.argsort(-s, kind='stable')[:pre_nms]
    b, s = b[order], s[order]
    ok = ((b[:, 2] - b[:, 0]) >= min_size) & ((b[:, 3] - b[:, 1]) >= min_size)
    b, s = b[ok], s[ok]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = []
    for i in range(len(b)):
        if keep:
            k = np.array(keep)
            iw = np.clip(np.minimum(b[k, 2], b[i, 2]) - np.maximum(b[k, 0], b[i, 0]), 0, None)
            ih = np.clip(np.minimum(b[k, 3], b[i, 3]) - np.maximum(b[k, 1], b[i, 1]), 0, None)
            inter = iw * ih
            if (inter / (area[k] + area[i] - inter)).max() > nms_thr:
                continue
        keep.append(i)
        if len(keep) == post_nms:
            break
    return b[keep], s[keep]
'''

HINTS = [
    'Decode in centre-size form, then convert to corners and clip. The clamp on $d_w, d_h$ stops a wild prediction from producing an overflowing exponential.',
    'Sort once with a stable descending sort and truncate to `pre_nms` before filtering: every later step preserves that order.',
    'Greedy NMS: a box is kept if its IoU with every box kept so far is at most the threshold. You can stop as soon as `post_nms` boxes are kept.',
]

_r = np.random.default_rng(2107)
def _grid_anchors(H, W, stride, sizes, ratios):
    out = []
    for cy in np.arange(stride / 2, H, stride):
        for cx in np.arange(stride / 2, W, stride):
            for s in sizes:
                for r in ratios:
                    w, h = s * np.sqrt(1 / r), s * np.sqrt(r)
                    out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(out)
def _c(H, W, stride, sizes, ratios, pre, thr, post, ms, dscale=0.2):
    a = _grid_anchors(H, W, stride, sizes, ratios)
    return (a, _r.normal(size=len(a)) * 2, _r.normal(size=(len(a), 4)) * dscale, (H, W), pre, thr, post, ms)
TESTS = [
    _c(16, 16, 8, [8], [1.0], 10, 0.7, 10, 0.0),
    _c(32, 32, 8, [8, 16], [0.5, 1.0, 2.0], 50, 0.7, 10, 1.0),
    _c(32, 48, 8, [16, 32], [0.5, 1.0, 2.0], 100, 0.5, 20, 4.0),
    _c(40, 40, 10, [10, 20, 40], [1.0, 2.0], 30, 0.3, 30, 2.0, dscale=0.5),
    _c(24, 24, 6, [12], [0.5, 2.0], 64, 0.9, 5, 0.0),
]

EXPLANATION = r"""
## Where this sits in Faster R-CNN
A shared backbone produces a feature map; the RPN scores and refines a dense set of anchors on it; this layer turns that into around 300 to 2000 proposals; RoIAlign (RoIPool in the original paper) crops a fixed-size feature for each, and the second-stage head classifies and refines them. Because the RPN shares the backbone with the detector, proposals cost little extra, which was the paper's main speedup over Selective Search.

## Why each filter
- Top-$k$ before NMS bounds NMS's quadratic cost; with tens of thousands of anchors per image this matters.
- Clipping keeps proposals inside the image so RoI pooling has valid features.
- Tiny boxes carry almost no features and are usually noise.
- NMS at 0.7 removes near-duplicates while keeping overlapping distinct objects; it is deliberately looser than the 0.5 used on final detections.

## Common bugs
- Filtering small boxes before taking the top `pre_nms`, which changes which boxes survive.
- Applying NMS with `>=` instead of `>`.
- Forgetting the delta clamp, which can turn one bad prediction into an infinite box.

## Interview follow-ups
- How do training and inference differ in `pre_nms` and `post_nms`, and why?
- With an FPN, are proposals selected per level or across levels?
"""
