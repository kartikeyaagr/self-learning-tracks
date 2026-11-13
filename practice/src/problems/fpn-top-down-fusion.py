import numpy as np

META = dict(title='Feature pyramid top-down pathway', tags=['Object Detection', 'Convolutional Networks'], diff='Medium', runtime='numpy',
            lessons=[], related=['anchor-box-generation', 'bilinear-resize', 'roi-align'])

STATEMENT = r"""
A Feature Pyramid Network gives every level of a CNN backbone the semantics of the deepest level. It projects each backbone map to a common width `D` with a 1x1 convolution (the **lateral** connection), then walks from the coarsest level to the finest, upsampling the running result and adding it to the next lateral.

Write `fpn_top_down(features, lateral_w, lateral_b)`.

## Input
- `features`: list of `L >= 1` float arrays, **finest first**: `features[l]` has shape `(C_l, H_l, W_l)`, and each level is no larger than the one before it (`H_{l+1} <= H_l`, `W_{l+1} <= W_l`).
- `lateral_w[l]`: `(D, C_l)` weights and `lateral_b[l]`: `(D,)` bias of level `l`'s 1x1 convolution.

## Computation
1. Laterals: $\Lambda_l[d, i, j] = b_l[d] + \sum_c w_l[d, c]\, F_l[c, i, j]$.
2. Top-down: $P_{L-1} = \Lambda_{L-1}$, and for $l = L-2$ down to $0$: $P_l = \Lambda_l + \mathrm{up}(P_{l+1})$.
3. `up` is **nearest-neighbour** resizing to level `l`'s size: output row `i` copies source row `(i * H_src) // H_dst` (integer division), and the same for columns. This handles any size ratio, including odd sizes where an exact 2x doesn't fit.

## Output
A list of `L` `float64` arrays, finest first: `P_l` of shape `(D, H_l, W_l)`.
"""

STARTER = '''import numpy as np

def fpn_top_down(features, lateral_w, lateral_b):
    """features: list of (C_l, H_l, W_l), finest first. Return list of (D, H_l, W_l), finest first."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _nearest(x, H, W):
    h, w = x.shape[1:]
    rows = (np.arange(H) * h) // H
    cols = (np.arange(W) * w) // W
    return x[:, rows][:, :, cols]

def fpn_top_down(features, lateral_w, lateral_b):
    lat = [np.einsum('dc,chw->dhw', np.asarray(w, float), np.asarray(f, float)) + np.asarray(b, float)[:, None, None]
           for f, w, b in zip(features, lateral_w, lateral_b)]
    out = [lat[-1]]
    for l in range(len(lat) - 2, -1, -1):          # coarse -> fine
        _, H, W = lat[l].shape
        out.append(lat[l] + _nearest(out[-1], H, W))
    return out[::-1]                                # finest first
'''

HINTS = [
    'A 1x1 convolution is just a matrix product over the channel axis, applied at every pixel.',
    'Compute all laterals first. Then start from the coarsest one and repeatedly upsample the current result to the next finer size and add.',
    'Nearest resize with index arrays: `rows = (np.arange(H) * h) // H`, `cols = (np.arange(W) * w) // W`, then `x[:, rows][:, :, cols]`. Build the list coarse to fine and reverse it at the end.',
]

_r = np.random.default_rng(26)
def _level(C, H, W):
    return _r.normal(size=(C, H, W))
TESTS = [
    ([np.arange(4, dtype=float).reshape(1, 2, 2), np.array([[[10.0]]])], [np.array([[1.0]]), np.array([[1.0]])], [np.zeros(1), np.zeros(1)]),
    ([np.ones((2, 2, 2)), np.ones((1, 1, 1))], [np.array([[1.0, -1.0]]), np.array([[2.0]])], [np.array([0.5]), np.array([1.0])]),
    ([np.array([[[3.0]]])], [np.array([[2.0], [-1.0]])], [np.array([0.0, 1.0])]),
    ([_level(3, 8, 8), _level(4, 4, 4), _level(5, 2, 2)],
     [_r.normal(size=(2, 3)), _r.normal(size=(2, 4)), _r.normal(size=(2, 5))], [_r.normal(size=2) for _ in range(3)]),
    ([_level(2, 7, 5), _level(3, 4, 3), _level(2, 2, 2), _level(4, 1, 1)],
     [_r.normal(size=(3, 2)), _r.normal(size=(3, 3)), _r.normal(size=(3, 2)), _r.normal(size=(3, 4))], [_r.normal(size=3) for _ in range(4)]),
    ([_level(1, 6, 9), _level(1, 3, 3)], [np.array([[1.0]]), np.array([[1.0]])], [np.zeros(1), np.zeros(1)]),
]

EXPLANATION = r"""
## Why top-down plus lateral
Deep backbone layers know *what* is in the image but at coarse resolution; shallow layers are precise about *where* but semantically weak. The top-down pass carries the deep semantics to every scale, and the laterals reinject the fine spatial detail. A detector then runs the same head on every $P_l$, assigning small objects to fine levels and large ones to coarse levels.

## Nearest upsampling with integer indices
`(i * h) // H` is the source pixel whose span contains output pixel $i$'s left edge. For an exact 2x ratio it repeats every source pixel twice. For a 7-row level above a 4-row level it maps rows $0..6$ to $0, 0, 1, 1, 2, 2, 3$, so odd input sizes need no special-casing.

## Complexity
Dominated by the laterals: $O(\sum_l D\, C_l H_l W_l)$. The top-down additions are $O(\sum_l D H_l W_l)$.

## Common bugs
- Iterating fine to coarse, so each level only receives the unmodified lateral from above instead of the accumulated sum.
- Upsampling the **lateral** of the coarser level instead of its fused output $P_{l+1}$.
- Hard-coding `np.repeat(x, 2, axis)`: breaks as soon as a level has odd size.
- Returning the list coarse-first.

## Interview follow-ups
- Real FPNs apply a 3x3 conv to each $P_l$ after the sum. Why? (To reduce the aliasing introduced by nearest upsampling.)
- Which level should an RoI of size $w \times h$ be pooled from? (The usual rule is $l = \lfloor l_0 + \log_2(\sqrt{wh}/224) \rfloor$, clamped to the available levels.)
"""
