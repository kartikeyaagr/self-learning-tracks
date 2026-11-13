import numpy as np

META = dict(title='Channel dropout for feature maps', tags=['Neural Networks', 'Training', 'Convolutional Networks'], diff='Easy',
            runtime='numpy', lessons=['air-fanout-v1:05.07'], related=['dropout', 'batch-normalization'])

STATEMENT = r"""
Ordinary dropout zeroes individual activations. In a convolutional feature map neighbouring pixels are strongly correlated, so dropping single pixels barely regularises anything. **Channel dropout** (often called spatial dropout or `Dropout2d`) drops whole channels instead: for each sample, a channel is either kept everywhere or zeroed everywhere.

Write `channel_dropout(x, u, p, train, dy)` that returns the tuple `(y, dx)`: the forward output and the gradient of $\sum y \odot dy$ with respect to `x`.

## Conventions
- `x`, `dy`: float arrays `(N, C, H, W)`. `u`: float array `(N, C)` of uniform draws in $[0, 1)$, one per (sample, channel). `p`: drop probability, $0 \le p < 1$. `train`: bool.
- When `train` is `True`: channel $(n, c)$ is **kept** iff `u[n, c] >= p`. Kept channels are scaled by $1/(1 - p)$ (inverted dropout), dropped channels become 0.
- When `train` is `False`: the layer is the identity, `y = x` and `dx = dy`, and `u` is ignored.
- Return two float arrays of shape `(N, C, H, W)`.
"""

STARTER = '''import numpy as np

def channel_dropout(x, u, p, train, dy):
    """x, dy: (N, C, H, W); u: (N, C) uniforms. Return (y, dx)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def channel_dropout(x, u, p, train, dy):
    x, dy = np.asarray(x, dtype=float), np.asarray(dy, dtype=float)
    if not train:
        return x.copy(), dy.copy()
    keep = (np.asarray(u) >= p) / (1.0 - p)           # (N, C): 0 or 1/(1-p)
    scale = keep[:, :, None, None]                    # broadcast over H and W
    return x * scale, dy * scale                      # y is linear in x, so the gradient uses the same mask
'''

HINTS = [
    'The mask has one entry per (sample, channel), not per pixel. Build it from `u` with shape `(N, C)`.',
    'Fold the keep test and the rescaling into one array: `(u >= p) / (1 - p)`. Then reshape it to `(N, C, 1, 1)` so it broadcasts over the spatial axes.',
    '`y = x * scale` is linear in `x`, so `dx = dy * scale` with the very same `scale`. In eval mode both are pass-throughs.',
]

_r = np.random.default_rng(21)
_x = _r.normal(size=(2, 3, 2, 2))
_dy = _r.normal(size=(2, 3, 2, 2))
_u = np.array([[0.1, 0.7, 0.5], [0.9, 0.2, 0.49]])
TESTS = [
    (np.ones((1, 2, 1, 2)), np.array([[0.2, 0.8]]), 0.5, True, np.ones((1, 2, 1, 2))),
    (np.arange(4.0).reshape(1, 1, 2, 2), np.array([[0.3]]), 0.5, False, np.ones((1, 1, 2, 2))),
    (_x, _u, 0.5, True, _dy),
    (_x, _u, 0.0, True, _dy),
    (_x, np.array([[0.25, 0.25, 0.26], [0.0, 0.99, 0.25]]), 0.25, True, _dy),
    (_x, _u, 0.5, False, _dy),
    (_r.normal(size=(3, 4, 3, 5)), _r.random((3, 4)), 0.3, True, _r.normal(size=(3, 4, 3, 5))),
]

EXPLANATION = r"""
## Why scale by $1/(1-p)$
Each channel survives with probability $1 - p$, so $\mathbb E[\text{mask}/(1-p)] = 1$ and the expected activation equals the eval-mode activation. Scaling at train time ("inverted" dropout) means inference needs no change at all.

## Why drop whole channels
A 3x3 conv sees a pixel's neighbours, which carry almost the same information. Dropping isolated pixels is easy for the next layer to interpolate around, so the noise does little. Removing an entire feature map forces the network not to rely on any one channel, which is the co-adaptation dropout is meant to break.

## Complexity
$O(NCHW)$ time; the mask itself is only $O(NC)$.

## Common bugs
- Building the mask per pixel (that is ordinary dropout).
- Using `u > p` instead of `u >= p`; test 5 has draws exactly equal to `p`.
- Forgetting that `dx` uses the same mask as the forward pass, or rescaling in eval mode.

## Interview follow-ups
- What happens if you use dropout right before BatchNorm? (The train-time variance includes dropout noise, the eval-time variance does not, so the statistics shift.)
- How would you estimate predictive uncertainty with dropout left on at test time (MC dropout)?
"""
