import numpy as np

META = dict(title='Squeeze-and-excitation channel gating', tags=['Convolutional Networks', 'Attention', 'Computer Vision'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:04.02'],
            related=['resnet-residual-block', 'depthwise-separable-convolution', 'pooling-layers'])

STATEMENT = r"""
A squeeze-and-excitation (SE) block lets a conv layer reweight its own channels using global context:

1. **Squeeze**: global average pool each channel, $s_{n,c} = \frac{1}{HW}\sum_{i,j} x_{n,c,i,j}$.
2. **Excite**: a two-layer bottleneck MLP, $a = \sigma\big(\mathrm{ReLU}(s W_1 + b_1)\,W_2 + b_2\big)$, with `W1` `(C, C/r)` and `W2` `(C/r, C)`.
3. **Scale**: $y_{n,c,i,j} = a_{n,c}\, x_{n,c,i,j}$.

Write `se_block(x, W1, b1, W2, b2, mask=None)` that returns `(y, a)`.

## Masked squeeze
Real batches can contain padded images. `mask` is an optional float array `(N, H, W)` of 0s and 1s marking valid pixels; when given, the squeeze averages only over valid pixels:
$$s_{n,c} = \frac{\sum_{i,j} m_{n,i,j}\,x_{n,c,i,j}}{\sum_{i,j} m_{n,i,j}},$$
and $s_{n,c} = 0$ for an image with no valid pixel. The scaling step multiplies every pixel, valid or not.

## Output
`y`: float `(N, C, H, W)`; `a`: the gates, float `(N, C)` with values in $(0, 1)$. Compute the sigmoid stably.
"""

STARTER = '''import numpy as np

def se_block(x, W1, b1, W2, b2, mask=None):
    """x: (N, C, H, W). Return (y (N, C, H, W), gates (N, C))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def se_block(x, W1, b1, W2, b2, mask=None):
    x = np.asarray(x, dtype=float)
    if mask is None:
        s = x.mean(axis=(2, 3))                                       # (N, C)
    else:
        m = np.asarray(mask, dtype=float)[:, None]                    # (N, 1, H, W)
        cnt = m.sum(axis=(2, 3))                                      # (N, 1)
        s = np.divide((x * m).sum(axis=(2, 3)), cnt, out=np.zeros(x.shape[:2]), where=cnt > 0)
    u = np.maximum(s @ W1 + b1, 0.0) @ W2 + b2
    e = np.exp(-np.abs(u))
    a = np.where(u >= 0, 1 / (1 + e), e / (1 + e))
    return x * a[:, :, None, None], a
'''

HINTS = [
    'The squeeze is `x.mean(axis=(2, 3))`, giving one number per (sample, channel). Everything after that is an ordinary two-layer MLP on a `(N, C)` matrix.',
    'For the masked squeeze, broadcast the mask to `(N, 1, H, W)`, sum `x * m` and `m` over the spatial axes, and divide with `np.divide(..., where=count > 0)` so empty images give 0.',
    'Scale by reshaping the gates to `(N, C, 1, 1)`. Use the two-branch sigmoid so large logits do not overflow.',
]

_r = np.random.default_rng(131)
def _w(C, r):
    h = C // r
    return _r.normal(size=(C, h)) / np.sqrt(C), _r.normal(size=h) * 0.1, _r.normal(size=(h, C)) / np.sqrt(h), _r.normal(size=C) * 0.1
_m = np.ones((2, 3, 4)); _m[0, :, 2:] = 0; _m[1] = 0
TESTS = [
    (np.ones((1, 2, 1, 2)), np.array([[1.0], [1.0]]), np.zeros(1), np.array([[1.0, -1.0]]), np.zeros(2)),
    (np.array([[[[1.0, 3.0]], [[0.0, 0.0]]]]), np.eye(2), np.zeros(2), np.eye(2), np.zeros(2), np.array([[[1.0, 0.0]]])),
    (_r.normal(size=(2, 8, 4, 4)), *_w(8, 4)),
    (_r.normal(size=(3, 6, 5, 3)), *_w(6, 2)),
    (_r.normal(size=(2, 4, 3, 4)), *_w(4, 2), _m),
    (_r.normal(size=(1, 4, 2, 2)) * 50, _r.normal(size=(4, 2)) * 50, np.zeros(2), _r.normal(size=(2, 4)) * 50, np.zeros(4)),
    (_r.normal(size=(2, 3, 1, 1)), *_w(3, 3)),
]

EXPLANATION = r"""
## Channel attention
A convolution only sees a local window. The squeeze summarises each channel over the **whole** image, and the excitation MLP decides, from that summary, which channels to amplify or damp for this particular input. It's attention over channels with one query per image, and costs almost nothing: $2C^2/r$ weights per block (with $r = 16$, about 1% extra parameters in SE-ResNet-50) for a sizeable accuracy gain.

## Why a sigmoid, not a softmax
Channels aren't competing for a fixed budget: several can be fully on at once. Independent sigmoids allow that; a softmax over channels would force the gates to sum to 1.

## Masked pooling
With padded images, averaging over all pixels mixes padding into the channel statistics, so the same image gets different gates depending on how much padding it received. Masked means fix that; an all-padding image (test 5, second sample) has nothing to average and gets $s = 0$.

## Complexity
$O(NCHW)$ for the pool and the scaling, $O(NC^2/r)$ for the MLP.

## Common bugs
- Pooling over the batch or channel axis instead of the spatial axes.
- Applying the gates to the pooled vector instead of the feature map.
- Dividing by $HW$ in the masked case, or producing `nan` for an empty mask.

## Interview follow-ups
- Where is the SE block placed inside a residual block, and why before the addition?
- How do CBAM-style blocks add spatial attention on top of channel attention?
"""
