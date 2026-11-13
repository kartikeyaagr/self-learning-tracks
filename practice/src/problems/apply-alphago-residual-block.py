import numpy as np

META = dict(title='Residual tower block (conv, batch norm, skip)', tags=['Convolutional Networks', 'Neural Networks'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:04.02', 'ieng:02.03'],
            related=['encode-alphago-zero-board-history', 'policy-and-value-heads', 'assemble-compact-dual-head-network'])

STATEMENT = r"""
The trunk of an AlphaGo Zero-style network is a stack of identical residual blocks that keep the board size and channel count fixed. One block maps $x \in \mathbb{R}^{C \times N \times N}$ to

$$y = \mathrm{relu}\Big(x + \mathrm{BN}_2\big(\mathrm{conv}_2(\mathrm{relu}(\mathrm{BN}_1(\mathrm{conv}_1(x))))\big)\Big)$$

Write `residual_block(x, w1, bn1, w2, bn2, eps=1e-5)`.

## Convolution
Each `conv` is a 3x3 convolution with **no bias**, stride 1 and one ring of **zero padding**, so the output is also `N x N`. Weights have shape `(C_out, C_in, 3, 3)` with `C_out = C_in = C`. Use the deep-learning convention (cross-correlation, no kernel flip):

$$\mathrm{conv}(x)[o, i, j] = \sum_{c=0}^{C-1} \sum_{u=0}^{2} \sum_{v=0}^{2} w[o, c, u, v]\; x_{\text{pad}}[c,\, i + u,\, j + v]$$

where $x_{\text{pad}}$ is $x$ with one row/column of zeros added on every side.

## Batch norm (inference mode)
`bn1` and `bn2` are tuples `(gamma, beta, mean, var)`, each of shape `(C,)`. Per channel $o$:

$$\mathrm{BN}(h)[o] = \gamma_o \frac{h[o] - \mu_o}{\sqrt{\sigma^2_o + \varepsilon}} + \beta_o$$

using the stored running statistics (not statistics of `x`).

## Output
A float array of shape `(C, N, N)`.
"""

STARTER = '''import numpy as np

def residual_block(x, w1, bn1, w2, bn2, eps=1e-5):
    """x: (C, N, N); w1, w2: (C, C, 3, 3); bn = (gamma, beta, mean, var), each (C,). Return (C, N, N)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def conv3x3(x, w):
    """Same-padding 3x3 cross-correlation: (C_in, N, N) x (C_out, C_in, 3, 3) -> (C_out, N, N)."""
    n = x.shape[-1]
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((w.shape[0], n, n))
    for u in range(3):                              # 9 shifted views, each a channel-mixing matmul
        for v in range(3):
            out += np.einsum('oc,chw->ohw', w[:, :, u, v], xp[:, u:u + n, v:v + n])
    return out

def batch_norm(h, bn, eps):
    gamma, beta, mean, var = (np.asarray(t, dtype=float)[:, None, None] for t in bn)
    return gamma * (h - mean) / np.sqrt(var + eps) + beta

def residual_block(x, w1, bn1, w2, bn2, eps=1e-5):
    x = np.asarray(x, dtype=float)
    h = np.maximum(batch_norm(conv3x3(x, np.asarray(w1, dtype=float)), bn1, eps), 0.0)
    h = batch_norm(conv3x3(h, np.asarray(w2, dtype=float)), bn2, eps)
    return np.maximum(x + h, 0.0)                   # skip connection is added before the final relu
'''

HINTS = [
    'Write two helpers: `conv3x3(x, w)` and `batch_norm(h, bn, eps)`. The block is then four lines.',
    'For the convolution, pad once with `np.pad(x, ((0,0),(1,1),(1,1)))`. For each kernel offset `(u, v)` the slice `xp[:, u:u+N, v:v+N]` lines up with the output grid; mix channels with `np.einsum("oc,chw->ohw", w[:, :, u, v], slice)` and add up the 9 terms.',
    'Batch-norm parameters are per channel, so reshape them to `(C, 1, 1)` to broadcast. Order: conv, BN, relu, conv, BN, then add `x`, then relu.',
]

def _bn(rng, c):
    return (rng.uniform(0.5, 1.5, c), rng.normal(0, 0.1, c), rng.normal(0, 0.1, c), rng.uniform(0.5, 2.0, c))

_id = (np.ones(1), np.zeros(1), np.zeros(1), np.ones(1) - 1e-5)
_center = np.zeros((1, 1, 3, 3)); _center[0, 0, 1, 1] = 1.0
_ones = np.ones((1, 1, 3, 3))
_rng = np.random.default_rng(13)
_x2 = _rng.normal(size=(2, 3, 3))
_w2a, _w2b = _rng.normal(0, 0.4, (2, 2, 3, 3)), _rng.normal(0, 0.4, (2, 2, 3, 3))
_bn2a, _bn2b = _bn(_rng, 2), _bn(_rng, 2)
_x4 = _rng.normal(size=(4, 5, 5))
_w4a, _w4b = _rng.normal(0, 0.3, (4, 4, 3, 3)), _rng.normal(0, 0.3, (4, 4, 3, 3))
_bn4a, _bn4b = _bn(_rng, 4), _bn(_rng, 4)

TESTS = [
    (np.array([[[1.0, -2.0], [3.0, 0.5]]]), _center, _id, _center, _id),
    (np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]), _ones, _id, _ones, _id),
    (np.array([[[1.0, -1.0, 2.0], [0.0, 3.0, -2.0], [1.0, 1.0, -1.0]]]), _ones, _id, -0.5 * _center, _id),
    (_x2, _w2a, _bn2a, _w2b, _bn2b),
    (_x2, _w2a, _bn2a, _w2b, _bn2b, 1e-3),
    (_x4, _w4a, _bn4a, _w4b, _bn4b),
    (np.abs(_x4), np.zeros((4, 4, 3, 3)), _bn4a, _w4b, _bn4b),
]

EXPLANATION = r"""
## Why this block
The identity path lets gradients flow straight through a deep tower, so a 20 or 40 block network trains as easily as a shallow one. Inside the block, two 3x3 convolutions grow the receptive field by 4 points per block; Go needs whole-board context (think of a ladder), and stacking blocks is how a local operator sees the whole board. Batch norm keeps activations well scaled; at inference time it is a fixed per-channel affine map, which is why engines fold it into the convolution weights.

## The convolution as 9 matmuls
Zero padding means every output point has a full 3x3 window. For a fixed kernel offset $(u, v)$, the shifted slice of the padded input aligns with the output grid, and the contribution is a $C_\text{out} \times C_\text{in}$ matrix applied at every point. Summing the 9 offsets is exactly the triple sum in the statement and avoids building a large im2col matrix.

## Complexity
$9 C^2 N^2$ multiply-adds per convolution, two per block. For $C = 256$, $N = 19$ that is about 215M per block.

## Common bugs
- Flipping the kernel (true convolution) instead of cross-correlating.
- Normalising with the batch's own mean and variance at inference time.
- Applying the final relu before the skip addition, or forgetting it.
- Padding the channel axis as well as the spatial axes.

## Interview follow-ups
- Fold batch norm into the preceding convolution: what are the new weights and bias?
- KataGo adds global pooling inside some blocks. What can a global pooling branch express that stacked 3x3 convolutions struggle with?
"""
