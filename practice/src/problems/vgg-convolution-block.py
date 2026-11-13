import numpy as np

META = dict(title='A VGG block: stacked 3x3 convs, then pool', tags=['Convolutional Networks', 'Computer Vision'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:04.02', 'air-fanout-v1:04.08'],
            related=['2d-convolution', 'pooling-layers', 'resnet-residual-block'])

STATEMENT = r"""
VGG networks are built from one repeated unit: a few $3\times3$ convolutions with "same" padding, each followed by ReLU, then a $2\times2$ max pool that halves the resolution.

Write `vgg_block(x, convs, pool=True)` that returns `(y, n_params, receptive_field)`.

## Inputs
- `x`: float `(N, C, H, W)`.
- `convs`: a list of `(W, b)` pairs. `W` has shape `(F, C_prev, 3, 3)`, `b` has shape `(F,)`; each conv's input channels equal the previous output channels.

## Computation
1. Each conv uses stride 1 and zero padding 1 (so $H$, $W$ are preserved), adds its bias, then applies ReLU.
2. If `pool`: max pool with window 2 and stride 2; an odd last row or column is dropped (output $\lfloor H/2\rfloor \times \lfloor W/2\rfloor$).

## Output
- `y`: the block's output.
- `n_params`: total number of weights and biases in `convs`, an `int`.
- `receptive_field`: side length (an `int`) of the input region that one output pixel of the block depends on. With $L$ convs it is $2L + 1$ before pooling; a $2\times2$/stride-2 pool adds 1 more (it covers two adjacent conv outputs).
"""

STARTER = '''import numpy as np

def vgg_block(x, convs, pool=True):
    """convs: list of (W (F, C, 3, 3), b (F,)). Return (y, n_params, receptive_field)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _conv3x3_same(x, W, b):
    N, C, H, Wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    y = np.zeros((N, W.shape[0], H, Wd))
    for u in range(3):
        for v in range(3):
            y += np.einsum('nchw,fc->nfhw', xp[:, :, u:u + H, v:v + Wd], W[:, :, u, v])
    return y + b[None, :, None, None]

def vgg_block(x, convs, pool=True):
    y = np.asarray(x, dtype=float)
    n_params = 0
    for W, b in convs:
        W, b = np.asarray(W, dtype=float), np.asarray(b, dtype=float)
        y = np.maximum(_conv3x3_same(y, W, b), 0.0)
        n_params += W.size + b.size
    rf = 2 * len(convs) + 1                        # each 3x3 conv grows the field by 2
    if pool:
        N, C, H, Wd = y.shape
        y = y[:, :, :H // 2 * 2, :Wd // 2 * 2].reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))
        rf += 1                                    # two neighbouring conv outputs, 1 pixel apart
    return y, int(n_params), int(rf)
'''

HINTS = [
    'Write one helper for a 3x3 "same" convolution: pad by 1, then for each of the 9 taps add `einsum("nchw,fc->nfhw", xp[:, :, u:u+H, v:v+W], W[:, :, u, v])`.',
    'Apply conv, bias and ReLU for every pair in `convs`, counting `W.size + b.size` as you go.',
    'For the 2x2/stride-2 pool, crop to even sizes and use `reshape(N, C, H//2, 2, W//2, 2).max(axis=(3, 5))`. The receptive field is $2L + 1$, plus 1 if you pooled.',
]

_r = np.random.default_rng(41)
def _convs(chans):
    return [(_r.normal(size=(o, i, 3, 3)) / 3, _r.normal(size=o) * 0.1) for i, o in zip(chans, chans[1:])]
TESTS = [
    (np.arange(16.0).reshape(1, 1, 4, 4), [(np.array([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]), np.array([-5.0]))]),
    (np.ones((1, 1, 3, 3)), [(np.ones((1, 1, 3, 3)), np.zeros(1))], False),
    (_r.normal(size=(2, 3, 8, 8)), _convs([3, 4, 4])),
    (_r.normal(size=(1, 2, 7, 5)), _convs([2, 3, 5, 2])),
    (_r.normal(size=(2, 2, 6, 6)), _convs([2, 6]), False),
    (_r.normal(size=(1, 3, 2, 3)), _convs([3, 2, 2])),
    (_r.normal(size=(1, 4, 5, 5)), []),
]

EXPLANATION = r"""
## Why stacks of 3x3
Two 3x3 convs see a 5x5 region and three see 7x7, with $2 \cdot 9C^2 = 18C^2$ or $27C^2$ weights versus $25C^2$ or $49C^2$ for a single large kernel, and with a ReLU between every pair. VGG's insight was that depth with tiny kernels beats shallow wide kernels.

## Receptive field
Each stride-1 $k\times k$ conv adds $k - 1$ to the receptive field, so $L$ stacked 3x3 convs give $2L + 1$. The pool's output pixel covers two adjacent conv outputs one pixel apart, adding 1, and every later layer's growth is multiplied by the accumulated stride (2 after this block).

## Complexity
Each conv costs $9\,C_\text{in} C_\text{out} H W$ multiply-adds per sample. VGG-16 has about 138M parameters, most of them in the first fully connected layer, not in these blocks.

## Common bugs
- Forgetting the ReLU after the **last** conv of the block, or applying it after the pool (harmless for max pool, but not what the spec says).
- Pooling odd sizes with a reshape that doesn't crop first (test 4 is 7x5).
- Counting parameters without the biases.

## Interview follow-ups
- Where do most of VGG-16's FLOPs go, and where do most of its parameters go? Why are those different places?
- Why did ResNet need skip connections to go deeper than VGG's 19 layers?
"""
