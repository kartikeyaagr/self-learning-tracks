import numpy as np

META = dict(title='Build a VGG network from its configuration list', tags=['Convolutional Networks', 'Computer Vision'],
            diff='Medium', runtime='numpy', lessons=['dlcore:03.03'],
            related=['vgg-convolution-block', 'max-pool-2d', '2d-convolution', 'dropout'])

STATEMENT = r"""
The VGG paper describes its networks as lists: numbers are 3×3 convolution layers with that many output channels, and `'M'` is a 2×2 max pool. VGG-16's feature extractor is `[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']`. Build a network from such a list and report what it costs.

Write `vgg_forward(x, cfg, convs, fcs)` returning `(logits, n_params)`.

## Rules
- `x`: `(B, C, H, W)`. Walk `cfg` in order. For a number, take the next `(W, b)` from `convs` (`W` `(O, C, 3, 3)`, `b` `(O,)`): 3×3 convolution, stride 1, zero padding 1, plus bias, then ReLU. For `'M'`: 2×2 max pooling, stride 2 (drop a trailing odd row or column).
- Then flatten each sample in `(C, H, W)` order and apply the classifier `fcs`, a list of `(W, b)` with `W` `(in, out)`: ReLU after every layer except the last. (Dropout sits between the hidden layers during training; this is inference, so it is the identity.)
- `n_params`: total number of weights and biases in `convs` and `fcs` (an int).
"""

STARTER = '''import numpy as np

def vgg_forward(x, cfg, convs, fcs):
    """Return (logits (B, K), n_params)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _conv3(z, W, b):
    B, C, H, Wd = z.shape
    zp = np.pad(z, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((B, W.shape[0], H, Wd)) + b[None, :, None, None]
    for i in range(3):
        for j in range(3):
            out += np.einsum('bchw,oc->bohw', zp[:, :, i:i + H, j:j + Wd], W[:, :, i, j])
    return out

def _pool(z):
    B, C, H, W = z.shape
    z = z[:, :, : H // 2 * 2, : W // 2 * 2]
    return z.reshape(B, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))

def vgg_forward(x, cfg, convs, fcs):
    z = np.asarray(x, dtype=float)
    it = iter(convs)
    for item in cfg:
        if item == 'M':
            z = _pool(z)
        else:
            W, b = next(it)
            z = np.maximum(_conv3(z, np.asarray(W), np.asarray(b)), 0.0)
    h = z.reshape(z.shape[0], -1)
    for i, (W, b) in enumerate(fcs):
        h = h @ W + b
        if i < len(fcs) - 1:
            h = np.maximum(h, 0.0)
    n = sum(np.size(W) + np.size(b) for W, b in list(convs) + list(fcs))
    return h, int(n)
'''

HINTS = [
    'Keep an iterator over `convs` and advance it each time `cfg` has a number; `\'M\'` consumes no weights.',
    'With padding 1 a 3×3 convolution keeps $H \\times W$, so only the pools change the spatial size. Crop to even sizes before reshaping for the 2×2 pool.',
    'Flatten with `z.reshape(B, -1)`, which is `(C, H, W)` order for a contiguous `(B, C, H, W)` array.',
]

_r = np.random.default_rng(1409)
def _make(cfg, C, H, W, hidden, K):
    convs, c, h, w = [], C, H, W
    for item in cfg:
        if item == 'M':
            h, w = h // 2, w // 2
        else:
            convs.append((_r.normal(size=(item, c, 3, 3)) * np.sqrt(2 / (9 * c)), 0.1 * _r.normal(size=item)))
            c = item
    dims = [c * h * w] + hidden + [K]
    fcs = [(_r.normal(size=(a, b)) * np.sqrt(2 / a), 0.1 * _r.normal(size=b)) for a, b in zip(dims, dims[1:])]
    return convs, fcs
_c1 = [2, 'M']
_c2 = [3, 3, 'M', 4, 'M']
_c3 = [2, 'M', 4, 4, 'M', 4, 'M']
TESTS = [
    (_r.normal(size=(1, 1, 4, 4)), _c1, *_make(_c1, 1, 4, 4, [], 3)),
    (_r.normal(size=(2, 1, 4, 4)), _c1, *_make(_c1, 1, 4, 4, [5], 3)),
    (_r.normal(size=(2, 3, 8, 8)), _c2, *_make(_c2, 3, 8, 8, [6, 6], 4)),
    (_r.normal(size=(1, 3, 9, 10)), _c2, *_make(_c2, 3, 9, 10, [6, 6], 4)),
    (_r.normal(size=(2, 2, 16, 16)), _c3, *_make(_c3, 2, 16, 16, [8, 8], 5)),
    (np.zeros((1, 2, 16, 16)), _c3, *_make(_c3, 2, 16, 16, [8], 5)),
]

EXPLANATION = r"""
## VGG's argument
Two stacked 3×3 convolutions see a 5×5 region and three see 7×7, with fewer parameters than one large kernel ($2 \cdot 9C^2 = 18C^2$ vs $25C^2$) and an extra non-linearity in between. VGG built every network from this one layer type and showed that depth (16 to 19 weight layers) was what improved accuracy.

## Where the parameters are
For VGG-16 at 224×224 the convolutions hold about 15M parameters, while the first fully connected layer alone ($7\cdot7\cdot512 \times 4096$) holds about 103M of the 138M total. The convolutions dominate FLOPs; the classifier dominates parameters. Later networks replaced the flatten-and-FC head with global average pooling.

## Complexity
A 3×3 layer at resolution $H\times W$ costs $9\,C_{in}C_{out}HW$ multiply-adds; channels double whenever the resolution halves, so the cost per stage stays roughly constant.

## Common bugs
- Forgetting the conv bias or the ReLU after each convolution.
- Applying ReLU after the final linear layer.
- Flattening in `(H, W, C)` order when the classifier weights expect `(C, H, W)`.

## Interview follow-ups
- Compute VGG-16's parameter count from its configuration list.
- Why did VGG need careful initialisation (or pre-training shallower nets first) before batch norm existed?
"""
