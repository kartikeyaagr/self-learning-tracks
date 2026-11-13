import numpy as np

META = dict(title='ResNet basic block with a projection shortcut', tags=['Convolutional Networks', 'Computer Vision'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:04.02'],
            related=['2d-convolution', 'batch-normalization', 'vgg-convolution-block'])

STATEMENT = r"""
A residual block learns a correction $F(x)$ on top of a shortcut, $y = \mathrm{ReLU}(F(x) + \mathrm{shortcut}(x))$. Implement ResNet's **basic block** at inference time (BatchNorm uses stored statistics):

$$F(x) = \mathrm{BN}_2\big(\mathrm{conv}_2(\mathrm{ReLU}(\mathrm{BN}_1(\mathrm{conv}_1(x))))\big)$$

Write `basic_block(x, p, stride=1)`.

## Layers
- `conv1`: 3x3, stride `stride`, zero padding 1, **no bias**, weight `p['w1']` of shape `(F, C, 3, 3)`.
- `conv2`: 3x3, stride 1, zero padding 1, no bias, weight `p['w2']` `(F, F, 3, 3)`.
- `BN` in eval mode: $\gamma\,(z - \mu)/\sqrt{\sigma^2 + 10^{-5}} + \beta$ per channel. `p['bn1']`, `p['bn2']` are tuples `(gamma, beta, mean, var)` of `(F,)` arrays.
- Shortcut: if `p['proj']` is `None`, the identity (then $F = C$ and `stride == 1`). Otherwise `p['proj']` is a tuple `(w, bn)`: a 1x1 conv with weight `w` of shape `(F, C, 1, 1)`, stride `stride`, no padding, no bias, followed by BN with the tuple `bn`.

## Output
Float `(N, F, H_out, W_out)` with $H_\text{out} = \lfloor (H - 1)/s \rfloor + 1$ (the same for both paths).
"""

STARTER = '''import numpy as np

def basic_block(x, p, stride=1):
    """x: (N, C, H, W); p: dict with w1, bn1, w2, bn2, proj. Return (N, F, H_out, W_out)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _conv(x, W, stride, pad):
    N, C, H, Wd = x.shape
    F, _, k, _ = W.shape
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (Wd + 2 * pad - k) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    y = np.zeros((N, F, Ho, Wo))
    for u in range(k):
        for v in range(k):
            y += np.einsum('nchw,fc->nfhw', xp[:, :, u:u + stride * (Ho - 1) + 1:stride, v:v + stride * (Wo - 1) + 1:stride], W[:, :, u, v])
    return y

def _bn(z, bn, eps=1e-5):
    g, b, m, v = (np.asarray(a, dtype=float)[None, :, None, None] for a in bn)
    return g * (z - m) / np.sqrt(v + eps) + b

def basic_block(x, p, stride=1):
    x = np.asarray(x, dtype=float)
    h = np.maximum(_bn(_conv(x, np.asarray(p['w1'], dtype=float), stride, 1), p['bn1']), 0.0)
    h = _bn(_conv(h, np.asarray(p['w2'], dtype=float), 1, 1), p['bn2'])
    if p['proj'] is None:
        sc = x
    else:
        w, bn = p['proj']
        sc = _bn(_conv(x, np.asarray(w, dtype=float), stride, 0), bn)   # match channels and resolution
    return np.maximum(h + sc, 0.0)                                       # ReLU after the addition
'''

HINTS = [
    'You need one conv helper with stride and padding (no bias) and one eval-mode BN helper that reshapes its four `(F,)` arrays to `(1, F, 1, 1)`.',
    'Residual path: conv1 (stride $s$) -> BN -> ReLU -> conv2 (stride 1) -> BN. Note there is **no** ReLU after the second BN yet.',
    'Shortcut: `x` itself, or a 1x1 conv with the same stride followed by its own BN. Add the two paths, and only then apply the final ReLU.',
]

_r = np.random.default_rng(53)
def _bnp(F):
    return (_r.uniform(0.5, 1.5, F), _r.normal(size=F) * 0.1, _r.normal(size=F) * 0.1, _r.uniform(0.5, 2.0, F))
def _block(C, F, proj):
    p = {'w1': _r.normal(size=(F, C, 3, 3)) / np.sqrt(9 * C), 'bn1': _bnp(F),
         'w2': _r.normal(size=(F, F, 3, 3)) / np.sqrt(9 * F), 'bn2': _bnp(F), 'proj': None}
    if proj:
        p['proj'] = (_r.normal(size=(F, C, 1, 1)) / np.sqrt(C), _bnp(F))
    return p
_id_bn = (np.ones(1), np.zeros(1), np.zeros(1), np.ones(1) - 1e-5)
TESTS = [
    (np.arange(9.0).reshape(1, 1, 3, 3) - 4, {'w1': np.zeros((1, 1, 3, 3)), 'bn1': _id_bn, 'w2': np.zeros((1, 1, 3, 3)), 'bn2': _id_bn, 'proj': None}),
    (np.ones((1, 1, 2, 2)), {'w1': np.ones((1, 1, 3, 3)), 'bn1': _id_bn, 'w2': np.ones((1, 1, 3, 3)), 'bn2': _id_bn, 'proj': None}),
    (_r.normal(size=(2, 3, 5, 5)), _block(3, 3, False)),
    (_r.normal(size=(2, 3, 6, 6)), _block(3, 4, True), 2),
    (_r.normal(size=(1, 2, 7, 5)), _block(2, 4, True), 2),
    (_r.normal(size=(1, 4, 4, 4)), _block(4, 4, True), 1),
    (_r.normal(size=(3, 2, 3, 3)), _block(2, 2, False)),
]

EXPLANATION = r"""
## Why the shortcut helps
If the best thing a block can do is nothing, a plain stack must learn an identity through two convs and ReLUs, which is hard. A residual block only has to drive $F(x)$ to zero. Gradients also get a direct path: $\partial y/\partial x = I + \partial F/\partial x$ (before the final ReLU), so they don't vanish through depth. This is what let ResNets train 100+ layers where VGG-style stacks degraded.

## Projection shortcuts
When the block halves the resolution (stride 2) or changes the channel count, the identity no longer has the right shape. A strided 1x1 conv with BN matches both. Test 6 uses a projection with stride 1, which is legal but only needed when $C \ne F$.

## Where the ReLU goes
The original block applies ReLU **after** the addition. The "pre-activation" ResNet (v2) moves BN and ReLU before each conv and leaves the sum untouched, so the shortcut is a pure identity across the whole network.

## Complexity
Two 3x3 convs: $9FC + 9F^2$ weights, plus $FC$ for a projection.

## Common bugs
- Applying the stride to conv2 instead of conv1, or forgetting it on the projection.
- ReLU on the residual branch before the add (after BN2).
- Giving the convs a bias even though BN follows (redundant, and the spec says none).

## Interview follow-ups
- What is the bottleneck block (1x1 reduce, 3x3, 1x1 expand) and why is it used in ResNet-50 and deeper?
- Why do some recipes initialise $\gamma$ of the last BN in each block to zero?
"""
