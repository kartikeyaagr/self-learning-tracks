import numpy as np

META = dict(title='A tiny ResNet, end to end', tags=['Convolutional Networks', 'Computer Vision'], diff='Hard',
            runtime='numpy', lessons=['dlcore:03.04'],
            related=['resnet-residual-block', 'batch-normalization', 'max-pool-2d', 'adaptive-average-pool-2d'])

STATEMENT = r"""
Assemble a small residual network in inference mode: a stem, stages of basic residual blocks, global average pooling and a linear classifier.

Write `resnet_forward(x, p)` returning `(B, K)` logits for images `x` `(B, C, H, W)`.

## Building blocks
- `conv(z, W, stride, pad)`: cross-correlation, weights `(O, C, k, k)`, no bias, zero padding.
- `bn(z, s)`: inference-mode batch norm per channel with `s = (gamma, beta, mean, var)`, $\epsilon = 10^{-5}$: $\gamma (z - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$.
- **Stem**: `relu(bn(conv(x, stem_w, stride=1, pad=1), stem_bn))`, then 3×3 max pooling with stride 2 and padding 1 (padding counts as $-\infty$).
- **Basic block** with stride $s$ (a dict):
  $\text{out} = \mathrm{relu}\big(\mathrm{bn}_2(\mathrm{conv}(\mathrm{relu}(\mathrm{bn}_1(\mathrm{conv}(z, W_1, s, 1))), W_2, 1, 1)) + \text{short}(z)\big)$
  where `short(z) = z` if the block has no key `'proj'`, otherwise `bn_p(conv(z, proj, s, 0))` with a 1×1 kernel.
  Keys: `W1`, `bn1`, `W2`, `bn2`, `stride`, optional `proj`, `bn_p`.
- **Head**: global average over $H, W$, then `@ fc_w + fc_b` with `fc_w` `(C_last, K)`.

`p` has `stem_w`, `stem_bn`, `blocks` (a flat list, applied in order), `fc_w`, `fc_b`. Output sizes follow $\lfloor (H + 2\,\text{pad} - k)/s \rfloor + 1$.
"""

STARTER = '''import numpy as np

def resnet_forward(x, p):
    """Return (B, K) logits."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _conv(z, W, stride, pad):
    B, C, H, Wd = z.shape
    O, _, k, _ = W.shape
    zp = np.pad(z, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (Wd + 2 * pad - k) // stride + 1
    out = np.zeros((B, O, Ho, Wo))
    for i in range(k):
        for j in range(k):
            patch = zp[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride]      # (B, C, Ho, Wo)
            out += np.einsum('bchw,oc->bohw', patch, W[:, :, i, j])
    return out

def _bn(z, s, eps=1e-5):
    g, b, m, v = (np.asarray(a, dtype=float)[None, :, None, None] for a in s)
    return g * (z - m) / np.sqrt(v + eps) + b

def _maxpool(z, k=3, s=2, pad=1):
    B, C, H, W = z.shape
    zp = np.pad(z, ((0, 0), (0, 0), (pad, pad), (pad, pad)), constant_values=-np.inf)
    Ho, Wo = (H + 2 * pad - k) // s + 1, (W + 2 * pad - k) // s + 1
    out = np.full((B, C, Ho, Wo), -np.inf)
    for i in range(k):
        for j in range(k):
            out = np.maximum(out, zp[:, :, i:i + s * Ho:s, j:j + s * Wo:s])
    return out

relu = lambda z: np.maximum(z, 0.0)

def resnet_forward(x, p):
    z = np.asarray(x, dtype=float)
    z = _maxpool(relu(_bn(_conv(z, p['stem_w'], 1, 1), p['stem_bn'])))
    for blk in p['blocks']:
        s = blk['stride']
        h = relu(_bn(_conv(z, blk['W1'], s, 1), blk['bn1']))
        h = _bn(_conv(h, blk['W2'], 1, 1), blk['bn2'])
        short = _bn(_conv(z, blk['proj'], s, 0), blk['bn_p']) if 'proj' in blk else z
        z = relu(h + short)
    return z.mean(axis=(2, 3)) @ p['fc_w'] + p['fc_b']
'''

HINTS = [
    'A convolution can be written as a sum over the $k \\times k$ kernel offsets: slice the padded input with the stride for each offset and contract over input channels. No im2col needed.',
    'The shortcut is identity only when shapes match. A block that changes stride or channel count has `proj`: a strided 1×1 convolution followed by its own batch norm.',
    'The final ReLU comes **after** adding the shortcut. Max-pool padding must be $-\\infty$, not 0.',
]

_r = np.random.default_rng(1512)
def _bnp(C):
    return (1 + 0.1 * _r.normal(size=C), 0.1 * _r.normal(size=C), 0.1 * _r.normal(size=C), 0.5 + _r.random(C))
def _w(O, C, k):
    return _r.normal(size=(O, C, k, k)) * np.sqrt(2.0 / (C * k * k))
def _block(Cin, Cout, s):
    b = dict(W1=_w(Cout, Cin, 3), bn1=_bnp(Cout), W2=_w(Cout, Cout, 3), bn2=_bnp(Cout), stride=s)
    if s != 1 or Cin != Cout:
        b.update(proj=_w(Cout, Cin, 1), bn_p=_bnp(Cout))
    return b
def _model(C, widths, K):
    blocks, cin = [], widths[0]
    for i, w in enumerate(widths):
        blocks.append(_block(cin, w, 1 if i == 0 else 2))
        blocks.append(_block(w, w, 1))
        cin = w
    return dict(stem_w=_w(widths[0], C, 3), stem_bn=_bnp(widths[0]), blocks=blocks,
                fc_w=_r.normal(size=(widths[-1], K)) / np.sqrt(widths[-1]), fc_b=0.1 * _r.normal(size=K))
_m1 = _model(1, [2], 3)
_m2 = _model(3, [4, 6], 5)
_m3 = dict(_m2, blocks=_m2['blocks'][:3])
TESTS = [
    (_r.normal(size=(1, 1, 4, 4)), _m1),
    (_r.normal(size=(2, 1, 6, 6)), _m1),
    (_r.normal(size=(2, 3, 8, 8)), _m2),
    (_r.normal(size=(1, 3, 9, 7)), _m2),
    (_r.normal(size=(2, 3, 8, 8)), dict(_m3, fc_w=_r.normal(size=(6, 5)))),
    (np.zeros((1, 3, 8, 8)), _m2),
]

EXPLANATION = r"""
## Why residual connections
A deeper plain network should be able to match a shallower one by learning identity layers, yet in practice deeper plain nets trained worse (higher *training* error). Writing each block as $x + F(x)$ makes the identity the default: $F = 0$ is easy to reach, and the gradient has a direct path through every skip connection. That is what let ResNet train 152 layers.

## Shapes through the network
The stem keeps the resolution, the max pool halves it, and the first block of each later stage halves it again with stride 2 while widening the channels; its shortcut must do the same, hence the strided 1×1 projection. Global average pooling removes the spatial dimensions, so the classifier works at any input size (test 4 uses 9×7).

## Inference-mode BN
At inference, BN is a fixed per-channel affine map, so it can be folded into the preceding convolution's weights and bias. In training it uses batch statistics.

## Complexity
Each 3×3 conv costs $O(B\,C_{in} C_{out}\,9\,H_oW_o)$; the projection shortcuts are cheap by comparison.

## Common bugs
- ReLU before the addition, or no ReLU after it.
- Identity shortcut when the stride or width changes (shape error) or a projection shortcut without its BN.
- Zero padding in max pooling, which matters after a ReLU-free path or with negative inputs.

## Interview follow-ups
- Compare the basic block with the bottleneck block (1×1, 3×3, 1×1) in parameters and FLOPs at 256 channels.
- What changes in "pre-activation" ResNets, and why did that help at 1000 layers?
"""
