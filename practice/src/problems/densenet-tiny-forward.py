import numpy as np

META = dict(title='DenseNet: dense blocks and transition layers', tags=['Convolutional Networks', 'Computer Vision'], diff='Hard',
            runtime='numpy', lessons=['dlcore:03.05'],
            related=['multichannel-2d-convolution', 'batch-normalization', 'average-pool-2d', 'resnet-tiny-forward'])

STATEMENT = r"""
In a dense block every layer receives the concatenation of **all** earlier feature maps in the block and adds $k$ new channels (the growth rate). Transition layers between blocks compress the channels and halve the resolution. Build a small DenseNet-BC in inference mode.

Write `densenet_forward(x, p)` returning `(B, K)` logits.

## Pieces
- `bn(z, s)`: inference batch norm with `s = (gamma, beta, mean, var)`, $\epsilon = 10^{-5}$. `conv(z, W, pad)`: stride 1, zero padding, no bias.
- **Stem**: `conv(x, stem_w, pad=1)`.
- **Dense layer** (bottleneck "BC" form), dict `bn1, W1 (4k, C_in, 1, 1), bn2, W2 (k, 4k, 3, 3)`:
  $\text{new} = \mathrm{conv}(\mathrm{relu}(\mathrm{bn}_2(\mathrm{conv}(\mathrm{relu}(\mathrm{bn}_1(z)), W_1, 0))), W_2, 1)$, then $z \leftarrow [z;\ \text{new}]$ along channels (old channels first).
- **Transition**, dict `bn, W (C_out, C_in, 1, 1)`: `conv(relu(bn(z)), W, 0)`, then 2×2 average pooling with stride 2 (sizes are even).
- **Head**: `relu(bn(z, final_bn))`, global average pool, `@ fc_w + fc_b`.

`p['blocks']` is a list of dense blocks (each a list of dense-layer dicts) and `p['transitions']` has one fewer entry: block 0, transition 0, block 1, ..., last block, head.
"""

STARTER = '''import numpy as np

def densenet_forward(x, p):
    """Return (B, K) logits."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _conv(z, W, pad):
    B, C, H, Wd = z.shape
    O, _, k, _ = W.shape
    zp = np.pad(z, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((B, O, H + 2 * pad - k + 1, Wd + 2 * pad - k + 1))
    for i in range(k):
        for j in range(k):
            out += np.einsum('bchw,oc->bohw', zp[:, :, i:i + out.shape[2], j:j + out.shape[3]], W[:, :, i, j])
    return out

def _bn(z, s, eps=1e-5):
    g, b, m, v = (np.asarray(a, dtype=float)[None, :, None, None] for a in s)
    return g * (z - m) / np.sqrt(v + eps) + b

relu = lambda z: np.maximum(z, 0.0)

def densenet_forward(x, p):
    z = _conv(np.asarray(x, dtype=float), p['stem_w'], 1)
    for bi, block in enumerate(p['blocks']):
        for L in block:
            h = _conv(relu(_bn(z, L['bn1'])), L['W1'], 0)
            new = _conv(relu(_bn(h, L['bn2'])), L['W2'], 1)
            z = np.concatenate([z, new], axis=1)
        if bi < len(p['transitions']):
            t = p['transitions'][bi]
            z = _conv(relu(_bn(z, t['bn'])), t['W'], 0)
            B, C, H, W = z.shape
            z = z.reshape(B, C, H // 2, 2, W // 2, 2).mean(axis=(3, 5))
    z = relu(_bn(z, p['final_bn']))
    return z.mean(axis=(2, 3)) @ p['fc_w'] + p['fc_b']
'''

HINTS = [
    'Inside a block keep one growing array: after each layer, `z = np.concatenate([z, new], axis=1)`. A layer\'s input width is the stem width plus $k$ times the number of earlier layers in that block.',
    'DenseNet uses pre-activation ordering: BN, then ReLU, then convolution, everywhere (including the transition).',
    'Average pooling 2×2 with stride 2 is a reshape to `(B, C, H/2, 2, W/2, 2)` and a mean over axes 3 and 5.',
]

_r = np.random.default_rng(1608)
def _bnp(C):
    return (1 + 0.1 * _r.normal(size=C), 0.1 * _r.normal(size=C), 0.1 * _r.normal(size=C), 0.5 + _r.random(C))
def _w(O, C, k):
    return _r.normal(size=(O, C, k, k)) * np.sqrt(2.0 / (C * k * k))
def _model(Cimg, C0, k, layers, theta, K):
    blocks, trans, c = [], [], C0
    for bi, n in enumerate(layers):
        blk = []
        for _ in range(n):
            blk.append(dict(bn1=_bnp(c), W1=_w(4 * k, c, 1), bn2=_bnp(4 * k), W2=_w(k, 4 * k, 3)))
            c += k
        blocks.append(blk)
        if bi < len(layers) - 1:
            co = int(theta * c)
            trans.append(dict(bn=_bnp(c), W=_w(co, c, 1)))
            c = co
    return dict(stem_w=_w(C0, Cimg, 3), blocks=blocks, transitions=trans, final_bn=_bnp(c),
                fc_w=_r.normal(size=(c, K)) / np.sqrt(c), fc_b=0.1 * _r.normal(size=K))
_m1 = _model(1, 2, 2, [2], 0.5, 3)
_m2 = _model(3, 4, 2, [2, 2], 0.5, 4)
_m3 = _model(2, 4, 3, [3, 1, 2], 0.5, 5)
TESTS = [
    (_r.normal(size=(1, 1, 4, 4)), _m1),
    (_r.normal(size=(2, 1, 5, 3)), _m1),
    (_r.normal(size=(2, 3, 4, 4)), _m2),
    (_r.normal(size=(1, 3, 6, 8)), _m2),
    (_r.normal(size=(2, 2, 8, 8)), _m3),
    (np.ones((1, 2, 4, 4)), _m3),
]

EXPLANATION = r"""
## Concatenation instead of addition
ResNet adds a block's output to its input; DenseNet concatenates. Every layer can read every earlier feature map in its block directly, so features are reused instead of relearned, and each layer can be narrow (growth rate $k = 12$ to 32). A block with $L$ layers has $L(L+1)/2$ direct connections.

## Why bottlenecks and compression
Without them the input width grows linearly with depth and the 3×3 convolutions get expensive. The 1×1 bottleneck reduces any input to $4k$ channels before the 3×3, and transitions multiply the channel count by $\theta$ (0.5 in DenseNet-BC) before downsampling.

## Channel count bookkeeping
After a block with $n$ layers starting from $c$ channels there are $c + nk$; after the transition $\lfloor\theta(c + nk)\rfloor$. Getting this right is most of the work of building the weights.

## Complexity and memory
FLOPs are modest, but naive implementations are memory-hungry: each concatenation copies all previous maps. Memory-efficient implementations share storage for the concatenated tensor and recompute BN-ReLU in the backward pass.

## Common bugs
- Post-activation ordering (conv, BN, ReLU) copied from ResNet.
- Concatenating only the previous layer's output instead of the running stack.
- Max pooling in the transition (DenseNet uses average pooling there).

## Interview follow-ups
- Why does DenseNet have fewer parameters than a ResNet of similar accuracy, but not lower memory use?
- Count the input channels of the last 1×1 conv in a 12-layer block with $c = 24$, $k = 12$, and the channels leaving the block.
"""
