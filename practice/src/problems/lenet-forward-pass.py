import numpy as np

META = dict(title='A LeNet-style CNN, end to end', tags=['Convolutional Networks', 'Computer Vision'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:04.02'], related=['2d-convolution', 'pooling-layers', 'multi-layer-perceptron-forward-pass'])

STATEMENT = r"""
Assemble a complete small convolutional classifier in the style of LeNet-5:

```
conv1 -> tanh -> avgpool 2x2 -> conv2 -> tanh -> avgpool 2x2 -> flatten -> fc1 -> tanh -> fc2 -> tanh -> fc3
```

Write `lenet_forward(x, params)` that returns `(logits, shapes)`.

## Parameters
`params` is a dict of arrays:

| key | shape | meaning |
|---|---|---|
| `'c1w'`, `'c1b'` | `(F1, C, k1, k1)`, `(F1,)` | first conv |
| `'c2w'`, `'c2b'` | `(F2, F1, k2, k2)`, `(F2,)` | second conv |
| `'f1w'`, `'f1b'` | `(D, U1)`, `(U1,)` | fully connected, rows are inputs |
| `'f2w'`, `'f2b'` | `(U1, U2)`, `(U2,)` | |
| `'f3w'`, `'f3b'` | `(U2, K)`, `(K,)` | output layer, no activation |

## Rules
- `x`: float `(N, C, H, W)`. Both convs use stride 1 and **no padding** ("valid"), with cross-correlation (no kernel flip), plus their bias.
- Average pools use a 2x2 window with stride 2; an odd last row or column is dropped.
- Flatten each sample's `(F2, h, w)` map in C order (channel-major, then rows, then columns) to length $D = F2 \cdot h \cdot w$.
- `shapes`: a list of 7 tuples of ints, the shape after each of: conv1, pool1, conv2, pool2, flatten, fc1, fc2 (activation shapes equal their layer's).
- `logits`: float `(N, K)`.
"""

STARTER = '''import numpy as np

def lenet_forward(x, params):
    """Return (logits (N, K), shapes: list of 7 tuples)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _conv_valid(x, W, b):
    N, C, H, Wd = x.shape
    F, _, k, _ = W.shape
    Ho, Wo = H - k + 1, Wd - k + 1
    y = np.zeros((N, F, Ho, Wo))
    for u in range(k):
        for v in range(k):
            y += np.einsum('nchw,fc->nfhw', x[:, :, u:u + Ho, v:v + Wo], W[:, :, u, v])
    return y + b[None, :, None, None]

def _avgpool2(x):
    N, C, H, W = x.shape
    x = x[:, :, :H // 2 * 2, :W // 2 * 2]
    return x.reshape(N, C, H // 2, 2, W // 2, 2).mean(axis=(3, 5))

def lenet_forward(x, params):
    P = {k: np.asarray(v, dtype=float) for k, v in params.items()}
    h = np.asarray(x, dtype=float)
    shapes = []
    def log(a):
        shapes.append(tuple(int(d) for d in a.shape))
        return a
    h = log(np.tanh(_conv_valid(h, P['c1w'], P['c1b'])))
    h = log(_avgpool2(h))
    h = log(np.tanh(_conv_valid(h, P['c2w'], P['c2b'])))
    h = log(_avgpool2(h))
    h = log(h.reshape(h.shape[0], -1))                     # C-order flatten: channel, row, column
    h = log(np.tanh(h @ P['f1w'] + P['f1b']))
    h = log(np.tanh(h @ P['f2w'] + P['f2b']))
    return h @ P['f3w'] + P['f3b'], shapes
'''

HINTS = [
    'Write three helpers first: a valid conv (loop over kernel taps, `einsum` over channels), a 2x2 average pool (crop to even, reshape, mean), and the dense layer `h @ W + b`.',
    'Shapes for a 32x32 input with 5x5 kernels: 28 -> 14 -> 10 -> 5, so $D = F_2 \\cdot 25$. In general each valid conv subtracts $k - 1$ and each pool halves (rounding down).',
    '`h.reshape(N, -1)` on a C-contiguous `(N, F2, h, w)` array gives exactly the channel-major flatten. Record `h.shape` after every stage except the final logits.',
]

_r = np.random.default_rng(47)
def _params(C, F1, k1, F2, k2, D, U1, U2, K):
    g = lambda *s: _r.normal(size=s) / np.sqrt(np.prod(s[1:]) if len(s) > 2 else s[0])
    return {'c1w': g(F1, C, k1, k1), 'c1b': _r.normal(size=F1) * 0.1, 'c2w': g(F2, F1, k2, k2), 'c2b': _r.normal(size=F2) * 0.1,
            'f1w': g(D, U1), 'f1b': _r.normal(size=U1) * 0.1, 'f2w': g(U1, U2), 'f2b': _r.normal(size=U2) * 0.1,
            'f3w': g(U2, K), 'f3b': _r.normal(size=K) * 0.1}
_tiny = {'c1w': np.ones((1, 1, 2, 2)) * 0.25, 'c1b': np.zeros(1), 'c2w': np.ones((1, 1, 1, 1)), 'c2b': np.zeros(1),
         'f1w': np.ones((1, 2)), 'f1b': np.zeros(2), 'f2w': np.eye(2), 'f2b': np.zeros(2),
         'f3w': np.array([[1.0, -1.0], [2.0, 0.0]]), 'f3b': np.array([0.0, 0.5])}
TESTS = [
    (np.ones((1, 1, 5, 5)), _tiny),
    (np.zeros((2, 1, 5, 5)), _tiny),
    (_r.normal(size=(2, 1, 32, 32)), _params(1, 6, 5, 16, 5, 400, 120, 84, 10)),
    (_r.normal(size=(1, 3, 16, 16)), _params(3, 4, 3, 5, 3, 20, 8, 6, 3)),
    (_r.normal(size=(3, 2, 15, 13)), _params(2, 3, 4, 4, 2, 4 * 2 * 2, 5, 5, 4)),
    (_r.normal(size=(1, 1, 28, 28)), _params(1, 6, 5, 16, 5, 256, 32, 16, 10)),
]

EXPLANATION = r"""
## Tracking shapes is the whole game
Valid convs shrink by $k - 1$, pools halve with a floor. For the classic 32x32 input: $32 \xrightarrow{5\times5} 28 \xrightarrow{\text{pool}} 14 \xrightarrow{5\times5} 10 \xrightarrow{\text{pool}} 5$, so 16 channels give $D = 400$. For MNIST's 28x28 the same layers give $4 \times 4$ and $D = 256$ (test 6), which is why the first FC layer's size depends on the input resolution and global pooling was later introduced to remove that coupling.

## Flatten order matters
The FC weights were trained against a particular flatten order. NumPy's default C order on `(N, F2, h, w)` matches PyTorch's `x.view(N, -1)`. TensorFlow's NHWC layout flattens as rows, columns, channels; porting weights between the two needs a permutation of `f1w`'s rows.

## Complexity
Dominated by conv2 for the classic sizes: $N \cdot 16 \cdot 6 \cdot 25 \cdot 100 = 240{,}000 N$ multiply-adds. The whole network has about 61k parameters, most in fc1 (48k).

## Common bugs
- Padding the convs, or using max instead of average pooling.
- Flattening `(N, h, w, F2)` after a transpose so features come out in the wrong order.
- Applying tanh to the logits.

## Interview follow-ups
- Count the parameters of the classic LeNet-5 layout above layer by layer.
- Replace the flatten and fc1 with global average pooling: what changes in parameters and in input-size flexibility?
"""
