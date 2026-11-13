import numpy as np

META = dict(title='Inception module with 1x1 reductions', tags=['Convolutional Networks', 'Computer Vision'], diff='Hard',
            runtime='numpy', lessons=['air-fanout-v1:04.09'],
            related=['2d-convolution', 'pooling-layers', 'depthwise-separable-convolution'])

STATEMENT = r"""
An Inception module runs several filter sizes side by side on the same input and concatenates the results along the channel axis. Cheap $1\times1$ convolutions shrink the channel count before the expensive $3\times3$ and $5\times5$ filters.

Write `inception(x, p)` that returns `(y, macs)`.

## Branches (each conv adds its bias and is followed by ReLU)
1. `p['b1']`: a 1x1 conv.
2. `p['b2_reduce']` (1x1) then `p['b2']` (3x3, zero padding 1).
3. `p['b3_reduce']` (1x1) then `p['b3']` (5x5, zero padding 2).
4. A 3x3 **max** pool with stride 1 and padding 1, where padded positions never win (treat them as $-\infty$), then `p['b4']` (1x1).

Every entry of `p` is a `(W, b)` pair, with `W` of shape `(F, C_in, k, k)` and `b` of shape `(F,)`. All convs have stride 1, so every branch keeps the input's $H \times W$.

## Output
- `y`: float `(N, F_1 + F_2 + F_3 + F_4, H, W)`, branches concatenated in the order 1, 2, 3, 4 (for branch 2 and 3, their **second** conv's channels).
- `macs`: the number of multiply-accumulates performed by all six convolutions for the whole batch, an `int`. A conv producing an `(N, F, H, W)` output from `C_in` channels with a $k \times k$ kernel costs $N \cdot F \cdot H \cdot W \cdot C_\text{in} \cdot k^2$. Pooling, biases and ReLUs are free.
"""

STARTER = '''import numpy as np

def inception(x, p):
    """p: dict of (W, b) for b1, b2_reduce, b2, b3_reduce, b3, b4. Return (y, macs)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _conv_relu(x, Wb, counter):
    W, b = (np.asarray(a, dtype=float) for a in Wb)
    N, C, H, Wd = x.shape
    F, _, k, _ = W.shape
    pad = k // 2                                         # "same" padding for odd k
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    y = np.zeros((N, F, H, Wd))
    for u in range(k):
        for v in range(k):
            y += np.einsum('nchw,fc->nfhw', xp[:, :, u:u + H, v:v + Wd], W[:, :, u, v])
    counter[0] += N * F * H * Wd * C * k * k
    return np.maximum(y + b[None, :, None, None], 0.0)

def _maxpool3_same(x):
    H, W = x.shape[2:]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)   # padding never wins
    return np.max([xp[:, :, u:u + H, v:v + W] for u in range(3) for v in range(3)], axis=0)

def inception(x, p):
    x = np.asarray(x, dtype=float)
    macs = [0]
    b1 = _conv_relu(x, p['b1'], macs)
    b2 = _conv_relu(_conv_relu(x, p['b2_reduce'], macs), p['b2'], macs)
    b3 = _conv_relu(_conv_relu(x, p['b3_reduce'], macs), p['b3'], macs)
    b4 = _conv_relu(_maxpool3_same(x), p['b4'], macs)
    return np.concatenate([b1, b2, b3, b4], axis=1), int(macs[0])
'''

HINTS = [
    'Every conv here is stride 1 with "same" padding $k // 2$, so a single helper (conv + bias + ReLU) serves all six. Have it also add $N F H W C_\\text{in} k^2$ to a running total.',
    'For the pooling branch pad with `constant_values=-np.inf`, then take the elementwise max of the nine shifted `(H, W)` slices. Zero padding would be wrong whenever a border window is all negative.',
    'Chain the reduce convs into their 3x3/5x5 convs, run the pool branch, and `np.concatenate` the four outputs on axis 1 in order.',
]

_r = np.random.default_rng(59)
def _cw(F, C, k):
    return (_r.normal(size=(F, C, k, k)) / np.sqrt(C * k * k), _r.normal(size=F) * 0.1)
def _mod(C, f1, r2, f2, r3, f3, f4):
    return {'b1': _cw(f1, C, 1), 'b2_reduce': _cw(r2, C, 1), 'b2': _cw(f2, r2, 3),
            'b3_reduce': _cw(r3, C, 1), 'b3': _cw(f3, r3, 5), 'b4': _cw(f4, C, 1)}
_one = lambda F, C, k, w=1.0: (np.full((F, C, k, k), w), np.zeros(F))
TESTS = [
    (-np.arange(4.0).reshape(1, 1, 2, 2), {'b1': _one(1, 1, 1, -1.0), 'b2_reduce': _one(1, 1, 1, -1.0), 'b2': _one(1, 1, 3),
                                          'b3_reduce': _one(1, 1, 1), 'b3': _one(1, 1, 5), 'b4': _one(1, 1, 1, -1.0)}),
    (np.ones((1, 2, 3, 3)), {'b1': _one(1, 2, 1), 'b2_reduce': _one(1, 2, 1), 'b2': _one(2, 1, 3),
                             'b3_reduce': _one(1, 2, 1), 'b3': _one(1, 1, 5), 'b4': _one(1, 2, 1)}),
    (_r.normal(size=(2, 4, 6, 6)), _mod(4, 3, 2, 4, 1, 2, 2)),
    (_r.normal(size=(1, 6, 5, 7)), _mod(6, 2, 3, 3, 2, 2, 1)),
    (-np.abs(_r.normal(size=(1, 3, 4, 4))), _mod(3, 1, 1, 1, 1, 1, 3)),
    (_r.normal(size=(2, 8, 3, 3)), _mod(8, 4, 4, 6, 2, 3, 2)),
]

EXPLANATION = r"""
## Why 1x1 reductions
A 5x5 conv from 192 to 32 channels on a 28x28 map costs $28^2 \cdot 32 \cdot 192 \cdot 25 \approx 120$M MACs. Reducing to 16 channels first costs $28^2 \cdot 16 \cdot 192 \approx 2.4$M plus $28^2 \cdot 32 \cdot 16 \cdot 25 \approx 10$M: about 10x cheaper. The `macs` output lets you see this trade directly. The 1x1 convs also add an extra ReLU, so the reduction is a learned, nonlinear compression rather than a loss.

## Multi-scale in one layer
The four branches see 1x1, 3x3 and 5x5 neighbourhoods plus a pooled summary, and the next layer can pick whichever scale is useful. Every branch keeps $H \times W$, so concatenation along channels just works.

## Max pooling with padding
Padding a max pool with zeros changes the answer wherever a border window is all negative (test 5). Frameworks pad max pools with $-\infty$, which is the same as taking the max over the valid part of each window.

## Complexity
The output channel count is $F_1 + F_2 + F_3 + F_4$, so stacking modules without reductions would grow channels (and cost) without bound.

## Common bugs
- Concatenating the reduce outputs instead of the second convs.
- Forgetting the ReLU after the reduce convs.
- Counting MACs with the input channels of the module instead of each conv's own $C_\text{in}$.

## Interview follow-ups
- Inception-v3 replaces a 5x5 by two stacked 3x3s, and an $n \times n$ by $1 \times n$ then $n \times 1$. Compute the savings.
- How does this relate to depthwise separable convolutions (Xception = "extreme Inception")?
"""
