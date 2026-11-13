import numpy as np

META = dict(title='Xavier, He and LeCun initialisation', tags=['Neural Networks', 'Training'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:05.02'], related=['multi-layer-perceptron-forward-pass', 'batch-normalization'])

STATEMENT = r"""
Initialisation schemes pick the weight variance so that activations (or gradients) keep roughly the same scale from layer to layer. They all depend on the layer's **fan-in** and **fan-out**.

Write `init_weights(shape, scheme, seed)` that returns a dict with keys `'W'`, `'fan_in'`, `'fan_out'` and `'std'`.

## Fans (PyTorch layout)
- `shape` is a tuple of at least 2 ints: `(out_features, in_features)` for a linear layer, or `(out_channels, in_channels, k_1, k_2, ...)` for a convolution.
- Let `rf` be the product of `shape[2:]` (1 for a linear layer). Then `fan_in = shape[1] * rf` and `fan_out = shape[0] * rf`.

## Schemes
| `scheme` | distribution | |
|---|---|---|
| `'xavier_uniform'` | $U(-a, a)$ | $a = \sqrt{6 / (\text{fan\_in} + \text{fan\_out})}$ |
| `'xavier_normal'` | $N(0, \sigma^2)$ | $\sigma = \sqrt{2 / (\text{fan\_in} + \text{fan\_out})}$ |
| `'he_uniform'` | $U(-a, a)$ | $a = \sqrt{6 / \text{fan\_in}}$ |
| `'he_normal'` | $N(0, \sigma^2)$ | $\sigma = \sqrt{2 / \text{fan\_in}}$ |
| `'lecun_normal'` | $N(0, \sigma^2)$ | $\sigma = \sqrt{1 / \text{fan\_in}}$ |

## Random draws (so your numbers match)
`rng = np.random.default_rng(seed)`, then exactly one call: `rng.uniform(-a, a, size=shape)` for the uniform schemes or `rng.normal(0.0, sigma, size=shape)` for the normal ones.

## Output
- `'W'`: float array of shape `shape`. `'fan_in'`, `'fan_out'`: Python `int`s.
- `'std'`: the standard deviation of the distribution as a Python `float` (for $U(-a, a)$ it is $a/\sqrt 3$).
"""

STARTER = '''import numpy as np

def init_weights(shape, scheme, seed):
    """Return {'W': array of `shape`, 'fan_in': int, 'fan_out': int, 'std': float}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def init_weights(shape, scheme, seed):
    shape = tuple(int(s) for s in shape)
    rf = int(np.prod(shape[2:])) if len(shape) > 2 else 1     # receptive field size of a conv kernel
    fan_in, fan_out = shape[1] * rf, shape[0] * rf
    var = {'xavier_uniform': 2.0 / (fan_in + fan_out), 'xavier_normal': 2.0 / (fan_in + fan_out),
           'he_uniform': 2.0 / fan_in, 'he_normal': 2.0 / fan_in, 'lecun_normal': 1.0 / fan_in}[scheme]
    std = float(np.sqrt(var))
    rng = np.random.default_rng(seed)
    if scheme.endswith('uniform'):
        a = np.sqrt(3.0) * std                                 # Var U(-a, a) = a^2 / 3
        W = rng.uniform(-a, a, size=shape)
    else:
        W = rng.normal(0.0, std, size=shape)
    return {'W': W, 'fan_in': fan_in, 'fan_out': fan_out, 'std': std}
'''

HINTS = [
    'Start with the fans: everything past the first two dimensions is the kernel, and it multiplies both fans.',
    'Every scheme is really a target variance: $2/(\\text{in}+\\text{out})$ for Xavier, $2/\\text{in}$ for He, $1/\\text{in}$ for LeCun. A uniform $U(-a, a)$ has variance $a^2/3$, which is where the 6 in the bounds comes from.',
    'Create the generator, then make exactly one draw with `size=shape`. The uniform bound is `sqrt(3) * std`, and `std` is what you return in both cases.',
]

TESTS = [
    ((3, 2), 'xavier_uniform', 0),
    ((2, 4), 'he_normal', 1),
    ((64, 32, 3, 3), 'he_uniform', 2),
    ((8, 3, 5, 5), 'xavier_normal', 3),
    ((5, 7), 'lecun_normal', 4),
    ((4, 2, 3), 'xavier_uniform', 5),
    ((16, 16), 'he_uniform', 6),
    ((1, 1, 1, 1), 'he_normal', 7),
]

EXPLANATION = r"""
## Where the variances come from
For $z = \sum_{i=1}^{n_\text{in}} w_i x_i$ with independent zero-mean weights and inputs, $\operatorname{Var} z = n_\text{in}\operatorname{Var} w\,\mathbb E[x^2]$.
- **LeCun**: keep $\operatorname{Var} z = \operatorname{Var} x$ for a linear or tanh-like unit, so $\operatorname{Var} w = 1/n_\text{in}$.
- **He**: a ReLU zeroes half of its inputs, so $\mathbb E[x^2] = \tfrac12 \operatorname{Var} z_\text{prev}$; doubling gives $2/n_\text{in}$.
- **Xavier/Glorot**: the backward pass wants $1/n_\text{out}$, the forward pass $1/n_\text{in}$; the compromise is the harmonic-style average $2/(n_\text{in} + n_\text{out})$.

## Conv fans
A conv output pixel sums over `in_channels * k * k` inputs, so that is its fan-in. Each input pixel influences `out_channels * k * k` outputs, which is the fan-out.

## Complexity
$O(\prod \text{shape})$ to fill the tensor.

## Common bugs
- Using `shape[0]` as the fan-in: PyTorch's linear weight is `(out, in)`, Keras stores `(in, out)`.
- Forgetting the receptive field for convolutions, which makes conv weights $k^2$ times too large.
- Using $a = \sigma$ for the uniform bound instead of $\sqrt 3\,\sigma$.

## Interview follow-ups
- Why does initialising every weight to the same constant fail, even if it has the "right" variance? (Symmetry: all units compute and receive the same thing forever.)
- With BatchNorm or LayerNorm after every layer, how much does the initial scale still matter? What about the residual branch scaling in very deep transformers?
"""
