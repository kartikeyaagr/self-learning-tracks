import numpy as np

META = dict(title='Forward pass of a configurable MLP', tags=['Neural Networks'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:05.03', 'air-fanout-v1:05.02'],
            related=['perceptron', 'activation-functions', 'backpropagation-single-hidden-layer'])

STATEMENT = r"""
A multi-layer perceptron is a stack of affine maps and nonlinearities:

$$z^{(k)} = a^{(k-1)} W_k + b_k, \qquad a^{(k)} = f_k\big(z^{(k)}\big), \qquad a^{(0)} = x.$$

Write `mlp_forward(x, layers)` that runs the network and returns `(out, cache)`.

## Inputs
- `x`: float array, either a single sample `(d,)` or a batch `(N, d)`.
- `layers`: a list of `(W, b, act)` tuples. `W` has shape `(in, out)` (rows are inputs), `b` has shape `(out,)`, and `act` is one of `'relu'`, `'tanh'`, `'sigmoid'`, `'identity'`, `'softmax'`. Softmax is over the last axis and must be stable for large logits.

## Output
- `out`: the final activation $a^{(L)}$. A 1-D input gives a 1-D output `(out,)`; a 2-D input gives `(N, out)`.
- `cache`: a list with one tuple `(z, a)` per layer, in order, holding that layer's pre-activation and activation, each shaped like `out` would be at that layer (1-D for a 1-D input). Backprop needs exactly these.
- An empty `layers` list returns `(x as float, [])`.
"""

STARTER = '''import numpy as np

def mlp_forward(x, layers):
    """layers: list of (W (in, out), b (out,), act). Return (out, [(z, a), ...])."""
    a = np.asarray(x, dtype=float)
    cache = []
    for W, b, act in layers:
        z = W @ a + b                      # is this the right orientation?
        a = np.maximum(z, 0.0)
        cache.append((z, a))
    return a, cache
'''

SOLUTION = '''import numpy as np

def _act(z, name):
    if name == 'relu':
        return np.maximum(z, 0.0)
    if name == 'tanh':
        return np.tanh(z)
    if name == 'sigmoid':
        e = np.exp(-np.abs(z))
        return np.where(z >= 0, 1 / (1 + e), e / (1 + e))
    if name == 'identity':
        return z
    if name == 'softmax':
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    raise ValueError(name)

def mlp_forward(x, layers):
    a = np.asarray(x, dtype=float)
    cache = []
    for W, b, act in layers:
        z = a @ np.asarray(W, dtype=float) + b      # works for (d,) and (N, d) alike
        a = _act(z, act)
        cache.append((z, a))
    return a, cache
'''

HINTS = [
    'With `W` stored as `(in, out)`, the layer is `a @ W + b`. For a 1-D `a` of shape `(in,)` that gives `(out,)` automatically; for `(N, in)` it gives `(N, out)`.',
    'Write one helper that maps an activation name to a function. Sigmoid and softmax need the usual stability tricks (split on the sign; subtract the row max).',
    'Store `(z, a)` after every layer. Note that the identity layer has `z` and `a` equal, and an empty network returns the input unchanged.',
]

_r = np.random.default_rng(8)
_L = [(_r.normal(size=(4, 6)), _r.normal(size=6), 'relu'),
      (_r.normal(size=(6, 5)), _r.normal(size=5), 'tanh'),
      (_r.normal(size=(5, 3)), _r.normal(size=3), 'softmax')]
TESTS = [
    (np.array([1.0, 2.0]), [(np.array([[1.0, -1.0], [0.0, 1.0]]), np.array([0.0, -2.0]), 'relu')]),
    (np.array([[1.0], [-1.0]]), [(np.array([[2.0]]), np.array([0.0]), 'identity'),
                                 (np.array([[1.0, -1.0]]), np.array([0.0, 0.0]), 'softmax')]),
    (_r.normal(size=(5, 4)), _L),
    (_r.normal(size=4), _L),
    (_r.normal(size=(3, 4)), [(_r.normal(size=(4, 2)) * 300, np.zeros(2), 'sigmoid'),
                              (_r.normal(size=(2, 4)) * 300, np.zeros(4), 'softmax')]),
    (np.array([[0.5, -0.5]]), []),
    (_r.normal(size=(2, 3)), [(_r.normal(size=(3, 3)), _r.normal(size=3), a) for a in ('tanh', 'sigmoid', 'identity', 'relu')]),
]

EXPLANATION = r"""
## Row-vector convention
Storing `W` as `(in, out)` and multiplying `a @ W` treats each sample as a row. A whole batch then goes through one matrix product, and the same code handles a single 1-D sample because NumPy's `@` treats a 1-D left operand as a row vector. The column convention $W a$ (PyTorch stores `(out, in)` and computes `x @ W.T`) is equivalent; what matters is being consistent.

## Why keep the cache
The backward pass needs $f_k'(z^{(k)})$ and $a^{(k-1)}$ at every layer: $\partial L/\partial W_k = (a^{(k-1)})^\top\,\delta^{(k)}$. Recomputing them is possible (activation checkpointing trades compute for memory), but by default frameworks store them, and that is what dominates training memory.

## Complexity
$\sum_k N \cdot n_{k-1} n_k$ multiply-adds; memory for the cache is $\sum_k N n_k$.

## Common bugs
- `W @ a` with `(in, out)` weights: shapes only line up by accident when the layer is square.
- An unstable softmax or sigmoid: test 5 feeds logits in the hundreds.
- Caching `a` by reference and then modifying it in place on the next layer.

## Interview follow-ups
- Count the parameters of a 784-256-128-10 MLP. (200,704 + 32,896 + 1,290 = 234,890.)
- Why does stacking linear layers without nonlinearities add no expressive power?
"""
