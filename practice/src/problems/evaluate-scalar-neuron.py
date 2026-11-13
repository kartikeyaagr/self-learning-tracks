import numpy as np

META = dict(title='Forward pass of one neuron with Value nodes', tags=['Neural Networks', 'Activation Functions'],
            diff='Easy', runtime='python', lessons=['air-fanout-v1:05.01'],
            related=['differentiate-tanh-activation', 'backpropagate-through-scalar-neuron', 'build-scalar-neuron-module'])

STATEMENT = r"""
A neuron takes a weighted sum of its inputs, adds a bias, and squashes the result:

$$n = \sum_{i} w_i x_i + b, \qquad o = \phi(n)$$

Write `neuron(x, w, b, act)` that builds this with `Value` nodes and returns `{"n": n.data, "o": o.data}`.

## Inputs
- `x`, `w`: lists of floats of the same length $k \ge 0$. With $k = 0$ the pre-activation is just $b$.
- `b`: float. `act`: one of

| `act` | $\phi(n)$ |
|---|---|
| `"tanh"` | $\tanh n$ |
| `"relu"` | $\max(0, n)$ |
| `"linear"` | $n$ |

## Your job
The starter's `Value` already supports `+` and `*` (with numbers on either side). Add `tanh()` and `relu()` (forward only: a new node with the result and `_prev = (self,)`), then write `neuron` using these operators: wrap the inputs, weights and bias in `Value`s and build $n$ and $o$ from them.
"""

STARTER = '''import math

def neuron(x, w, b, act):
    """Return {"n": pre-activation, "o": activation output} computed with Value nodes."""
    raise NotImplementedError


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self._prev = tuple(_prev)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    __radd__ = __add__
    __rmul__ = __mul__

    # TODO: tanh() and relu()
'''

SOLUTION = '''import math

def neuron(x, w, b, act):
    n = Value(b)
    for xi, wi in zip(x, w):
        n = n + Value(xi) * Value(wi)          # each term is one * node feeding one + node
    o = {"tanh": n.tanh, "relu": n.relu, "linear": lambda: n}[act]()
    return {"n": n.data, "o": o.data}


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self._prev = tuple(_prev)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    __radd__ = __add__
    __rmul__ = __mul__

    def tanh(self):
        return Value(math.tanh(self.data), (self,), "tanh")

    def relu(self):
        return Value(max(0.0, self.data), (self,), "relu")
'''

HINTS = [
    'Start the running sum at the bias, `n = Value(b)`, so an empty input list needs no special case.',
    'In a loop over `zip(x, w)`, do `n = n + Value(xi) * Value(wi)`. Each step builds one multiply node and one add node.',
    'The activations are one-liners: `Value(math.tanh(self.data), (self,), "tanh")` and `Value(max(0.0, self.data), (self,), "relu")`. For `"linear"`, `o` is just `n`.',
]

_r = np.random.default_rng(12)
_v = lambda k: [float(round(t, 3)) for t in _r.uniform(-1, 1, size=k)]
TESTS = [
    ([1.0, -2.0], [0.5, 0.25], 0.3, "tanh"),
    ([1.0, 2.0, 3.0], [0.5, -1.0, 0.5], 0.0, "relu"),
    ([1.0, 2.0, 3.0], [0.5, -1.0, 0.5], 0.1, "relu"),
    ([], [], -0.7, "tanh"),
    ([4.0], [2.5], -1.0, "linear"),
    ([10.0, 10.0], [3.0, 3.0], 0.0, "tanh"),
    (_v(5), _v(5), float(round(_r.uniform(-1, 1), 3)), "tanh"),
    (_v(8), _v(8), float(round(_r.uniform(-1, 1), 3)), "relu"),
]

EXPLANATION = r"""
## What the neuron computes
The weighted sum is a dot product, a linear function of the inputs. The bias shifts it so the neuron can fire even when every input is 0. The nonlinearity is what makes stacks of neurons more expressive than a single linear map: without it, any number of layers collapses into one matrix multiply.

## Why build it from Value nodes
The numbers alone would be a one-line loop. Building them from `Value` nodes leaves behind a graph: one leaf per input, weight and bias, a `*` and a `+` per term, and the activation on top. The next problem walks that graph backwards to get the gradient for every weight, which is what makes the neuron trainable.

## Activation behaviour
- `tanh` maps to $(-1, 1)$ and saturates: test 6 has $n = 60$, where the output is exactly 1.0 in floating point.
- `relu` is 0 for negative inputs and the identity otherwise. Test 2 lands exactly at $n = 0$; test 3 sits just above it.
- `linear` is used for the last layer of a regression network, where outputs shouldn't be bounded.

## Complexity
$O(k)$ time and $2k + 1$ leaves, $2k + 1$ operation nodes in total for $k$ inputs.

## Common bugs
- Forgetting the bias, or adding it once per input inside the loop.
- Starting the sum from Python `0` and then needing `__radd__` (fine here, but it adds a hidden constant node).
- Returning `o` (a `Value`) instead of `o.data`.

## Interview follow-ups
- Why do neural networks need nonlinear activations at all?
- What happens to a ReLU neuron whose pre-activation is negative for every training example? (It is "dead": zero output and zero gradient, so it never recovers.)
"""
