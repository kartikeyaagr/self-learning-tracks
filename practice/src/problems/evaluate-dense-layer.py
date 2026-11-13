import numpy as np

META = dict(title='A dense layer of Value neurons', tags=['Neural Networks'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:05.02'],
            related=['build-scalar-neuron-module', 'evaluate-multi-layer-perceptron', 'jacobian-of-neural-network-layer'])

STATEMENT = r"""
A dense (fully connected) layer is a list of independent neurons that all read the same input vector:

$$\text{out}_j = \phi\Big(b_j + \sum_i W_{j,i}\, x_i\Big), \qquad j = 0, \dots, n_{\text{out}} - 1$$

with $\phi = \tanh$ when `nonlin` is true and the identity otherwise.

The starter has the `Value` engine and a finished `Neuron(w, b, nonlin)` (call it with a list, get a `Value`; `parameters()` is weights then bias). You write:

1. `Layer(W, b, nonlin)`: one `Neuron` per row of `W` (row `j` holds neuron `j`'s weights, `b[j]` its bias). Calling the layer on a list `x` returns the **list** of neuron outputs. `parameters()` returns every neuron's parameters, neuron by neuron.
2. `dense_layer(W, b, x, upstream, nonlin)`: wrap `x` in `Value`s, run the layer, form the scalar $L = \sum_j g_j\, \text{out}_j$ with $g$ = `upstream`, backpropagate, and return

```
{"out": [out_j], "x_grad": [dL/dx_i], "W_grad": [[dL/dW_ji] for each row j], "b_grad": [dL/db_j]}
```

## Inputs
`W`: list of $n_{\text{out}} \ge 1$ rows, each of length $n_{\text{in}} \ge 1$. `b`, `upstream`: lists of length $n_{\text{out}}$. `x`: list of length $n_{\text{in}}$. `nonlin`: bool.

Weighting the outputs by `upstream` is how you probe a layer in isolation: $g$ plays the role of the gradient arriving from the rest of the network.
"""

HEAD_STARTER = '''import math

def dense_layer(W, b, x, upstream, nonlin):
    """Return {"out", "x_grad", "W_grad", "b_grad"} for L = sum_j upstream[j] * out[j]."""
    raise NotImplementedError


class Layer:
    def __init__(self, W, b, nonlin=True):
        raise NotImplementedError   # TODO

    def __call__(self, x):
        raise NotImplementedError   # TODO

    def parameters(self):
        raise NotImplementedError   # TODO
'''

HEAD_SOLUTION = '''import math

def dense_layer(W, b, x, upstream, nonlin):
    layer = Layer(W, b, nonlin)
    xs = [Value(v) for v in x]                     # Values, so the inputs receive gradients too
    outs = layer(xs)
    L = sum(g * o for g, o in zip(upstream, outs))
    L.backward()
    return {
        "out": [o.data for o in outs],
        "x_grad": [v.grad for v in xs],
        "W_grad": [[w.grad for w in n.w] for n in layer.neurons],
        "b_grad": [n.b.grad for n in layer.neurons],
    }


class Layer:
    def __init__(self, W, b, nonlin=True):
        self.neurons = [Neuron(row, bj, nonlin) for row, bj in zip(W, b)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]        # every neuron reads the same input nodes

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
'''

NEURON = '''

class Neuron:
    def __init__(self, w, b, nonlin=True):
        self.w = [Value(wi) for wi in w]
        self.b = Value(b)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]
'''

ENGINE = '''

class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
'''

STARTER = HEAD_STARTER + NEURON + ENGINE
SOLUTION = HEAD_SOLUTION + NEURON + ENGINE

HINTS = [
    'The layer is a thin wrapper: build `[Neuron(row, bj, nonlin) for row, bj in zip(W, b)]` and call each neuron on the same `x`.',
    'Wrap the inputs in `Value`s before calling the layer, otherwise there\'s nowhere for $\\partial L / \\partial x$ to accumulate. Then `L = sum(g * o for g, o in zip(upstream, outs))` and `L.backward()`.',
    'Read the gradients off the stored objects: `[[w.grad for w in n.w] for n in layer.neurons]` and `[n.b.grad for n in layer.neurons]`. Each input node collects one contribution per neuron.',
]

_r = np.random.default_rng(73)
_v = lambda *s: np.round(_r.uniform(-1, 1, size=s), 3).tolist()
TESTS = [
    ([[1.0, 2.0], [0.5, -1.0]], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], False),
    ([[1.0, 2.0], [0.5, -1.0]], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], True),
    ([[3.0]], [-1.0], [2.0], [0.5], True),
    ([[0.2, -0.4, 0.6]], [0.1], [1.0, 2.0, 3.0], [2.0], False),
    ([[1.0], [-1.0], [2.0]], [0.0, 0.0, 0.0], [0.5], [1.0, 1.0, 1.0], True),
    ([[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0], [5.0, -5.0], [1.0, -1.0], True),
    (_v(4, 3), _v(4), _v(3), _v(4), True),
    (_v(2, 6), _v(2), _v(6), _v(2), False),
]

EXPLANATION = r"""
## The layer's backward pass in matrix form
Write $a = Wx + b$, $o = \phi(a)$ and $\delta_j = g_j\, \phi'(a_j)$. Then

$$\frac{\partial L}{\partial b_j} = \delta_j, \qquad \frac{\partial L}{\partial W_{j,i}} = \delta_j\, x_i, \qquad \frac{\partial L}{\partial x_i} = \sum_j W_{j,i}\, \delta_j$$

so `W_grad` is the outer product $\delta x^\top$ and `x_grad` is $W^\top \delta$. The scalar engine computes exactly this, one multiply node at a time. The sum in `x_grad` appears because every neuron reads the same input nodes: each contributes through its own weight, and `+=` adds them up.

## Checking by hand
Test 1 is linear with $g = (1, 1)$: `x_grad` is the column sums of $W$, `(1.5, 1.0)`, and `W_grad` repeats $x$ in each row. In test 6 all weights are zero, so $x$ gets no gradient but the weights do: $\partial L/\partial W_{j,i} = g_j x_i$.

## Complexity
$O(n_{\text{out}} \cdot n_{\text{in}})$ nodes and time, the same as a matrix-vector product, but with a large constant: each scalar multiply is a Python object. Real frameworks treat the whole layer as one op with matrix-valued rules.

## Common bugs
- Passing plain floats as `x`, so `x_grad` can't be read.
- Building one `Neuron` and reusing it for every row.
- Transposing `W`: row $j$ is neuron $j$'s weights, so `W_grad[j]` must have length $n_{\text{in}}$.

## Interview follow-ups
- Write the backward pass of a dense layer for a whole batch $X \in \mathbb{R}^{B \times n_{\text{in}}}$ using matrix products.
- Why is it fine for the gradient of $x$ to be computed even when $x$ is data, not a parameter? When is it useful? (Adversarial examples, saliency maps, and passing gradients to the previous layer.)
"""
