import numpy as np

META = dict(title='Forward and backward through a Value MLP', tags=['Neural Networks'], diff='Hard', runtime='python',
            lessons=['air-fanout-v1:05.03'],
            related=['evaluate-dense-layer', 'collect-module-parameters-recursively', 'jvp-and-vjp-for-2-layer-network'])

STATEMENT = r"""
Stack dense layers into a multi-layer perceptron and push gradients through all of them.

`MLP(nin, sizes, params)` has `len(sizes)` layers: layer $\ell$ maps $d_\ell$ inputs to $d_{\ell+1}$ outputs where $d = [\text{nin}] + \text{sizes}$. Hidden layers use $\tanh$; the **last** layer is linear (no activation). Calling the MLP on a list of length `nin` returns the list of the last layer's outputs.

## Parameter layout
Weights come in as one flat list `params`, consumed in this order: layer by layer, and inside a layer neuron by neuron; for each neuron, its $d_\ell$ weights and then its bias. So the total is $\sum_\ell (d_\ell + 1)\, d_{\ell+1}$, and `parameters()` must return the `Value`s in exactly the same order.

## What you write
The starter has the `Value` engine and finished `Neuron(nin, take, nonlin)` / `Layer(nin, nout, take, nonlin)` classes, where `take()` returns the next number from the flat list. Write the `MLP` class (construction, call, `parameters()`), then `mlp(nin, sizes, params, x)`, which runs the MLP on `x` (wrapped in `Value`s), backpropagates the **sum** of the outputs, and returns

```
{"out": [outputs], "grads": [d sum / d param, in layout order], "x_grad": [d sum / d x_i]}
```

## Inputs
`nin` $\ge 1$; `sizes`: list of $\ge 1$ positive ints; `params`: list of exactly the right length; `x`: list of `nin` floats.
"""

HEAD_STARTER = '''import math

def mlp(nin, sizes, params, x):
    """Return {"out", "grads", "x_grad"} for the sum of the MLP outputs."""
    raise NotImplementedError


class MLP:
    def __init__(self, nin, sizes, params):
        raise NotImplementedError   # TODO

    def __call__(self, x):
        raise NotImplementedError   # TODO

    def parameters(self):
        raise NotImplementedError   # TODO
'''

HEAD_SOLUTION = '''import math

def mlp(nin, sizes, params, x):
    model = MLP(nin, sizes, params)
    xs = [Value(v) for v in x]
    out = model(xs)
    sum(out).backward()
    return {"out": [o.data for o in out], "grads": [p.grad for p in model.parameters()],
            "x_grad": [v.grad for v in xs]}


class MLP:
    def __init__(self, nin, sizes, params):
        it = iter(params)
        take = lambda: next(it)                    # every layer pulls from the same iterator
        dims = [nin] + list(sizes)
        self.layers = [Layer(dims[i], dims[i + 1], take, nonlin=(i < len(sizes) - 1))
                       for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)                           # a list of Values in, a list of Values out
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
'''

MODULES = '''

class Neuron:
    def __init__(self, nin, take, nonlin=True):
        self.w = [Value(take()) for _ in range(nin)]
        self.b = Value(take())
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, take, nonlin=True):
        self.neurons = [Neuron(nin, take, nonlin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
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

STARTER = HEAD_STARTER + MODULES + ENGINE
SOLUTION = HEAD_SOLUTION + MODULES + ENGINE

HINTS = [
    'Turn `params` into an iterator once (`it = iter(params)`) and hand every layer the same `take = lambda: next(it)`. Construction order then matches the layout automatically.',
    'Layer `i` has `dims[i]` inputs and `dims[i + 1]` outputs with `dims = [nin] + sizes`, and `nonlin = i < len(sizes) - 1` so only the last layer is linear.',
    'The forward pass is `for layer in self.layers: x = layer(x)`. For the gradients, wrap `x` in `Value`s, call `sum(outputs).backward()`, and read `p.grad` for `p in model.parameters()`.',
]


def _case(rng, nin, sizes):
    dims = [nin] + sizes
    n = sum((dims[i] + 1) * dims[i + 1] for i in range(len(sizes)))
    return nin, sizes, np.round(rng.uniform(-1, 1, size=n), 3).tolist(), np.round(rng.uniform(-1, 1, size=nin), 3).tolist()


_r = np.random.default_rng(79)
TESTS = [
    (1, [1], [2.0, 0.5], [3.0]),
    (2, [1, 1], [0.5, -0.5, 0.1, 2.0, -1.0], [1.0, 2.0]),
    (2, [2, 1], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0.5, -0.5]),
    (1, [2], [1.0, 0.0, -1.0, 0.0], [0.7]),
    (3, [2, 2], [0.0] * 14, [1.0, 2.0, 3.0]),
    _case(_r, 3, [4, 4, 1]),
    _case(_r, 2, [3, 2]),
]

EXPLANATION = r"""
## How the pieces compose
An MLP is function composition: $f = f_L \circ \dots \circ f_1$. The scalar engine doesn't need to know that. Each layer builds its nodes on top of the previous layer's output nodes, so one `backward()` from the final sum walks through every layer in reverse, multiplying by each layer's local derivatives on the way. This is the chain rule applied across layers, which is all "backpropagation" means.

## Why the last layer is linear
Hidden tanh layers add the nonlinearity; the output layer should be free to produce any real number (for regression or as logits for a later softmax). A tanh on the output would cap predictions at $\pm 1$.

## Checking the plumbing
- Test 1 has no hidden layer: out $= 2 \cdot 3 + 0.5$, grads `[3, 1]`, `x_grad = [2]`.
- Test 3 uses identity-like first-layer weights, so the hidden layer is $\tanh(x)$ and the output just adds the two hidden units.
- Test 5 has all-zero parameters: the outputs are 0, the hidden activations are 0, so only the last layer's biases get gradient (1 each), and nothing reaches the first layer. That's the symmetry problem with zero initialisation.

## Complexity
$O(\#\text{params})$ nodes for one input, and the backward pass costs about the same as the forward. In this scalar engine both are slow by a factor of a few hundred compared with NumPy, which is fine for learning and hopeless for real models.

## Common bugs
- Creating a fresh iterator per layer, so every layer starts reading from `params[0]`.
- Applying tanh on the last layer, or on none.
- `parameters()` order differing from construction order, so `grads[i]` doesn't belong to `params[i]`.

## Interview follow-ups
- How many parameters does an MLP with `nin = 784` and `sizes = [128, 64, 10]` have? ($785 \cdot 128 + 129 \cdot 64 + 65 \cdot 10 = 109{,}386$.)
- What changes in the backward pass if the hidden activation is ReLU and many units are inactive?
"""
