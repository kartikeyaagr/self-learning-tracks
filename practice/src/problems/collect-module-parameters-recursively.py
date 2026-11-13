import numpy as np

META = dict(title='Named parameters of nested modules', tags=['Neural Networks'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:05.03'],
            related=['evaluate-multi-layer-perceptron', 'build-scalar-neuron-module', 'apply-one-full-sgd-training-step'])

STATEMENT = r"""
An optimiser, a checkpoint and a `zero_grad()` all need the same thing: every parameter of a model, in a fixed order, with a stable name. Collect them recursively through the module tree.

The model is `MLP(nin, sizes)`: `len(sizes)` layers, layer $\ell$ has `sizes[ℓ]` neurons, each neuron has one weight per input of its layer plus a bias. Hidden layers apply $\tanh$, the last layer is linear. The starter builds all of this with every parameter set to `0.0`.

## What you write
Implement `named_parameters(prefix="")` for each class, returning a list of `(name, Value)` pairs:

| class | entries |
|---|---|
| `Neuron` | `prefix + "w.k"` for each weight $k$, then `prefix + "b"` |
| `Layer` | for each neuron $j$: that neuron's entries with prefix `prefix + "neurons.j."` |
| `MLP` | for each layer $i$: that layer's entries with prefix `prefix + "layers.i."` |

So the first name of any MLP is `"layers.0.neurons.0.w.0"`. The base class `Module` turns this into `parameters()` (the `Value`s alone, same order) and `zero_grad()`; both are given and rely on your method.

`inspect_model(nin, sizes, x)` (given) lists the names, then assigns parameter number $i$ (in `parameters()` order) the value $\frac{(i \bmod 7) - 3}{10}$ **through the objects `parameters()` returns**, and evaluates the model on `x`. It returns `{"count": ..., "names": [...], "out": [...]}`. If `parameters()` hands back copies, or uses a different order, `out` will be wrong.
"""

HEAD = '''import math

def inspect_model(nin, sizes, x):
    model = MLP(nin, sizes)
    names = [name for name, _ in model.named_parameters()]
    for i, p in enumerate(model.parameters()):
        p.data = ((i % 7) - 3) / 10
    return {"count": len(names), "names": names, "out": [o.data for o in model(x)]}


class Module:
    def named_parameters(self, prefix=""):
        return []

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
'''

STARTER = HEAD + '''

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(0.0) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    # TODO: named_parameters(self, prefix="")


class Layer(Module):
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    # TODO: named_parameters(self, prefix="")


class MLP(Module):
    def __init__(self, nin, sizes):
        dims = [nin] + list(sizes)
        self.layers = [Layer(dims[i], dims[i + 1], nonlin=(i < len(sizes) - 1)) for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # TODO: named_parameters(self, prefix="")
'''

SOLUTION = HEAD + '''

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(0.0) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def named_parameters(self, prefix=""):
        return [(f"{prefix}w.{k}", w) for k, w in enumerate(self.w)] + [(f"{prefix}b", self.b)]


class Layer(Module):
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def named_parameters(self, prefix=""):
        # Each child extends the prefix and recurses; the leaves (neurons) own the Values.
        return [item for j, n in enumerate(self.neurons) for item in n.named_parameters(f"{prefix}neurons.{j}.")]


class MLP(Module):
    def __init__(self, nin, sizes):
        dims = [nin] + list(sizes)
        self.layers = [Layer(dims[i], dims[i + 1], nonlin=(i < len(sizes) - 1)) for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def named_parameters(self, prefix=""):
        return [item for i, l in enumerate(self.layers) for item in l.named_parameters(f"{prefix}layers.{i}.")]
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

STARTER += ENGINE
SOLUTION += ENGINE

HINTS = [
    'Only `Neuron` owns `Value`s. `Layer` and `MLP` just ask their children and glue the results together, each adding one level to the prefix.',
    'For the neuron: `[(f"{prefix}w.{k}", w) for k, w in enumerate(self.w)] + [(f"{prefix}b", self.b)]`.',
    'For a container: `[item for j, child in enumerate(children) for item in child.named_parameters(f"{prefix}neurons.{j}.")]` (with `layers.{i}.` for the MLP). Return the stored `Value`s themselves, never `Value(p.data)`.',
]

TESTS = [
    (2, [1], [1.0, 1.0]),
    (1, [2, 1], [0.5]),
    (3, [2, 2], [1.0, -1.0, 2.0]),
    (2, [3, 3, 1], [0.25, -0.75]),
    (1, [1, 1, 1, 1], [2.0]),
    (4, [1], [0.1, 0.2, 0.3, 0.4]),
    (2, [12, 2], [-1.0, 0.5]),
]

EXPLANATION = r"""
## Why recursion
A model is a tree: an MLP has layers, a layer has neurons, a neuron has values. Each node of the tree only knows its own children, so "all parameters" is naturally a recursive concatenation. Adding a new kind of module (say, a residual block containing two layers) only requires that module to list its children; the base class's `parameters()` and `zero_grad()` keep working unchanged. PyTorch's `nn.Module.named_parameters()` works the same way and produces names like `layers.0.weight`.

## Why names and order matter
- **Optimisers** keep per-parameter state (momentum, Adam's moments) in the same order as `parameters()`. A different order on reload silently pairs the wrong state with the wrong weight.
- **Checkpoints** store `{name: value}`, so a model can be reloaded even after its code is refactored, as long as the names are stable.
- **Identity.** `inspect_model` changes `p.data` through the returned objects and then runs the model. That only works if `parameters()` returns the live `Value`s the model computes with.

## Counting
For $d = [\text{nin}] + \text{sizes}$ there are $\sum_\ell (d_\ell + 1)\,d_{\ell + 1}$ parameters: test 3 has $4 \cdot 2 + 3 \cdot 2 = 14$.

## Complexity
$O(P)$ for $P$ parameters per call; the list concatenations are linear in total because each name is built once.

## Common bugs
- Forgetting the trailing dot in the prefix (`layers.0neurons.1...`).
- Numbering neurons globally instead of per layer.
- Returning copies, so updates or `zero_grad()` never reach the model.

## Interview follow-ups
- Two layers share (tie) the same weight object. How should `parameters()` avoid returning it twice, and why would returning it twice be a problem for an optimiser?
- How would you implement `state_dict()` and `load_state_dict()` on top of `named_parameters()`?
"""
