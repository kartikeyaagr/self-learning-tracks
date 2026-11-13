import numpy as np

META = dict(title='A Neuron module with shared parameters', tags=['Neural Networks'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:05.01'],
            related=['evaluate-scalar-neuron', 'evaluate-dense-layer', 'collect-module-parameters-recursively'])

STATEMENT = r"""
So far a neuron was a one-off expression. A *module* owns its parameters as long-lived `Value` objects, so it can be called many times and every call contributes to the same parameter gradients.

The starter has a complete `Value` engine (`+`, `-`, `*`, `**`, `tanh`, `backward()`). Implement `Neuron`:

- `Neuron(w, b, nonlin)` stores one `Value` per weight in `self.w` (a list) and the bias as a `Value` in `self.b`. `nonlin` is a bool.
- `neuron(x)` takes a list of plain floats (or `Value`s) of length `len(w)` and returns the `Value`

$$\text{out} = \begin{cases} \tanh\left(b + \sum_i w_i x_i\right) & \text{nonlin} \\ b + \sum_i w_i x_i & \text{otherwise} \end{cases}$$

- `neuron.parameters()` returns `self.w + [self.b]`: the weights in order, then the bias. Return the stored objects themselves, not copies.

`neuron_module(w, b, xs, nonlin)` (given) calls the **same** neuron on every input vector in `xs`, adds the outputs into one `Value`, calls `backward()` on that sum, and returns

```
{"outs": [output data for each x], "grads": [p.grad for p in neuron.parameters()], "n_params": count}
```

## Inputs
`w`: list of $k \ge 1$ floats; `b`: float; `xs`: list of $m \ge 1$ input lists, each of length $k$; `nonlin`: bool.
"""

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

HEAD = '''import math

def neuron_module(w, b, xs, nonlin):
    neuron = Neuron(w, b, nonlin)
    outs = [neuron(x) for x in xs]
    total = sum(outs)
    total.backward()
    params = neuron.parameters()
    return {"outs": [o.data for o in outs], "grads": [p.grad for p in params], "n_params": len(params)}
'''

STARTER = HEAD + '''

class Neuron:
    def __init__(self, w, b, nonlin=True):
        raise NotImplementedError   # TODO: store Values

    def __call__(self, x):
        raise NotImplementedError   # TODO

    def parameters(self):
        raise NotImplementedError   # TODO
''' + ENGINE

SOLUTION = HEAD + '''

class Neuron:
    def __init__(self, w, b, nonlin=True):
        self.w = [Value(wi) for wi in w]
        self.b = Value(b)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)    # start the sum at the bias
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]            # the live objects, so their .grad is visible to the caller
''' + ENGINE

HINTS = [
    'Create the `Value`s once, in `__init__`. If `__call__` created fresh `Value`s from plain floats each time, every call would get its own parameters and nothing would accumulate.',
    '`sum(iterable, start)` takes a start value: `sum((wi * xi for wi, xi in zip(self.w, x)), self.b)` builds $b + \\sum w_i x_i$ from `Value` weights and float inputs.',
    '`parameters()` is `self.w + [self.b]`. Because the same objects are used by every call, `total.backward()` leaves each parameter holding the sum of its gradients over all inputs.',
]

_r = np.random.default_rng(71)
_v = lambda *s: np.round(_r.uniform(-1, 1, size=s), 3).tolist()
TESTS = [
    ([0.5, -1.0], 0.25, [[1.0, 2.0]], False),
    ([0.5, -1.0], 0.25, [[1.0, 2.0], [3.0, 0.0]], False),
    ([0.5, -1.0], 0.25, [[1.0, 2.0], [3.0, 0.0]], True),
    ([2.0], 0.0, [[0.0], [0.0], [0.0]], True),
    ([1.0, 1.0, 1.0], -0.5, [[10.0, 10.0, 10.0], [0.1, 0.2, 0.3]], True),
    (_v(4), float(round(_r.uniform(-1, 1), 3)), _v(5, 4), True),
    (_v(3), float(round(_r.uniform(-1, 1), 3)), _v(6, 3), False),
]

EXPLANATION = r"""
## Parameters live in the module
A module is a bundle of persistent `Value`s plus a function that builds a fresh graph on top of them on each call. Every call creates new intermediate nodes, but they all point back to the **same** weight objects. So when the outputs of several calls are summed and backpropagated, each weight receives one contribution per call, which is how a batch gradient is formed. Compare tests 1 and 2: adding a second input adds that input's contribution to every gradient.

For the linear neuron the result is easy to verify by hand: $\partial/\partial w_i \sum_m \text{out}_m = \sum_m x_{m,i}$ and $\partial/\partial b = m$ (test 2: grads `[4, 2, 2]`).

## Why return the live objects
An optimiser updates `p.data` for every `p in model.parameters()`. If `parameters()` returned copies, the update would hit the copies and the model would never change, without any error.

## Complexity
Each call is $O(k)$ nodes; a batch of $m$ inputs builds $O(mk)$ nodes, and one backward pass over them is $O(mk)$.

## Common bugs
- Re-wrapping the weights in new `Value`s inside `__call__`.
- Starting the sum at Python `0` and adding the bias inside the loop, once per input element.
- Putting the bias first in `parameters()`: the order is a contract that optimisers and checkpoints rely on.

## Interview follow-ups
- micrograd-style neurons initialise weights randomly in $[-1, 1]$. Why not zero? (Every neuron in a layer would compute the same thing and receive the same gradient, so they could never become different.)
- How does a framework like PyTorch find the parameters of a module without each class listing them by hand? (Attribute registration: assigning a `Parameter` or sub-module to `self` records it.)
"""
