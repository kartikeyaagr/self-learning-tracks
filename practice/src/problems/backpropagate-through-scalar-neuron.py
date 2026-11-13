import numpy as np

META = dict(title='Backward closures for a tanh neuron', tags=['Neural Networks', 'Calculus'], diff='Medium',
            runtime='python', lessons=['air-fanout-v1:01.07', 'air-fanout-v1:05.01'],
            related=['evaluate-scalar-neuron', 'differentiate-tanh-activation', 'gradient-check-scalar-neuron'])

STATEMENT = r"""
The neuron $o = \tanh\left(b + \sum_i x_i w_i\right)$ is built from `Value` nodes. Make gradients flow back through it.

Each node has `.grad` (starting at `0.0`) and a `._backward()` that reads the node's own `.grad` and **adds** the right amount to each operand's `.grad`. Write the three backward rules, for out $= a + b$, out $= a \cdot b$ and out $= \tanh(a)$:

| op | contribution to `a.grad` | contribution to `b.grad` |
|---|---|---|
| `+` | `out.grad` | `out.grad` |
| `*` | `b.data * out.grad` | `a.data * out.grad` |
| `tanh` | `(1 - out.data**2) * out.grad` | (no second operand) |

`neuron_grads(x, w, b)` (given in the starter) builds the graph, keeping every operation node in creation order: for each input it creates `p = x_i * w_i` and then `n = n + p`, starting from `n = bias`, and finally `o = n.tanh()`. It sets `o.grad = 1.0` and calls `_backward()` on those nodes in **reverse** creation order. It returns

```
{"x": [dO/dx_i ...], "w": [dO/dw_i ...], "b": dO/db, "n": dO/dn}
```

## Inputs
`x`, `w`: lists of floats of equal length $k \ge 0$. `b`: float. All answers are compared with a relative tolerance of $10^{-6}$.
"""

STARTER = '''import math

def neuron_grads(x, w, b):
    xs, ws, bias = [Value(v) for v in x], [Value(v) for v in w], Value(b)
    order = []                       # operation nodes in creation order
    n = bias
    for xi, wi in zip(xs, ws):
        p = xi * wi
        n = n + p
        order += [p, n]
    o = n.tanh()
    order.append(o)

    o.grad = 1.0
    for node in reversed(order):     # children are always created before their parents
        node._backward()
    return {"x": [v.grad for v in xs], "w": [v.grad for v in ws], "b": bias.grad, "n": n.grad}


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            raise NotImplementedError    # TODO
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            raise NotImplementedError    # TODO
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _backward():
            raise NotImplementedError    # TODO
        out._backward = _backward
        return out
'''

SOLUTION = '''import math

def neuron_grads(x, w, b):
    xs, ws, bias = [Value(v) for v in x], [Value(v) for v in w], Value(b)
    order = []                       # operation nodes in creation order
    n = bias
    for xi, wi in zip(xs, ws):
        p = xi * wi
        n = n + p
        order += [p, n]
    o = n.tanh()
    order.append(o)

    o.grad = 1.0
    for node in reversed(order):     # children are always created before their parents
        node._backward()
    return {"x": [v.grad for v in xs], "w": [v.grad for v in ws], "b": bias.grad, "n": n.grad}


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():                 # d(a+b)/da = d(a+b)/db = 1: pass the gradient through
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():                 # each side gets the *other* side's value
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
'''

HINTS = [
    'Every rule has the form `operand.grad += (local derivative) * out.grad`. The closure runs later, so `out.grad` is filled in by then.',
    'Addition routes the gradient unchanged to both sides. Multiplication swaps: `self` gets `other.data * out.grad` and `other` gets `self.data * out.grad`.',
    'For tanh use the forward output: `self.grad += (1 - out.data ** 2) * out.grad`. Use `+=` everywhere; the bias node receives its gradient from the first `+` node only, but in bigger graphs nodes have several consumers.',
]

_r = np.random.default_rng(19)
_v = lambda k, s=1.0: [float(round(t, 3)) for t in _r.uniform(-s, s, size=k)]
TESTS = [
    ([1.0, -2.0], [0.5, 0.25], 0.3),
    ([3.0], [0.0], 0.0),
    ([], [], 0.5),
    ([1.0, 1.0, 1.0], [10.0, 10.0, 10.0], 0.0),
    ([0.0, 0.0], [2.0, -2.0], -0.4),
    (_v(4), _v(4), float(round(_r.uniform(-1, 1), 3))),
    (_v(7, 0.5), _v(7, 0.5), float(round(_r.uniform(-1, 1), 3))),
]

EXPLANATION = r"""
## What the numbers say
With $d = 1 - o^2$ (the slope of tanh at the output), the chain rule gives

$$\frac{\partial o}{\partial n} = d, \qquad \frac{\partial o}{\partial b} = d, \qquad \frac{\partial o}{\partial w_i} = d\, x_i, \qquad \frac{\partial o}{\partial x_i} = d\, w_i$$

Each `+` node passes $d$ straight down the chain to the bias and to every product node; each product node then hands out the other factor. In test 2 the weight is 0, so the input's gradient is 0 while the weight's gradient is $d \cdot 3$: a zero weight still learns. In test 5 the inputs are 0, so the weights get nothing, which is why we never feed raw zeros and expect those weights to move.

## Why reverse creation order works
A node is always created after its operands, so creation order is already a topological order, and reversing it guarantees that a node's `.grad` is complete before its `_backward` runs. The general engine (later in this plan) can't rely on a hand-kept list, so it rebuilds this order with a depth-first search.

## Saturation
Test 4 has $n = 30$: $o$ rounds to 1, $d = 0$, and every gradient is exactly zero. A saturated tanh neuron stops learning.

## Complexity
One backward call per node, $O(k)$ total, the same order as the forward pass.

## Common bugs
- In `*`, giving each operand its *own* value instead of the other's.
- Using `=` instead of `+=`.
- Using `1 - self.data**2` (the input) in tanh.

## Interview follow-ups
- Count the multiplications in the backward pass versus the forward pass. Why is "backward costs about 2x forward" a good rule of thumb?
- What would change if the neuron used ReLU instead of tanh?
"""
