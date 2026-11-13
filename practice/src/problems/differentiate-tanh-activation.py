import numpy as np

META = dict(title='A tanh node with its backward rule', tags=['Activation Functions', 'Neural Networks'], diff='Easy',
            runtime='python', lessons=['air-fanout-v1:01.07'],
            related=['derivatives-of-common-activation-functions', 'evaluate-scalar-neuron', 'apply-local-vector-jacobian-rules'])

STATEMENT = r"""
Give the scalar `Value` node its first activation function, together with the rule that sends gradients back through it.

$$t = \tanh(x), \qquad \frac{dt}{dx} = 1 - t^2$$

Each `Value` now has a `.grad` (starts at `0.0`) and a `._backward` function. A node's `_backward` takes the node's own `.grad` (the gradient of the final output with respect to this node) and **adds** its contribution to the `.grad` of each operand.

Implement `Value.tanh()`:
- Return a new node with `data = tanh(self.data)`, `_prev = (self,)` and `_op = "tanh"`.
- Set that node's `_backward` so that calling it does `self.grad += (1 - t**2) * out.grad`.

`tanh_grads(xs, upstream)` (given) runs, for each pair: `v = Value(x)`, `t = v.tanh()`, `t.grad = g`, `t._backward()`, and returns a list of `[t.data, v.grad]` pairs.

## Inputs
`xs`, `upstream`: lists of floats of the same length. Inputs can be as large as $\pm 800$; your forward pass must not overflow there (at that size $\tanh = \pm 1$ and the gradient is 0).
"""

STARTER = '''import math

def tanh_grads(xs, upstream):
    out = []
    for x, g in zip(xs, upstream):
        v = Value(x)
        t = v.tanh()
        t.grad = g
        t._backward()
        out.append([t.data, v.grad])
    return out


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None     # leaves have nothing to pass on

    def tanh(self):
        # TODO: forward value, new node, and its _backward closure.
        raise NotImplementedError
'''

SOLUTION = '''import math

def tanh_grads(xs, upstream):
    out = []
    for x, g in zip(xs, upstream):
        v = Value(x)
        t = v.tanh()
        t.grad = g
        t._backward()
        out.append([t.data, v.grad])
    return out


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None

    def tanh(self):
        t = math.tanh(self.data)          # stable for any input; (e^2x - 1)/(e^2x + 1) overflows
        out = Value(t, (self,), "tanh")

        def _backward():
            # The closure captures self, out and t: it reads out.grad when it is finally called.
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out
'''

HINTS = [
    'The forward value is one call: `math.tanh(self.data)`. Building it from `math.exp(2 * x)` overflows for large inputs.',
    'Define `_backward` as a nested function inside `tanh`, after creating `out`. It can use `self`, `out` and the forward value `t` from the enclosing scope.',
    'Inside it: `self.grad += (1 - t * t) * out.grad`. Then assign `out._backward = _backward` and return `out`.',
]

_r = np.random.default_rng(4)
TESTS = [
    ([0.0, 0.5], [1.0, 1.0]),
    ([1.0, -1.0], [2.0, -3.0]),
    ([800.0, -800.0], [1.0, 1.0]),
    ([0.8814], [0.0]),
    ([20.0, -2.5, 0.001], [1.0, 1.0, 1000.0]),
    ([float(round(v, 4)) for v in _r.normal(0, 2, size=6)], [float(round(v, 3)) for v in _r.normal(0, 1, size=6)]),
]

EXPLANATION = r"""
## The derivative from the output
$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$, and the quotient rule gives $\frac{d}{dx}\tanh x = 1 - \tanh^2 x$. The derivative is written in terms of the forward *output*, so the node only needs to remember $t$. Sigmoid has the same convenient property: $\sigma' = \sigma(1 - \sigma)$.

## Why `+=` and why a closure
The node can't compute its contribution when it's created, because `out.grad` isn't known yet: it depends on everything computed later. A closure postpones the work until the backward pass, when `out.grad` has been filled in. The `+=` matters as soon as `self` feeds more than one node: each consumer adds its share, and the total is the sum over paths (the multivariable chain rule).

## Saturation
For $|x| \gtrsim 20$, $\tanh x$ rounds to exactly $\pm 1$ and $1 - t^2 = 0$: the gradient vanishes. Test 3 shows it at $\pm 800$; test 5 at 20. This is the vanishing-gradient problem that made deep tanh networks hard to train and pushed the field toward ReLU and careful initialisation.

## Complexity
$O(1)$ per node, forward and backward.

## Common bugs
- `self.grad = ...` instead of `+=`.
- Using `1 - x**2` (the input) instead of `1 - t**2` (the output).
- Forgetting to multiply by `out.grad`, which silently assumes the upstream gradient is 1. Test 2 catches this.

## Interview follow-ups
- Write the backward rule for $\sigma(x)$ and for $\mathrm{ReLU}(x)$. What should ReLU's gradient be at exactly 0?
- Why does a deep stack of tanh layers with large weights train slowly?
"""
