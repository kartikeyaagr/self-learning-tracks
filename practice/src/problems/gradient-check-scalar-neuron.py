import numpy as np

META = dict(title='Gradient-check a neuron\'s squared error', tags=['Neural Networks', 'Calculus'], diff='Medium',
            runtime='python', lessons=['air-fanout-v1:01.08', 'air-fanout-v1:05.01'],
            related=['backpropagate-through-scalar-neuron', 'check-product-chain-gradient', 'gradient-check-autodiff-graph'])

STATEMENT = r"""
Hand-written backward rules are easy to get subtly wrong, so you check them against finite differences. Do that for one tanh neuron trained with a squared error:

$$o = \tanh\Big(b + \sum_i w_i x_i\Big), \qquad L = (o - y)^2$$

The starter has a complete `Value` engine (`+`, `-`, `*`, `tanh` and `backward()`). Write `check_neuron(x, w, b, y, h)` that returns

```
{"loss": L, "analytic": [dL/dw_0, ..., dL/dw_{k-1}, dL/db], "numeric": [...same order...], "max_rel_error": float}
```

- `analytic`: from the engine. Build $L$ from `Value`s exactly as written, call `L.backward()`, read the parameter gradients. Only `w` and `b` are parameters; `x` and `y` are plain data.
- `numeric`: central differences $\dfrac{L(\theta + h e_j) - L(\theta - h e_j)}{2h}$ for each parameter $\theta_j$, in the same order.
- `max_rel_error`: $\max_j \dfrac{|a_j - \hat a_j|}{\max(|a_j| + |\hat a_j|,\ 10^{-12})}$ over the parameters ($a$ analytic, $\hat a$ numeric).

## Inputs
`x`, `w`: lists of $k \ge 1$ floats. `b`, `y`: floats. `h`: positive float (tests use $h \ge 0.01$, so the truncation error of the central difference is visible in `max_rel_error`).
"""

STARTER = '''import math

def check_neuron(x, w, b, y, h):
    """Return {"loss", "analytic", "numeric", "max_rel_error"} for L = (tanh(w.x + b) - y)**2."""
    raise NotImplementedError


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

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
'''

SOLUTION = '''import math

def check_neuron(x, w, b, y, h):
    # Analytic: build L from Values and backpropagate.
    ws, bias = [Value(v) for v in w], Value(b)
    n = bias
    for xi, wi in zip(x, ws):
        n = n + wi * xi
    d = n.tanh() - y
    L = d * d
    L.backward()
    analytic = [v.grad for v in ws] + [bias.grad]

    # Numeric: plain-float loss, one parameter shifted at a time.
    def loss(params):
        *wv, bv = params
        return (math.tanh(bv + sum(wi * xi for wi, xi in zip(wv, x))) - y) ** 2

    theta = list(w) + [b]
    numeric = []
    for j in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        numeric.append((loss(up) - loss(down)) / (2 * h))

    rel = max(abs(a - g) / max(abs(a) + abs(g), 1e-12) for a, g in zip(analytic, numeric))
    return {"loss": L.data, "analytic": analytic, "numeric": numeric, "max_rel_error": rel}


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

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
'''

HINTS = [
    'Two independent computations: one graph of `Value`s for the analytic gradient, and a plain-float function `loss(w, b)` you can call many times for the numeric one.',
    'Wrap only `w` and `b` in `Value`s (multiply a `Value` weight by a float input) and square with `d * d`. After `L.backward()`, the gradients are on those leaf nodes.',
    'For the numeric gradient, flatten the parameters into one list `w + [b]`, copy it, add or subtract `h` at index `j`, and evaluate the plain loss twice. Then take the max of the relative errors.',
]

_r = np.random.default_rng(29)
_v = lambda k, s=1.0: [float(round(t, 3)) for t in _r.uniform(-s, s, size=k)]
TESTS = [
    ([1.0, -2.0], [0.5, 0.25], 0.3, 1.0, 0.01),
    ([2.0], [0.0], 0.0, -0.5, 0.1),
    ([0.0, 1.0], [0.7, -0.3], 0.2, 0.0, 0.01),
    ([1.5, 0.5, -1.0], [0.2, -0.4, 0.6], -0.1, 0.5, 0.05),
    ([1.0], [0.3], 0.0, 0.28, 0.01),
    (_v(5), _v(5), float(round(_r.uniform(-1, 1), 3)), float(round(_r.uniform(-1, 1), 3)), 0.01),
    (_v(6, 2.0), _v(6, 0.5), 0.1, 0.8, 0.02),
]

EXPLANATION = r"""
## What a gradient check proves
The engine and the finite difference compute the same quantity in two unrelated ways: one by composing local backward rules, the other by literally measuring the slope. If they agree to a few digits at random points, a bug in a backward rule is very unlikely. If they disagree, the relative error points at the parameter whose rule is wrong.

## Reading the relative error
The central difference has truncation error $O(h^2)$, so with $h = 0.01$ expect errors around $10^{-5}$ to $10^{-8}$, not zero. A wrong rule shows up as an error of order 1. The relative form makes the check scale-free, and the floor in the denominator handles parameters with zero gradient: in test 3 the first input is 0, so both estimates of that weight's gradient are exactly 0 and the ratio is 0, not $0/0$.

Test 5 puts $y$ close to the neuron's output, so the loss is near its minimum and the gradients are small. Compare its relative error with test 1: the absolute truncation error barely changes, but it's divided by a smaller gradient. At an exact optimum the true gradient is zero and the ratio is pure rounding noise, which is why practical checks are done at random points, not at a minimum.

## Complexity
The analytic gradient costs one forward and one backward pass. The numeric one costs $2(k+1)$ forward passes. For $k$ in the millions that's why you check a few random coordinates, not all of them.

## Common bugs
- Reusing the same `Value` graph for the perturbed losses: its `data` is fixed at creation, so changing a Python float afterwards does nothing.
- Shifting a parameter in place and forgetting to undo it before the next coordinate.
- Computing $\partial L/\partial x$ instead of $\partial L/\partial w$.

## Interview follow-ups
- Why use float64 and a central difference for gradient checks rather than float32 and a forward difference?
- A check passes everywhere except at a few points. What kind of operation could explain that? (A kink, such as ReLU or `max`, where the function isn't differentiable.)
"""
