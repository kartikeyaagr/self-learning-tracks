import numpy as np

META = dict(title='Train a tiny MLP with your own autograd', tags=['Training', 'Neural Networks'], diff='Hard',
            runtime='python', lessons=['air-fanout-v1:05.07', 'air-fanout-v1:05.05'],
            related=['apply-one-full-sgd-training-step', 'accumulate-shared-path-gradients', 'evaluate-multi-layer-perceptron'])

STATEMENT = r"""
The finish line: train a small network end to end with the scalar engine you've built.

The starter has the `Value` engine and the modules: `MLP(nin, sizes, params)` (flat params in layout order: layer, neuron, weights then bias; tanh on hidden layers, linear output), `model(x)` returning the output list, `model.parameters()` and `model.zero_grad()`. Here `sizes[-1] == 1`.

Write `train(nin, sizes, params, X, y, lr, steps)`. Build the model once from `params`, then for $t = 0, 1, \dots, \text{steps} - 1$:

1. Forward every sample and form the mean squared error $L_t = \frac1N \sum_n (f(x_n) - y_n)^2$, where $f(x_n)$ is `model(x_n)[0]`.
2. `zero_grad()`, then `backward()` from $L_t$.
3. Update each parameter with a linearly decaying learning rate:
$$\eta_t = \text{lr} \cdot \left(1 - 0.9\,\frac{t}{\text{steps}}\right), \qquad \theta \leftarrow \theta - \eta_t\, \frac{\partial L_t}{\partial \theta}$$

Return

```
{"losses": [L_0, ..., L_{steps-1}], "final_loss": MSE after the last update, "preds": [f(x_n) after the last update]}
```

With `steps == 0` the model is not updated: `losses` is empty, and `final_loss`/`preds` describe the initial parameters.

## Inputs
`X`: $N \ge 1$ rows of length `nin`; `y`: $N$ floats; `lr` $> 0$; `steps` $\ge 0$.
"""

HEAD_STARTER = '''import math

def train(nin, sizes, params, X, y, lr, steps):
    """Full-batch gradient descent with lr decaying from lr to 0.1*lr. Return {"losses", "final_loss", "preds"}."""
    raise NotImplementedError
'''

HEAD_SOLUTION = '''import math

def train(nin, sizes, params, X, y, lr, steps):
    model = MLP(nin, sizes, params)

    def mse():
        preds = [model(x)[0] for x in X]
        loss = sum((p - t) ** 2 for p, t in zip(preds, y)) / len(X)
        return loss, preds

    losses = []
    for t in range(steps):
        loss, _ = mse()                       # a fresh graph every step
        model.zero_grad()                     # without this, gradients pile up across steps
        loss.backward()
        eta = lr * (1 - 0.9 * t / steps)
        for p in model.parameters():
            p.data -= eta * p.grad
        losses.append(loss.data)

    loss, preds = mse()
    return {"losses": losses, "final_loss": loss.data, "preds": [p.data for p in preds]}
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
    def __truediv__(self, other): return self * other ** -1
'''

MODULES = '''

class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, nin, take, nonlin=True):
        self.w = [Value(take()) for _ in range(nin)]
        self.b = Value(take())
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, take, nonlin=True):
        self.neurons = [Neuron(nin, take, nonlin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """len(sizes) layers; tanh on hidden layers, linear output. Flat params: layer, neuron, weights then bias."""

    def __init__(self, nin, sizes, params):
        it = iter(params)
        take = lambda: next(it)
        dims = [nin] + list(sizes)
        self.layers = [Layer(dims[i], dims[i + 1], take, nonlin=(i < len(sizes) - 1)) for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
'''

STARTER = HEAD_STARTER + MODULES + ENGINE
SOLUTION = HEAD_SOLUTION + MODULES + ENGINE

HINTS = [
    'Build the model once, outside the loop. Inside the loop rebuild only the loss graph: the parameters are the same `Value` objects every step, but the intermediate nodes are new.',
    'The order inside each step is: loss, `zero_grad()`, `backward()`, update. Record `loss.data` before or after updating; it is the loss at the parameters the step started from.',
    'Use `eta = lr * (1 - 0.9 * t / steps)` with `t` counting from 0, so the first step uses exactly `lr`. After the loop, one more forward pass gives `final_loss` and `preds`.',
]


def _init(rng, nin, sizes):
    dims = [nin] + sizes
    k = sum((dims[i] + 1) * dims[i + 1] for i in range(len(sizes)))
    return np.round(rng.uniform(-1, 1, size=k), 3).tolist()


_r = np.random.default_rng(97)
_X4 = [[1.0, -0.5, 2.0], [-1.5, 1.0, 0.5], [0.5, 2.0, -1.0], [-0.5, -1.0, -1.5]]
_Xr = np.round(_r.uniform(-1, 1, size=(8, 2)), 2).tolist()
TESTS = [
    (1, [1], [0.0, 0.0], [[1.0], [2.0]], [2.0, 4.0], 0.1, 3),
    (2, [2, 1], [0.5, -0.5, 0.0, 0.3, 0.8, 0.0, 1.0, -1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [1.0, -1.0], 0.2, 5),
    (1, [1], [0.5, 0.5], [[1.0]], [0.0], 0.1, 0),
    (1, [2, 1], [1.0, 0.0, -1.0, 0.0, 0.5, 0.5, 0.0], [[0.0], [1.0], [-1.0]], [0.0, 0.5, 0.5], 0.3, 10),
    (3, [3, 2, 1], _init(_r, 3, [3, 2, 1]), _X4, [1.0, -1.0, 1.0, -1.0], 0.1, 20),
    (2, [3, 1], _init(_r, 2, [3, 1]), _Xr, [float(round(a * b * 2, 3)) for a, b in _Xr], 0.2, 30),
]

EXPLANATION = r"""
## What one step really does
Each iteration builds a brand-new graph from the same parameter objects, backpropagates through it once, and nudges the parameters. The graph is thrown away; the parameters, and only the parameters, carry information to the next step. That's the define-by-run training loop of every modern framework, written in about fifteen lines.

## Why zero_grad sits inside the loop
`backward()` *adds* into `.grad`. Without resetting, step $t$ would update with $\sum_{s \le t} g_s$, a sum of stale gradients that keeps growing. In test 1 that makes the loss climb back up after two steps and finish around 3.5 instead of 0.22; in test 5 it ends about seven times higher than with zeroing. Zeroing right before `backward()` is the standard order.

## Why decay the learning rate
A large rate makes fast progress early but bounces around the minimum later; shrinking it lets the parameters settle. The linear schedule here goes from `lr` at $t = 0$ down toward $0.1 \cdot \text{lr}$. Warm-up plus cosine decay is the modern default for large models, but the idea is the same.

## Reading the tests
- Test 1 is linear regression ($y = 2x$) and the loss drops every step.
- Test 3 has `steps = 0`: nothing changes, `losses` is empty.
- Test 6 fits $y = 2ab$, which a linear model can't represent; the hidden tanh layer is what makes progress possible.

## Complexity
Each step costs one forward and one backward pass, $O(N \cdot P)$ for $N$ samples and $P$ parameters; the whole run is $O(\text{steps} \cdot N \cdot P)$, all in pure Python.

## Common bugs
- Building the model inside the loop, which resets the parameters every step.
- Computing `eta` with `t` starting at 1 or with integer division.
- Updating with `p.data -= eta * p.grad` *before* `backward()` (using last step's gradients).

## Interview follow-ups
- Replace full-batch descent with mini-batches of size 2. What changes in the loss curve, and why can noise help?
- How would you add momentum to this loop? What extra state does each parameter need?
"""
