import numpy as np

META = dict(title='One SGD step on a Value MLP with weight decay', tags=['Training', 'Neural Networks', 'Optimization'],
            diff='Hard', runtime='python', lessons=['air-fanout-v1:05.07'],
            related=['squared-error-loss-gradients', 'train-tiny-micrograd-mlp', 'collect-module-parameters-recursively'])

STATEMENT = r"""
A training step is four moves in a fixed order: forward, loss, backward, update. Do one complete step on a scalar MLP.

The starter has the `Value` engine and finished modules: `MLP(nin, sizes, params)` builds a network from a flat parameter list (layer by layer, neuron by neuron, weights then bias; tanh on hidden layers, linear output), `model(x)` returns the list of outputs, and `model.parameters()` / `model.zero_grad()` work as usual. Here `sizes[-1] == 1`, so each prediction is `model(x)[0]`.

## The loss
Mean squared error over the $N$ samples plus L2 weight decay on **every** parameter:

$$L(\theta) = \frac{1}{N} \sum_{n} \big(f_\theta(x_n) - y_n\big)^2 + \alpha \sum_{j} \theta_j^2$$

## What `sgd_step(nin, sizes, params, X, y, lr, alpha)` does
1. Build the model from `params`, compute $L$, call `zero_grad()`, then `backward()`.
2. Update every parameter in place: $\theta_j \leftarrow \theta_j - \eta\, \partial L/\partial \theta_j$.
3. Recompute the loss at the new parameters (same formula, a fresh forward pass).

Return

```
{"loss": L before the step, "grads": [dL/dtheta_j], "params": [updated theta_j], "loss_after": L after the step}
```

with `grads` and `params` in the layout order.

## Inputs
`X`: list of $N \ge 1$ rows of length `nin`; `y`: list of $N$ floats; `lr` $> 0$; `alpha` $\ge 0$.
"""

HEAD_STARTER = '''import math

def sgd_step(nin, sizes, params, X, y, lr, alpha):
    """Return {"loss", "grads", "params", "loss_after"} for one SGD step on MSE + alpha * sum(p**2)."""
    raise NotImplementedError
'''

HEAD_SOLUTION = '''import math

def sgd_step(nin, sizes, params, X, y, lr, alpha):
    model = MLP(nin, sizes, params)

    def objective():
        mse = sum((model(x)[0] - t) ** 2 for x, t in zip(X, y)) / len(X)
        return mse + alpha * sum(p * p for p in model.parameters())

    loss = objective()
    model.zero_grad()                      # a no-op on a fresh model, but the habit matters in a loop
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    for p in model.parameters():
        p.data -= lr * p.grad              # update the live Values in place
    return {"loss": loss.data, "grads": grads, "params": [p.data for p in model.parameters()],
            "loss_after": objective().data}
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
    'Write the objective as a small inner function so you can call it once before and once after the update. Each call does a fresh forward pass over all samples.',
    'Build the MSE as `sum((model(x)[0] - t) ** 2 for x, t in zip(X, y)) / len(X)` and the penalty as `alpha * sum(p * p for p in model.parameters())`. Both are `Value`s, so their sum backpropagates into every parameter.',
    'Order: loss, `zero_grad()`, `backward()`, copy the grads, then `p.data -= lr * p.grad` for each parameter. Read `p.data` for the new parameters and evaluate the objective again for `loss_after`.',
]


def _case(rng, nin, sizes, n, lr, alpha):
    dims = [nin] + sizes
    k = sum((dims[i] + 1) * dims[i + 1] for i in range(len(sizes)))
    return (nin, sizes, np.round(rng.uniform(-1, 1, size=k), 3).tolist(), np.round(rng.uniform(-1, 1, size=(n, nin)), 3).tolist(),
            np.round(rng.uniform(-1, 1, size=n), 3).tolist(), lr, alpha)


_r = np.random.default_rng(89)
TESTS = [
    (1, [1], [0.5, 0.0], [[1.0], [2.0]], [1.0, 2.0], 0.1, 0.0),
    (1, [1], [0.5, 0.0], [[1.0], [2.0]], [1.0, 2.0], 0.1, 0.5),
    (2, [2, 1], [0.1, 0.2, 0.0, -0.3, 0.4, 0.0, 1.0, -1.0, 0.5], [[1.0, 0.0], [0.0, 1.0]], [1.0, -1.0], 0.5, 0.0),
    (1, [1], [2.0, 1.0], [[0.0]], [0.0], 0.25, 0.0),
    (1, [1], [3.0, 0.0], [[1.0]], [0.0], 2.0, 0.0),
    _case(_r, 3, [4, 4, 1], 4, 0.05, 0.001),
    _case(_r, 2, [3, 1], 6, 0.2, 0.01),
]

EXPLANATION = r"""
## The four moves
- **Forward + loss** builds one graph that covers every sample and the penalty, ending in a single scalar.
- **zero_grad** clears whatever a previous step left in `.grad`. On a freshly built model it changes nothing, but in a loop forgetting it means every update includes all previous gradients.
- **backward** fills `.grad` on every parameter. Because each weight appears in every sample's sub-graph and in the penalty, its gradient is the sum of all those contributions.
- **update** subtracts `lr * grad` from `.data`. It must touch the live parameter objects; the next forward pass reads them.

## What weight decay adds
The penalty contributes $2\alpha\theta_j$ to each gradient, so the update becomes $\theta_j \leftarrow (1 - 2\eta\alpha)\theta_j - \eta\,\partial\,\text{MSE}/\partial\theta_j$: every step shrinks the weights a little toward zero. Compare tests 1 and 2, which differ only in $\alpha$.

## Does the loss go down?
For a small enough learning rate, yes: $L(\theta - \eta g) \approx L(\theta) - \eta \lVert g \rVert^2$. Test 5 uses $\eta = 2$ on a steep linear model and the loss after the step is *larger*: the step overshoots the minimum. Test 4 has an input of 0, so the weight gets no gradient and only the bias moves.

## Complexity
Two forward passes and one backward pass, each $O(N \cdot P)$ for $P$ parameters.

## Common bugs
- Updating parameters **before** reading the gradients you return, or reading `p.grad` after a second backward.
- Forgetting the penalty in `loss_after`, which makes the before/after comparison meaningless.
- Applying weight decay only to weights and not biases. That's a common real-world choice, but not what this statement specifies.

## Interview follow-ups
- AdamW decouples weight decay from the gradient. How does its update differ from adding $\alpha\lVert\theta\rVert^2$ to the loss under Adam?
- Why is it common *not* to decay biases and normalisation gains?
"""
