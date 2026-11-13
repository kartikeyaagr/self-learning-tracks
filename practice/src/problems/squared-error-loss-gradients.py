import numpy as np

META = dict(title='Squared-error loss through a tanh neuron', tags=['Loss Functions', 'Neural Networks'], diff='Medium',
            runtime='python', lessons=['air-fanout-v1:05.07'],
            related=['gradient-check-scalar-neuron', 'apply-one-full-sgd-training-step', 'differentiate-exponentials-and-powers'])

STATEMENT = r"""
Training needs one scalar to minimise. For regression the usual choice is the squared error between predictions and targets. Build it from `Value`s and read off every gradient that matters.

The model is one tanh neuron shared by all $N$ samples:

$$p_n = \tanh\Big(b + \sum_i w_i X_{n,i}\Big), \qquad L = \frac{1}{c}\sum_{n=1}^{N} (p_n - y_n)^2, \qquad c = \begin{cases} 1 & \text{reduction} = \texttt{"sum"} \\ N & \text{reduction} = \texttt{"mean"} \end{cases}$$

Write `squared_error(w, b, X, y, reduction)` using the starter's `Value` engine (`+`, `-`, `*`, `/`, `**`, `tanh`, `backward()`) and return

```
{"loss": L, "preds": [p_n], "pred_grads": [dL/dp_n], "w_grad": [dL/dw_i], "b_grad": dL/db}
```

`pred_grads` are the gradients on the prediction nodes themselves, read after `backward()`.

## Inputs
`w`: list of $k \ge 1$ floats; `b`: float; `X`: list of $N \ge 1$ rows of length $k$; `y`: list of $N$ floats; `reduction`: `"sum"` or `"mean"`.
"""

HEAD_STARTER = '''import math

def squared_error(w, b, X, y, reduction):
    """Return {"loss", "preds", "pred_grads", "w_grad", "b_grad"}."""
    raise NotImplementedError
'''

HEAD_SOLUTION = '''import math

def squared_error(w, b, X, y, reduction):
    ws, bias = [Value(v) for v in w], Value(b)
    preds = [sum((wi * xi for wi, xi in zip(ws, row)), bias).tanh() for row in X]
    loss = sum((p - t) ** 2 for p, t in zip(preds, y))
    if reduction == "mean":
        loss = loss / len(X)
    loss.backward()
    return {"loss": loss.data, "preds": [p.data for p in preds], "pred_grads": [p.grad for p in preds],
            "w_grad": [v.grad for v in ws], "b_grad": bias.grad}
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

STARTER = HEAD_STARTER + ENGINE
SOLUTION = HEAD_SOLUTION + ENGINE

HINTS = [
    'Wrap only `w` and `b` in `Value`s, and keep the list of prediction nodes: you need their `.grad` after the backward pass.',
    'The loss is `sum((p - t) ** 2 for p, t in zip(preds, y))`, divided by `len(X)` for `"mean"`. Python\'s `sum` starts at 0, which the engine handles through `__radd__`.',
    'Analytically $\\partial L / \\partial p_n = 2(p_n - y_n)/c$. Use that to sanity-check `pred_grads`; the weight gradients then follow by multiplying by $(1 - p_n^2)$ and $X_{n,i}$.',
]

_r = np.random.default_rng(83)
_v = lambda *s: np.round(_r.uniform(-1, 1, size=s), 3).tolist()
TESTS = [
    ([0.5], 0.0, [[1.0], [-1.0]], [1.0, -1.0], "sum"),
    ([0.5], 0.0, [[1.0], [-1.0]], [1.0, -1.0], "mean"),
    ([0.0, 0.0], 0.0, [[1.0, 2.0]], [0.5], "sum"),
    ([1.0, -1.0], 0.2, [[0.3, 0.3], [2.0, -2.0], [0.0, 1.0]], [0.2, 0.9, -0.6], "mean"),
    ([20.0], 0.0, [[1.0]], [0.0], "sum"),
    (_v(3), float(round(_r.uniform(-1, 1), 3)), _v(6, 3), _v(6), "mean"),
    (_v(2), float(round(_r.uniform(-1, 1), 3)), _v(4, 2), _v(4), "sum"),
]

EXPLANATION = r"""
## The gradient of the squared error
For each sample, $\partial L / \partial p_n = 2(p_n - y_n)/c$: the error itself, scaled. Large errors pull hard, small errors barely move the model. Then the chain rule continues through the neuron:

$$\frac{\partial L}{\partial w_i} = \sum_n \frac{2(p_n - y_n)}{c}\,(1 - p_n^2)\, X_{n,i}, \qquad \frac{\partial L}{\partial b} = \sum_n \frac{2(p_n - y_n)}{c}\,(1 - p_n^2)$$

The weights are shared across samples, so their gradients are sums over the batch; the engine does that automatically because every sample's graph points at the same weight nodes.

## Sum vs mean
The two reductions give the same direction and differ only by the factor $N$ (tests 1 and 2). That factor matters in practice: with `"sum"` the effective learning rate grows with the batch size, so `"mean"` is the default in most frameworks.

## Saturation hurts
In test 5 the pre-activation is 20, the prediction is 1 while the target is 0: the error is as large as it gets, yet the weight gradient is essentially zero because of the $(1 - p^2)$ factor. A saturated tanh output combined with squared error learns very slowly from its worst mistakes, one reason classification uses cross-entropy on logits instead.

## Complexity
$O(Nk)$ nodes and time for the forward and backward passes.

## Common bugs
- Building a fresh `Value` for the weights per sample, so the gradients are split across copies.
- Squaring with `abs` or forgetting the square entirely.
- Dividing by $N$ for `"sum"`, or not dividing for `"mean"`.

## Interview follow-ups
- Why is MSE the maximum-likelihood loss for Gaussian noise, and what loss corresponds to Laplace noise?
- How does the Huber loss change the gradient for large errors?
"""
