import numpy as np

META = dict(title='Shared nodes, stale gradients and zero_grad', tags=['Neural Networks', 'Training'], diff='Medium',
            runtime='python', lessons=['air-fanout-v1:01.08', 'air-fanout-v1:01.07'],
            related=['run-reverse-mode-autodiff', 'gradient-accumulation', 'apply-one-full-sgd-training-step'])

STATEMENT = r"""
The starter's autograd engine has a bug that only appears when a node is used more than once, and it's missing the method that every training loop calls before `backward()`. Fix both, then observe what stale gradients do.

## 1. Fix accumulation
Each `_backward` closure in the starter **assigns** to its operands' gradients (`self.grad = ...`). When a node feeds several consumers (or the same consumer twice, as in `x * x`), each assignment overwrites the previous contribution. The correct rule is the multivariable chain rule: a node's gradient is the **sum** of what all its consumers send it. Make every closure accumulate.

## 2. Add `zero_grad()`
`root.zero_grad()` sets `.grad = 0.0` on **every** node reachable from `root`, including `root` itself.

## 3. The experiment
`backward()` is unchanged: it seeds `self.grad = 1.0` (overwriting, not adding) and runs the closures in reverse topological order. `shared_grads(inputs, program)` (given) builds the graph from the usual program format (ops `"add"`, `"sub"`, `"mul"`, `"neg"`, `"tanh"`; arguments are names or numeric constants; the output is the last line) and returns

```
{"first": grads after one backward(),
 "second": grads after a second backward() with no reset in between,
 "after_zero": grads after zero_grad() and then backward() again}
```

each a dict `{name: grad}` over every input and line. Work out on paper why `"second"` is generally **not** twice `"first"`.
"""

STARTER = '''import math

def shared_grads(inputs, program):
    env = {name: Value(x) for name, x in inputs.items()}
    for name, op, *args in program:
        env[name] = OPS[op](*[env[a] if isinstance(a, str) else a for a in args])
    out = env[program[-1][0]]
    snap = lambda: {name: v.grad for name, v in env.items()}
    out.backward()
    first = snap()
    out.backward()
    second = snap()
    out.zero_grad()
    out.backward()
    return {"first": first, "second": second, "after_zero": snap()}


OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
       "neg": lambda a: -a, "tanh": lambda a: a.tanh()}


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
            self.grad = out.grad                       # BUG
            other.grad = out.grad                      # BUG
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad = other.data * out.grad          # BUG
            other.grad = self.data * out.grad          # BUG
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _backward():
            self.grad = (1 - out.data ** 2) * out.grad  # BUG
        out._backward = _backward
        return out

    def _topo(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        return topo

    def backward(self):
        self.grad = 1.0
        for v in reversed(self._topo()):
            v._backward()

    # TODO: zero_grad()

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
'''

SOLUTION = '''import math

def shared_grads(inputs, program):
    env = {name: Value(x) for name, x in inputs.items()}
    for name, op, *args in program:
        env[name] = OPS[op](*[env[a] if isinstance(a, str) else a for a in args])
    out = env[program[-1][0]]
    snap = lambda: {name: v.grad for name, v in env.items()}
    out.backward()
    first = snap()
    out.backward()
    second = snap()
    out.zero_grad()
    out.backward()
    return {"first": first, "second": second, "after_zero": snap()}


OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
       "neg": lambda a: -a, "tanh": lambda a: a.tanh()}


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
            self.grad += out.grad                      # sum over every use, not last-writer-wins
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

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def _topo(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        return topo

    def backward(self):
        self.grad = 1.0
        for v in reversed(self._topo()):
            v._backward()

    def zero_grad(self):
        for v in self._topo():
            v.grad = 0.0

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
'''

HINTS = [
    'Try `x * x` by hand with the buggy rule: the second assignment overwrites the first, so you get $x$ instead of $2x$. Change every `=` on a `.grad` inside a closure to `+=`.',
    '`zero_grad` needs the same set of nodes that `backward` visits. The starter already has `_topo()`; loop over it and reset each `.grad`.',
    'For the "second" column, trace a two-step chain $y = 2x$, $z = 3y$: the second pass leaves $y$ with $3 + 3 = 6$, and $y$ then sends $6 \\cdot 2$ to $x$ on top of the stale 6, giving 18, not 12.',
]

_r = np.random.default_rng(53)


def _prog(rng, names, n):
    pool, prog = list(names), []
    for i in range(n):
        op = ["add", "sub", "mul", "mul", "tanh"][rng.integers(5)]
        a = pool[rng.integers(len(pool))]
        if op == "tanh":
            prog.append([f"t{i}", op, a])
        else:
            prog.append([f"t{i}", op, a, pool[rng.integers(len(pool))]])
        pool.append(f"t{i}")
    return prog


TESTS = [
    ({"x": 3.0}, [["y", "mul", "x", "x"]]),
    ({"x": 1.0}, [["y", "mul", 2.0, "x"], ["z", "mul", "y", 3.0]]),
    ({"a": 1.0, "b": 2.0}, [["c", "add", "a", "b"], ["d", "mul", "c", "a"], ["e", "mul", "c", "b"], ["f", "add", "d", "e"]]),
    ({"x": 0.5}, [["a", "add", "x", "x"], ["b", "add", "a", "x"], ["t", "tanh", "b"]]),
    ({"a": 2.0, "b": -1.0}, [["s", "add", "a", "b"]]),
    ({"x": 0.3, "w": -0.8}, [["p", "mul", "x", "w"], ["q", "sub", "p", "x"], ["r", "mul", "q", "q"], ["o", "sub", "r", "w"]]),
    ({"a": 0.4, "b": -0.3, "c": 0.7}, _prog(_r, ["a", "b", "c"], 10)),
]

EXPLANATION = r"""
## Why gradients must add
If a node $v$ feeds consumers $u_1, \dots, u_m$, then a small change in $v$ moves the output through every one of them, and the multivariable chain rule sums the effects:

$$\frac{\partial L}{\partial v} = \sum_{j} \frac{\partial L}{\partial u_j}\,\frac{\partial u_j}{\partial v}$$

With `=`, only the last consumer to run is counted. Test 1 is the smallest case: $y = x \cdot x$ uses $x$ in both slots and must give $2x = 6$, not 3. Test 3 shares `c` between two products.

## What a second backward() does
`backward()` overwrites only the output's gradient. Every other node keeps its old value, and the new contributions are added on top. Worse, an intermediate node that already holds a stale gradient passes the *inflated* total further down. In test 2 ($y = 2x$, $z = 3y$) the first pass gives $x$: 6; the second leaves $y$ at $3 + 3 = 6$ and sends $2 \cdot 6 = 12$ more to $x$, for 18 in total, not 12. Only nodes one step from the output (test 5, a single addition) end up exactly doubled.

## Why training loops call zero_grad
Accumulation is what you want *within* one backward pass and sometimes across passes (summing gradients over micro-batches). Across optimisation steps it's a bug: each step's update would include every previous gradient. So every step does `zero_grad()` → forward → `backward()` → update. `"after_zero"` equals `"first"` because zeroing restores the fresh state exactly.

## Complexity
`zero_grad` is one $O(V + E)$ traversal, the same as `backward`.

## Common bugs
- Fixing `+=` in some closures but not all (here `tanh` is easy to miss).
- Zeroing only the leaves: intermediate nodes still carry stale gradients into the next pass.
- Zeroing *after* backward instead of before the next one, then reading the gradients.

## Interview follow-ups
- PyTorch frees the graph after `backward()` by default. What error do you get if you call it twice, and what does `retain_graph=True` change?
- How does gradient accumulation over $k$ micro-batches reproduce the gradient of one large batch, and where does the $1/k$ go?
"""
