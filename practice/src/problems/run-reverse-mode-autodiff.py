import numpy as np

META = dict(title='backward() for a scalar autograd engine', tags=['Neural Networks', 'Calculus'], diff='Hard',
            runtime='python', lessons=['air-fanout-v1:01.08', 'air-fanout-v1:01.07'],
            related=['topologically-sort-computation-dag', 'apply-local-vector-jacobian-rules',
                     'backpropagation-through-computation-graph', 'accumulate-shared-path-gradients'])

STATEMENT = r"""
Put the pieces together: a `Value` class whose `backward()` fills in the gradient of one output with respect to **every** node in its graph.

## What you write
1. The `_backward` closures for `+`, `*` and `tanh`. Each reads the node's own `.grad` and **adds** (`+=`) the local contribution to each operand's `.grad`.
2. `Value.backward()`, called on the output:
   - collect the nodes reachable from `self` in topological order (depth-first post-order with a visited set),
   - set `self.grad = 1.0`,
   - call `_backward()` on every node in **reverse** topological order.

Negation and subtraction are already defined in terms of `*` and `+` (`-v = v * -1`, `a - b = a + (-b)`), so they need no rules of their own.

## The harness
`autodiff(inputs, program)` (given) builds the graph from `inputs` (dict `{name: float}`) and `program` (lines `[name, op, arg]` / `[name, op, arg1, arg2]`, ops `"add"`, `"sub"`, `"mul"`, `"neg"`, `"tanh"`; arguments are names or numeric constants), calls `backward()` on the **last** line's node, and returns

```
{"value": output.data, "grads": {name: node.grad for every input and every line}}
```

A name may be used many times; its gradient is the sum over all uses. A line the output doesn't depend on keeps gradient `0.0`.
"""

STARTER = '''import math

def autodiff(inputs, program):
    env = {name: Value(x) for name, x in inputs.items()}
    for name, op, *args in program:
        env[name] = OPS[op](*[env[a] if isinstance(a, str) else a for a in args])
    out = env[program[-1][0]]
    out.backward()
    return {"value": out.data, "grads": {name: v.grad for name, v in env.items()}}


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
        # TODO: out._backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        # TODO: out._backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        # TODO: out._backward
        return out

    def backward(self):
        # TODO: topological order, seed, reverse sweep.
        raise NotImplementedError

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
'''

SOLUTION = '''import math

def autodiff(inputs, program):
    env = {name: Value(x) for name, x in inputs.items()}
    for name, op, *args in program:
        env[name] = OPS[op](*[env[a] if isinstance(a, str) else a for a in args])
    out = env[program[-1][0]]
    out.backward()
    return {"value": out.data, "grads": {name: v.grad for name, v in env.items()}}


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
                topo.append(v)            # after its operands
        build(self)
        self.grad = 1.0                   # d out / d out
        for v in reversed(topo):          # consumers before producers
            v._backward()

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
'''

HINTS = [
    'Start with the closures: `+` sends `out.grad` to both operands, `*` sends each operand the other one\'s `data` times `out.grad`, `tanh` sends `(1 - out.data**2) * out.grad`. Always `+=`.',
    'In `backward`, build the order with a nested recursive `build(v)` that skips visited nodes, recurses into `v._prev`, and appends `v` afterwards.',
    'Then `self.grad = 1.0` and `for v in reversed(topo): v._backward()`. Leaves keep the default no-op `_backward`.',
]


def _prog(rng, names, n):
    pool, prog = list(names), []
    for i in range(n):
        op = ["add", "sub", "mul", "mul", "neg", "tanh"][rng.integers(6)]
        pick = lambda: pool[rng.integers(max(0, len(pool) - 4), len(pool))]   # recent nodes: deeper graphs
        a = pick()
        if op in ("neg", "tanh"):
            prog.append([f"t{i}", op, a])
        else:
            b = pick() if rng.random() > 0.25 else float(round(rng.uniform(-2, 2), 1))
            prog.append([f"t{i}", op, a, b])
        pool.append(f"t{i}")
    return prog


_r = np.random.default_rng(37)
TESTS = [
    ({"a": 1.5, "b": -2.0, "c": 4.0}, [["m", "mul", "a", "b"], ["s", "add", "m", "c"]]),
    ({"x": 3.0}, [["y", "mul", "x", "x"], ["z", "add", "y", "x"]]),
    ({"x1": 1.0, "w1": -0.5, "x2": 2.0, "w2": 0.75, "b": 0.2},
     [["p1", "mul", "x1", "w1"], ["p2", "mul", "x2", "w2"], ["s", "add", "p1", "p2"], ["n", "add", "s", "b"], ["o", "tanh", "n"]]),
    ({"a": 1.5, "b": 0.5}, [["d", "sub", "a", "b"], ["e", "sub", 3.0, "d"], ["f", "neg", "e"], ["g", "mul", "f", "f"]]),
    ({"a": 1.0, "b": 2.0}, [["c", "add", "a", "b"], ["d", "mul", "c", "a"], ["e", "mul", "c", "b"], ["f", "add", "d", "e"]]),
    ({"x": 0.7, "k": 5.0}, [["dead", "mul", "k", "k"], ["y", "tanh", "x"]]),
    ({"x": 0.4}, [["a", "add", "x", "x"], ["b", "add", "a", "a"], ["c", "add", "b", "b"], ["d", "tanh", "c"]]),
    ({"a": 0.3, "b": -0.6, "c": 0.9}, _prog(_r, ["a", "b", "c"], 12)),
    ({"p": 0.5, "q": -0.2}, _prog(_r, ["p", "q"], 18)),
]

EXPLANATION = r"""
## The algorithm in one sentence
Visit the nodes from the output back to the inputs, and at each node turn its finished gradient into contributions to its operands using one local rule.

## Why every piece is needed
- **Topological order.** A node's gradient is the sum of contributions from all its consumers. In test 5, `c` feeds both `d` and `e`; if `c._backward` ran before `e._backward`, it would pass down only part of its gradient. Reverse topological order guarantees all consumers run first.
- **`+=`.** Test 2 uses `x` three times (twice in `y = x * x`, once in `z`). Its gradient is $2x + 1 = 7$, the sum of three contributions. Assignment would keep only the last one.
- **Seeding with 1.** $\partial\,\text{out}/\partial\,\text{out} = 1$ starts the chain; everything else is a product of local derivatives with that 1.
- **Reachability.** The DFS starts from the output, so `dead` in test 6 is never visited and keeps gradient 0, and so does its input `k`.

## Complexity
The forward pass, the DFS and the backward sweep are each $O(V + E)$. That's the remarkable part of reverse mode: the gradient with respect to *all* inputs costs a small constant times one evaluation, independent of the number of inputs.

## Common bugs
- Calling `_backward` in creation order instead of reverse order.
- Forgetting to seed the output's gradient, so everything stays 0.
- Visiting a shared node twice because the visited check is missing.
- In `*`, each operand getting its own `data` instead of the other's.

## Interview follow-ups
- Calling `backward()` twice without resetting gradients gives different answers. Why, and what do frameworks do about it?
- Forward mode (dual numbers) computes the derivative with respect to *one* input per pass. When would you prefer it to reverse mode?
"""
