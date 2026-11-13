import numpy as np

META = dict(title='Gradient-check a whole autograd graph', tags=['Neural Networks', 'Calculus'], diff='Hard',
            runtime='python', lessons=['air-fanout-v1:01.08'],
            related=['run-reverse-mode-autodiff', 'gradient-check-scalar-neuron', 'check-product-chain-gradient'])

STATEMENT = r"""
Every new op you add to an autograd engine deserves a test that doesn't trust the engine. Add three ops, then build a checker that compares `backward()` against central differences on **any** program.

## Part 1: three new ops
The starter's `Value` already has `+`, `*`, `tanh`, the derived `-`/negation, and a working `backward()`. Add, each with its forward value and a `_backward` closure that uses `+=`:

| method | forward | local derivative |
|---|---|---|
| `v.exp()` | $e^v$ | $e^v$ |
| `v.relu()` | $\max(0, v)$ | $1$ if $v > 0$, else $0$ |
| `v ** k` (`k` an int or float constant) | $v^k$ | $k\,v^{k-1}$ |

## Part 2: the checker
Programs use the format from earlier problems: `inputs` is a dict `{name: float}`; `program` is a list of lines `[name, op, arg]` / `[name, op, arg1, arg2]` with ops `"add"`, `"sub"`, `"mul"`, `"neg"`, `"tanh"`, `"exp"`, `"relu"` and `"pow"` (`["y", "pow", "x", 3]` is $y = x^3$; the exponent is always a constant). Arguments are names or numeric constants; the output is the last line.

Write `grad_check(inputs, program, h)` returning three dicts keyed by **input** name (in the order of `inputs`):

- `"analytic"`: the gradient of the output from `backward()`.
- `"numeric"`: $\dfrac{f(\ldots, x + h, \ldots) - f(\ldots, x - h, \ldots)}{2h}$, where $f$ re-runs the program with only that input shifted.
- `"rel_error"`: $\dfrac{|a - \hat a|}{\max(|a| + |\hat a|,\ 10^{-8})}$.

Tests keep every op inside its domain. One test deliberately puts a `relu` exactly at its kink.
"""

STARTER = '''import math

def grad_check(inputs, program, h):
    """Return {"analytic": {...}, "numeric": {...}, "rel_error": {...}} keyed by input name."""
    raise NotImplementedError


OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "neg": lambda a: -a,
       "tanh": lambda a: a.tanh(), "exp": lambda a: a.exp(), "relu": lambda a: a.relu(), "pow": lambda a, k: a ** k}


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

    # TODO: exp(), relu(), __pow__(k)

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

SOLUTION = '''import math

def grad_check(inputs, program, h):
    def run(values):
        env = {name: Value(x) for name, x in values.items()}
        for name, op, *args in program:
            env[name] = OPS[op](*[env[a] if isinstance(a, str) else a for a in args])
        return env, env[program[-1][0]]

    env, out = run(inputs)
    out.backward()
    analytic = {name: env[name].grad for name in inputs}

    numeric, rel = {}, {}
    for name in inputs:
        # A Value's data is fixed when it is created, so each shifted point needs a fresh graph.
        up = run({**inputs, name: inputs[name] + h})[1].data
        down = run({**inputs, name: inputs[name] - h})[1].data
        numeric[name] = (up - down) / (2 * h)
        a, g = analytic[name], numeric[name]
        rel[name] = abs(a - g) / max(abs(a) + abs(g), 1e-8)
    return {"analytic": analytic, "numeric": numeric, "rel_error": rel}


OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "neg": lambda a: -a,
       "tanh": lambda a: a.tanh(), "exp": lambda a: a.exp(), "relu": lambda a: a.relu(), "pow": lambda a, k: a ** k}


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

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad          # d/dx e^x = e^x, already computed
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
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

HINTS = [
    'The three ops follow the `tanh` template exactly: compute the forward number, create `out` with `_prev=(self,)`, and attach a closure that does `self.grad += local * out.grad`.',
    'For the checker, write an inner `run(values)` that builds a fresh graph from a dict of input values and returns the nodes. Call it once and `backward()` for the analytic gradients.',
    'For each input, call `run` with `{**inputs, name: inputs[name] + h}` and with `- h`, and take the output `data` of each. Reusing the first graph doesn\'t work, because changing a leaf\'s `data` doesn\'t recompute the nodes built from it.',
]

_r = np.random.default_rng(41)
_rand_in = lambda: {k: float(round(_r.uniform(0.2, 1.2), 3)) for k in "abc"}
TESTS = [
    ({"x": 2.0}, [["y", "pow", "x", 3]], 0.01),
    ({"x": 0.0}, [["y", "relu", "x"]], 0.01),
    ({"a": 0.5, "b": -1.0}, [["e", "exp", "a"], ["m", "mul", "e", "b"], ["t", "tanh", "m"]], 0.01),
    ({"a": 1.0, "b": 2.0, "unused": 3.0}, [["c", "add", "a", "b"], ["d", "pow", "c", 2], ["o", "sub", "d", "a"]], 0.001),
    ({"x": 1.0}, [["s", "sub", "x", 1.0], ["r", "relu", "s"], ["o", "add", "r", "x"]], 0.01),
    ({"x": 4.0, "y": 0.5}, [["r", "pow", "x", 0.5], ["q", "pow", "y", -1], ["o", "mul", "r", "q"]], 0.01),
    ({"w": 0.8, "x": -0.6, "b": 0.1}, [["p", "mul", "w", "x"], ["n", "add", "p", "b"], ["r", "relu", "n"], ["t", "tanh", "n"], ["o", "add", "r", "t"]], 0.01),
    (_rand_in(), [["u", "mul", "a", "b"], ["v", "exp", "u"], ["w", "pow", "c", 3], ["z", "sub", "v", "w"], ["o", "tanh", "z"]], 0.01),
    (_rand_in(), [["u", "add", "a", "b"], ["v", "mul", "u", "c"], ["w", "pow", "v", 2], ["z", "mul", "w", "a"], ["e", "exp", "z"], ["o", "sub", 1.0, "e"]], 0.001),
]

EXPLANATION = r"""
## What the checker catches
Analytic and numeric gradients come from independent code paths, so a wrong backward rule (a missing factor $k$ in `pow`, `=` instead of `+=`, a sign error) shows up as a relative error of order 1, while correct rules give errors of order $h^2$ (here about $10^{-5}$ or smaller). In test 4, `unused` gets 0 from both methods, and the floor in the denominator turns $0/0$ into 0.

## The kinks
Tests 2 and 5 put `relu` exactly at 0. The engine uses slope 0 there, but the central difference straddles the kink and measures $\frac{h - 0}{2h} = 0.5$. The relative error is $1$ in test 2, which a real checker would flag. It's not a bug in the engine: the function simply has no derivative at that point. Production gradient checkers avoid this by sampling random points, or by skipping coordinates where an input is within $h$ of a kink.

## Why a fresh graph per evaluation
A `Value` stores a number computed once, when the node was created. There's no "re-run" button: to evaluate at a shifted input you rebuild the graph from the program. Define-by-run frameworks work the same way: each forward pass builds a new graph.

## Complexity
One forward and one backward pass for the analytic gradients, then two forward passes per input: $O(n \cdot (V + E))$ for $n$ inputs. That ratio is why finite differences are a test, not a training method.

## Common bugs
- `exp` backward using `math.exp(out.data)` (that's $e^{e^x}$).
- `pow` backward missing the factor $k$, or using $v^k$ instead of $v^{k-1}$.
- `relu` backward reading `out.grad > 0` instead of the input's sign.
- Shifting all inputs at once instead of one at a time.

## Interview follow-ups
- Why do gradient checks run in float64? What happens to the best $h$ in float32? (Rounding error grows like $\varepsilon/h$, so $h$ must be larger and the check gets looser.)
- How would you check a *Hessian-vector product* with finite differences of gradients?
"""
