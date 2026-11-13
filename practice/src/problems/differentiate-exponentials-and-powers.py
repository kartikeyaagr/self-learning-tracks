import numpy as np

META = dict(title='exp, powers and division for Value', tags=['Neural Networks', 'Calculus'], diff='Medium',
            runtime='python', lessons=['air-fanout-v1:01.06', 'air-fanout-v1:01.08'],
            related=['run-reverse-mode-autodiff', 'apply-local-vector-jacobian-rules', 'gradient-check-autodiff-graph'])

STATEMENT = r"""
Round out the scalar engine so it can express sigmoids, softmaxes and squared errors. The starter has `+`, `*` (with numbers on either side), negation, subtraction and a working `backward()`. Add:

| method | meaning | backward rule (into `self.grad`) |
|---|---|---|
| `v.exp()` | $e^v$ | `+= out.data * out.grad` |
| `v ** k` | $v^k$ for a constant int/float `k` | `+= k * v.data**(k - 1) * out.grad` |
| `a / b` | `a * b**-1` | none: built from `*` and `**` |
| `c / v` (number on the left) | `c * v**-1` | none |

Only `exp` and `**` are new primitives with their own closures. Division must reuse them, so a `Value` in the denominator gets its gradient through the power rule.

`expr_grads(inputs, program)` (given) builds the graph from the usual format (`inputs` dict `{name: float}`; `program` lines with ops `"add"`, `"sub"`, `"mul"`, `"neg"`, `"div"`, `"exp"` and `"pow"`; arguments are names or numeric constants; a `"pow"` exponent is always a constant, as in `["y", "pow", "x", 2]`), backpropagates from the last line, and returns

```
{"value": output.data, "grads": {name: grad for every input and line}}
```

Tests never raise $0$ to a negative power or a negative number to a fractional one.
"""

STARTER = '''import math

def expr_grads(inputs, program):
    env = {name: Value(x) for name, x in inputs.items()}
    for name, op, *args in program:
        env[name] = OPS[op](*[env[a] if isinstance(a, str) else a for a in args])
    out = env[program[-1][0]]
    out.backward()
    return {"value": out.data, "grads": {name: v.grad for name, v in env.items()}}


OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "neg": lambda a: -a,
       "div": lambda a, b: a / b, "exp": lambda a: a.exp(), "pow": lambda a, k: a ** k}


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

    # TODO: exp(), __pow__(k), __truediv__, __rtruediv__

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

def expr_grads(inputs, program):
    env = {name: Value(x) for name, x in inputs.items()}
    for name, op, *args in program:
        env[name] = OPS[op](*[env[a] if isinstance(a, str) else a for a in args])
    out = env[program[-1][0]]
    out.backward()
    return {"value": out.data, "grads": {name: v.grad for name, v in env.items()}}


OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "neg": lambda a: -a,
       "div": lambda a, b: a / b, "exp": lambda a: a.exp(), "pow": lambda a, k: a ** k}


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

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad           # e^x is its own derivative
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other ** -1      # a Value or a number in the denominator

    def __rtruediv__(self, other):
        return other * self ** -1      # number / Value

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
    'Both new primitives have one operand: create `out = Value(..., (self,), ...)` and a closure that adds `local derivative * out.grad` to `self.grad`.',
    'For `exp`, the local derivative is the output itself, `out.data`. For `**`, it is `k * self.data ** (k - 1)`; `k` is a plain number captured by the closure.',
    'Division is one line each: `__truediv__` returns `self * other ** -1` (Python evaluates `**` first, and a plain number just becomes its reciprocal), `__rtruediv__` returns `other * self ** -1`.',
]

_r = np.random.default_rng(61)
_p = lambda: float(round(_r.uniform(0.3, 1.5), 3))
TESTS = [
    ({"x": 3.0}, [["y", "pow", "x", 2]]),
    ({"x": 0.0}, [["n", "neg", "x"], ["e", "exp", "n"], ["d", "add", "e", 1.0], ["s", "div", 1.0, "d"]]),
    ({"a": 6.0, "b": 2.0}, [["q", "div", "a", "b"]]),
    ({"x": 4.0}, [["r", "pow", "x", 0.5], ["i", "pow", "x", -1], ["o", "add", "r", "i"]]),
    ({"x": 1.5}, [["c", "pow", "x", 0], ["o", "mul", "c", "x"]]),
    ({"y": 0.4, "t": 1.0}, [["d", "sub", "y", "t"], ["sq", "pow", "d", 2], ["l", "div", "sq", 2.0]]),
    ({"a": 1.0, "b": 2.0, "c": 0.5}, [["ea", "exp", "a"], ["eb", "exp", "b"], ["ec", "exp", "c"], ["z1", "add", "ea", "eb"],
                                      ["z", "add", "z1", "ec"], ["pa", "div", "ea", "z"]]),
    ({"x": -2.0}, [["n", "neg", "x"], ["e", "exp", "n"], ["c", "pow", "x", 3], ["o", "div", "c", "e"]]),
    ({"u": _p(), "v": _p(), "w": _p()}, [["a", "mul", "u", "v"], ["b", "pow", "w", 1.5], ["c", "div", "a", "b"], ["d", "exp", "c"],
                                         ["e", "sub", "d", "u"], ["f", "div", 2.0, "e"]]),
]

EXPLANATION = r"""
## Two primitives buy a lot
With `exp` and constant powers, everything else here is composition:
- division: $a / b = a \cdot b^{-1}$, so $\partial/\partial b = a \cdot (-1) b^{-2}$ falls out of the power rule and the product rule (test 3 gives $-a/b^2 = -1.5$);
- the sigmoid $1/(1 + e^{-x})$ (test 2 computes it at 0 and gets slope $\tfrac14$);
- softmax $e^{a}/\sum e^{z}$ (test 7), where the gradient flows into `a` along two paths, through the numerator and through the sum.

## What the closures read
The closure runs during the backward pass, after the whole forward pass is done, and reads the forward value it needs. For `exp`, reusing `out.data` saves recomputing $e^x$; that's the standard trick of storing outputs for the backward pass.

## Edge cases
$x^0$ (test 5) is the constant 1, but the engine still records the node and sends back $0 \cdot x^{-1} = 0$. Its gradient is $0$, as it should be; with $x = 0$ that formula would divide by zero, which is why real frameworks special-case it.

## Complexity
$O(1)$ per new op; division costs two nodes instead of one, a fair price for not writing another backward rule.

## Common bugs
- `__truediv__` written as `Value(self.data / other.data)`: the value is right, but the result has no operands, so no gradient reaches `a` or `b`.
- `__rtruediv__` returning `self * other ** -1` (that's $v / c$, not $c / v$).
- Power rule with `self.data ** k` instead of `** (k - 1)`.

## Interview follow-ups
- Add `log()`. What's its backward rule, and why do softmax implementations prefer `logsumexp` over `log(sum(exp(z)))`?
- How would you support `Value ** Value`? (Add the rule $\partial a^b / \partial b = a^b \ln a$, valid for $a > 0$.)
"""
