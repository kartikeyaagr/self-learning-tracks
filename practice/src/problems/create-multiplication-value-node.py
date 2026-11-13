import numpy as np

META = dict(title='A Value node that multiplies', tags=['Neural Networks'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:01.08'],
            related=['create-addition-value-node', 'build-scalar-expression-graph'])

STATEMENT = r"""
Extend the scalar `Value` node with multiplication. Addition is already done in the starter; add `__mul__` and `__rmul__` with the same conventions:

- `a * b` returns a **new** `Value` with `data = a.data * b.data`, `_prev = (a, b)` (left operand first) and `_op = "*"`.
- A plain number on the right is wrapped in a fresh `Value`. A number on the left (`3 * v`) calls `v.__rmul__(3)`; treat it as `v * 3`, so the operands are `(v, Value(3))`.
- Multiplying a node by itself keeps **both** operand slots: `q * q` has `_prev == (q, q)`.

`scaled_product(a, b, k)` (already written) builds

```
p = Value(a) * Value(b)
q = k * p          # number on the left
s = q * q + 1      # node times itself, then a number on the right
```

and returns `{"p": describe(p), "q": describe(q), "s": describe(s), "s_left": describe(s._prev[0])}` with `describe(v) = {"data": v.data, "op": v._op, "children": [c.data for c in v._prev]}`.

## Inputs
`a`, `b`, `k`: floats.
"""

STARTER = '''def scaled_product(a, b, k):
    p = Value(a) * Value(b)
    q = k * p
    s = q * q + 1
    describe = lambda v: {"data": v.data, "op": v._op, "children": [c.data for c in v._prev]}
    return {"p": describe(p), "q": describe(q), "s": describe(s), "s_left": describe(s._prev[0])}


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self._prev = tuple(_prev)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        # TODO
        raise NotImplementedError

    def __rmul__(self, other):
        # TODO
        raise NotImplementedError
'''

SOLUTION = '''def scaled_product(a, b, k):
    p = Value(a) * Value(b)
    q = k * p
    s = q * q + 1
    describe = lambda v: {"data": v.data, "op": v._op, "children": [c.data for c in v._prev]}
    return {"p": describe(p), "q": describe(q), "s": describe(s), "s_left": describe(s._prev[0])}


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self._prev = tuple(_prev)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    def __rmul__(self, other):
        return self * other        # k * v  ->  v * k
'''

HINTS = [
    '`__mul__` has exactly the same shape as `__add__`: normalise `other`, then build one new node.',
    'Use a tuple for `_prev`, not a set, so `q * q` keeps two entries.',
    '`__rmul__` only needs `return self * other`.',
]

_r = np.random.default_rng(2)
TESTS = [
    (2.0, 3.0, 0.5),
    (-1.0, 4.0, -2.0),
    (0.0, 7.0, 3.0),
    (1.5, 1.5, 1.0),
    (1e-3, 2e3, 10.0),
    (*[float(round(v, 3)) for v in _r.normal(0, 2, size=3)],),
]

EXPLANATION = r"""
## Why multiplication needs both operands remembered
For addition the local derivative is 1 for each side, so in principle you could forget the operands' values. For multiplication $\partial(ab)/\partial a = b$: the backward pass needs the *other* operand's value. Keeping `(a, b)` in order is what makes that possible later.

## Self-multiplication
`q * q` is a node whose two operand slots hold the same object. Storing them as a tuple `(q, q)` keeps the information that $q$ was used twice, which is exactly why $\partial (q^2)/\partial q = 2q$ rather than $q$. A set would collapse it to one child and lose a factor of 2 once gradients flow.

## Complexity
$O(1)$ per operation; one new node per `*`, plus one per wrapped constant.

## Common bugs
- Wrapping a `Value` in another `Value`.
- Forgetting `__rmul__`, so `3 * v` raises `TypeError`.
- Implementing `__rmul__` as `Value(other) * self`. The value is right, but the operand order no longer matches the convention, so `children` comes out reversed.

## Interview follow-ups
- How would you add `__neg__` and `__sub__` without writing new backward rules? (`-v = v * -1` and `a - b = a + (-b)`.)
- Why do frameworks like PyTorch keep a graph per *operation* rather than per *expression*?
"""
