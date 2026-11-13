import numpy as np

META = dict(title='A Value node that adds', tags=['Neural Networks'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:01.08'],
            related=['create-multiplication-value-node', 'build-scalar-expression-graph', 'trace-reachable-expression-graph'])

STATEMENT = r"""
An autograd engine starts with a wrapper around a single number that also remembers **where it came from**. Complete the `Value` class so that `+` builds a new node:

- `Value(data)` stores `data` as a float in `.data`, has an empty tuple of operands `._prev == ()` and an empty op string `._op == ""`.
- `a + b` returns a **new** `Value` with `data = a.data + b.data`, `_prev = (a, b)` (left operand first) and `_op = "+"`.
- Plain numbers mix in on either side: `v + 2` and `2 + v` both work. The number is wrapped in a fresh `Value` first. For `2 + v` Python calls `v.__radd__(2)`; treat it as `v + 2`, so the node's operands are `(v, Value(2))`, the `Value` first.

The function `add_chain(numbers, offset)` (already written in the starter) wraps each number in a `Value`, adds them all with Python's built-in `sum` (which starts from the integer `0`, so the first addition is `0 + Value`), then adds `offset` on the right. It returns a description of the final node and of the node under it:

```
{"total": describe(total), "inner": describe(total._prev[0])}
```

where `describe(v) = {"data": v.data, "op": v._op, "children": [c.data for c in v._prev]}`.

## Inputs
`numbers`: list of at least one float. `offset`: float. You only need to finish the class; the rest of the starter is plumbing.
"""

STARTER = '''def add_chain(numbers, offset):
    """Sum Value-wrapped numbers with sum(), add offset, describe the last two nodes."""
    total = sum(Value(x) for x in numbers) + offset
    describe = lambda v: {"data": v.data, "op": v._op, "children": [c.data for c in v._prev]}
    return {"total": describe(total), "inner": describe(total._prev[0])}


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self._prev = tuple(_prev)
        self._op = _op

    def __add__(self, other):
        # TODO: wrap plain numbers, then return a new node with operands (self, other) and op "+".
        raise NotImplementedError

    def __radd__(self, other):
        # TODO: called for `number + Value`.
        raise NotImplementedError
'''

SOLUTION = '''def add_chain(numbers, offset):
    """Sum Value-wrapped numbers with sum(), add offset, describe the last two nodes."""
    total = sum(Value(x) for x in numbers) + offset
    describe = lambda v: {"data": v.data, "op": v._op, "children": [c.data for c in v._prev]}
    return {"total": describe(total), "inner": describe(total._prev[0])}


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self._prev = tuple(_prev)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)   # don't wrap a Value twice
        return Value(self.data + other.data, (self, other), "+")

    def __radd__(self, other):
        return self + other        # number + v  ->  v + number, so the Value stays the left operand
'''

HINTS = [
    'An operator method returns a brand-new `Value`; never modify `self`. The new node records its operands in `_prev`.',
    'Normalise the right operand first: `other = other if isinstance(other, Value) else Value(other)`. Wrapping a `Value` inside another `Value` is a classic bug, since `float(Value)` fails.',
    '`__radd__` can simply `return self + other`, which reuses `__add__` and puts the `Value` on the left.',
]

_r = np.random.default_rng(8)
TESTS = [
    ([1.0, 2.0, 3.0], 10.0),
    ([4.5], -1.5),
    ([0.0, 0.0], 0.0),
    ([-2.0, 2.0, -2.0, 2.0], 0.25),
    ([1e6, 1.0], -1e6),
    ([float(round(v, 3)) for v in _r.normal(0, 3, size=6)], float(round(_r.normal(), 3))),
]

EXPLANATION = r"""
## What the node stores
A `Value` is a number plus a pointer to the nodes it was computed from. That pointer is the whole graph: following `_prev` from the final result reaches every input, which is what backpropagation will walk later. The `_op` string is only for inspection and drawing, but it's handy when debugging.

## Why `__radd__` matters
Python's `sum` starts from the integer `0`, so its first step is `0 + Value(...)`. `int.__add__` doesn't know about `Value` and returns `NotImplemented`, so Python then tries `Value.__radd__(0)`. Without it, `sum` raises `TypeError`. That's why `inner` in test 2 has the children `[4.5, 0.0]`: the hidden zero is a real node in the graph.

## Complexity
Each `+` is $O(1)$ and allocates one node (plus one for a wrapped constant). Summing $n$ values builds a left-leaning chain of $n$ addition nodes.

## Common bugs
- Returning a float instead of a `Value`, which loses the graph.
- `Value(other)` when `other` is already a `Value`.
- Storing operands in a `set`: it loses their order and collapses `v + v` to a single child.

## Interview follow-ups
- A left-leaning chain from `sum` has depth $n$. Why might a recursive graph traversal then hit Python's recursion limit, and how would you avoid it?
- Should `Value` define `__eq__`? What would that break if nodes are stored in sets or dicts? (Equal data would make different nodes collide.)
"""
