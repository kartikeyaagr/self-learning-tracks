import numpy as np

META = dict(title='Walk everything an output depends on', tags=['Neural Networks'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:01.08'],
            related=['build-scalar-expression-graph', 'topologically-sort-computation-dag'])

STATEMENT = r"""
Starting from an output node, follow `_prev` pointers to find every node it depends on. This is the walk that drawing a graph, zeroing gradients and backpropagation all build on.

The starter gives a complete `Value` class (with `+`, `*`, unary `-` and `-`, where `-v` is `v * -1` and `a - b` is `a + (-b)`) and the program interpreter from the previous problem:

- `inputs`: dict `{name: float}`. `program`: lines `[name, op, arg]` / `[name, op, arg1, arg2]` with ops `"add"`, `"sub"`, `"mul"`, `"neg"`; each argument is the name of an input/earlier line or a numeric constant.
- Every numeric constant becomes its **own** new leaf node, and so does every hidden `-1` created by a negation.

Write `graph_stats(inputs, program)` for the output node, which is the node of the **last** line (the program has at least one line). Count only nodes reachable from it, each distinct node object **once**, even if several parents point at it. Return:

| key | value |
|---|---|
| `"nodes"` | number of distinct reachable nodes, the output included |
| `"edges"` | total number of operand slots over those nodes, `sum(len(v._prev))` (so `x * x` contributes 2) |
| `"leaves"` | sorted list of the `data` of the reachable nodes with no operands |
| `"ops"` | dict `{op: count}` over the reachable nodes that have operands |

Lines the output doesn't depend on are ignored.
"""

STARTER = '''def graph_stats(inputs, program):
    """Return {"nodes", "edges", "leaves", "ops"} for the graph reachable from the last line."""
    env = {name: Value(x) for name, x in inputs.items()}
    for name, op, *args in program:
        env[name] = OPS[op](*[env[a] if isinstance(a, str) else a for a in args])
    root = env[program[-1][0]]
    # TODO: visit every node reachable from root exactly once and collect the statistics.
    raise NotImplementedError


OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "neg": lambda a: -a}


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self._prev = tuple(_prev)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
'''

SOLUTION = '''def graph_stats(inputs, program):
    env = {name: Value(x) for name, x in inputs.items()}
    for name, op, *args in program:
        env[name] = OPS[op](*[env[a] if isinstance(a, str) else a for a in args])
    root = env[program[-1][0]]

    # Iterative DFS with a visited set keyed on object identity (Value defines no __eq__/__hash__,
    # so the default hash is identity: two nodes with equal data stay distinct).
    seen, stack = {root}, [root]
    while stack:
        v = stack.pop()
        for child in v._prev:
            if child not in seen:
                seen.add(child)
                stack.append(child)

    ops = {}
    for v in seen:
        if v._prev:
            ops[v._op] = ops.get(v._op, 0) + 1
    return {
        "nodes": len(seen),
        "edges": sum(len(v._prev) for v in seen),
        "leaves": sorted(v.data for v in seen if not v._prev),
        "ops": dict(sorted(ops.items())),     # sorted only so the output prints the same every run
    }


OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "neg": lambda a: -a}


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self._prev = tuple(_prev)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
'''

HINTS = [
    'This is plain graph search: a stack (or recursion) plus a `seen` set. Put the root in both, then repeatedly pop a node and push its unseen operands.',
    'Use the node objects themselves as set members. `Value` has no `__eq__`, so membership is by identity, and two different nodes with the same `data` are counted separately.',
    'Once `seen` is complete, every statistic is one pass over it: `len(seen)`, `sum(len(v._prev) for v in seen)`, the sorted leaf data, and a count of `_op` for nodes with operands.',
]


def _prog(rng, names, n):
    pool, prog = list(names), []
    for i in range(n):
        op = ["add", "sub", "mul", "neg"][rng.integers(4)]
        a = pool[rng.integers(len(pool))]
        if op == "neg":
            prog.append([f"t{i}", op, a])
        else:
            b = pool[rng.integers(len(pool))] if rng.random() > 0.3 else float(round(rng.uniform(-2, 2), 1))
            prog.append([f"t{i}", op, a, b])
        pool.append(f"t{i}")
    return prog


_r = np.random.default_rng(17)
TESTS = [
    ({"a": 1.5, "b": -2.0, "c": 4.0}, [["m", "mul", "a", "b"], ["s", "add", "m", "c"]]),
    ({"x": 3.0}, [["y", "mul", "x", "x"], ["z", "add", "y", "x"]]),
    ({"x": 1.0, "unused": 9.0}, [["junk", "mul", "unused", 4.0], ["y", "sub", "x", 2.0]]),
    ({"a": 1.0, "b": 2.0}, [["c", "add", "a", "b"], ["d", "mul", "c", "a"], ["e", "mul", "c", "b"], ["f", "add", "d", "e"]]),
    ({"x": 5.0}, [["y", "neg", "x"], ["z", "neg", "y"]]),
    ({"x": 0.5}, [["y", "add", "x", 1.0], ["z", "add", "y", 1.0]]),
    ({"a": 0.3, "b": -0.8, "c": 1.1, "d": 2.0}, _prog(_r, ["a", "b", "c", "d"], 12)),
    ({"a": -1.0, "b": 0.25}, _prog(_r, ["a", "b"], 20)),
]

EXPLANATION = r"""
## Why a visited set
An expression graph is a DAG, not a tree: in test 4, `c` feeds both `d` and `e`, and in test 2, `x` feeds `y` twice and `z` once. A traversal without a visited set would revisit shared nodes once per path, and the number of paths can grow exponentially with depth. With the set, each node is expanded once.

## Identity, not equality
Test 6 adds two constants that are both `1.0`. They are different nodes and both show up in `leaves`. Because `Value` doesn't override `__eq__`/`__hash__`, Python hashes it by identity, which is exactly what we want. If you ever add a value-based `__eq__` (as NumPy and PyTorch do for elementwise comparison), you must key the set by `id(v)` instead.

## Hidden nodes
Negations and subtractions create extra nodes you never named: a `-1` leaf and a `*` node per negation. Test 5 shows two negations producing two separate `-1.0` leaves.

## Complexity
$O(V + E)$ time and $O(V)$ memory for $V$ reachable nodes and $E$ operand slots. The iterative stack avoids Python's recursion limit on long chains.

## Common bugs
- Counting every input instead of only what's reachable from the output (test 3).
- Counting `x * x` as one edge, or visiting `x` twice.
- Keying `seen` on `v.data`, which merges distinct nodes that happen to share a value.

## Interview follow-ups
- How would you render this graph with Graphviz, with one box per node and one arrow per operand slot?
- In which order would you need to visit nodes to compute gradients, and why isn't DFS discovery order enough?
"""
