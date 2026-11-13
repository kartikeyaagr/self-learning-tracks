import numpy as np

META = dict(title='Topological order of a Value graph', tags=['Neural Networks'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:01.08'],
            related=['trace-reachable-expression-graph', 'run-reverse-mode-autodiff', 'backpropagation-through-computation-graph'])

STATEMENT = r"""
Backpropagation can call a node's backward rule only after **every** node that uses it has passed its gradient down. That means visiting nodes in reverse *topological order*: an order in which every node comes after all of its operands.

Build that order with a depth-first search, the way small autograd engines do it.

## The graph
`inputs` is a dict `{name: float}` and `program` a list of lines `[name, op, arg]` / `[name, op, arg1, arg2]` with ops `"add"`, `"mul"` and `"tanh"`. Every argument is the **name** of an input or an earlier line (no constants), so every node carries a `.label`. The output is the node of the **last** line.

## The exact order to produce
Write `topo_order(inputs, program)` returning the list of labels produced by this procedure, started at the output:

```
visit(v):
    if v was already visited: return
    mark v visited
    for each operand c of v, in order (left, then right):  visit(c)
    append v to the order
```

So operands are explored left to right, each node appears **once** even if it's used many times, the output comes last, and nodes the output doesn't depend on are left out.

The starter gives `Value` (with labels) and the interpreter. You write the traversal.
"""

STARTER = '''import math

def topo_order(inputs, program):
    env = {name: Value(x, label=name) for name, x in inputs.items()}
    for name, op, *args in program:
        a = [env[x] for x in args]
        env[name] = (a[0] + a[1] if op == "add" else a[0] * a[1] if op == "mul" else a[0].tanh())
        env[name].label = name
    root = env[program[-1][0]]
    # TODO: depth-first post-order from root; return the labels.
    raise NotImplementedError


class Value:
    def __init__(self, data, _prev=(), _op="", label=""):
        self.data = float(data)
        self._prev = tuple(_prev)
        self._op = _op
        self.label = label

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")

    def tanh(self):
        return Value(math.tanh(self.data), (self,), "tanh")
'''

SOLUTION = '''import math

def topo_order(inputs, program):
    env = {name: Value(x, label=name) for name, x in inputs.items()}
    for name, op, *args in program:
        a = [env[x] for x in args]
        env[name] = (a[0] + a[1] if op == "add" else a[0] * a[1] if op == "mul" else a[0].tanh())
        env[name].label = name
    root = env[program[-1][0]]

    order, visited = [], set()
    def visit(v):
        if v in visited:
            return
        visited.add(v)                 # mark before recursing: each node is expanded once
        for child in v._prev:
            visit(child)
        order.append(v)                # post-order: all operands are already in the list
    visit(root)
    return [v.label for v in order]


class Value:
    def __init__(self, data, _prev=(), _op="", label=""):
        self.data = float(data)
        self._prev = tuple(_prev)
        self._op = _op
        self.label = label

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")

    def tanh(self):
        return Value(math.tanh(self.data), (self,), "tanh")
'''

HINTS = [
    'A node may be appended only after all of its operands. Depth-first search with the append placed *after* the recursive calls (post-order) guarantees that.',
    'Keep a `visited` set of node objects and an `order` list in the enclosing function; a nested `visit(v)` can use both.',
    'Inside `visit`: return if `v` is visited, add it, call `visit(c)` for each `c in v._prev`, then `order.append(v)`. Call `visit(root)` and map the result to labels.',
]


def _prog(rng, names, n):
    pool, prog = list(names), []
    for i in range(n):
        op = ["add", "mul", "tanh"][rng.integers(3)]
        # Prefer recent nodes so the random graph is deep rather than a pile of unused lines.
        args = [pool[rng.integers(max(0, len(pool) - 4), len(pool))] for _ in range(1 if op == "tanh" else 2)]
        prog.append([f"n{i}", op, *args])
        pool.append(f"n{i}")
    return prog


_r = np.random.default_rng(13)
TESTS = [
    ({"a": 1.5, "b": -2.0, "c": 4.0}, [["m", "mul", "a", "b"], ["s", "add", "m", "c"], ["o", "tanh", "s"]]),
    ({"x": 0.5, "y": 1.5}, [["s", "add", "x", "y"], ["p", "mul", "s", "x"], ["q", "mul", "s", "y"], ["out", "add", "p", "q"]]),
    ({"x": 1.0}, [["sq", "mul", "x", "x"], ["cube", "mul", "sq", "x"]]),
    ({"a": 1.0, "b": 2.0}, [["unused", "add", "b", "b"], ["t", "tanh", "a"]]),
    ({"x": 0.1}, [["h1", "tanh", "x"], ["h2", "tanh", "h1"], ["h3", "tanh", "h2"], ["h4", "tanh", "h3"]]),
    ({"a": 1.0, "b": 2.0}, [["u", "add", "b", "a"], ["v", "mul", "a", "u"]]),
    ({"a": 0.3, "b": -0.7, "c": 1.2}, _prog(_r, ["a", "b", "c"], 10)),
    ({"p": 0.5, "q": 0.25, "r": -1.0, "s": 2.0}, _prog(_r, ["p", "q", "r", "s"], 16)),
]

EXPLANATION = r"""
## Why post-order DFS gives a topological order
A node is appended only after `visit` has returned for each of its operands, and `visit` returning means that operand (and everything below it) is already in the list. So every node lands after everything it depends on. Reversing the list gives an order in which every node comes **before** its operands, which is what backpropagation needs: by the time a node's `_backward` runs, all of its consumers have already added their contributions to its `.grad`.

## Why the visited set matters twice
- Correctness: in test 2, `s` feeds both `p` and `q`. Without the set, `s` would appear twice and its backward rule would run twice, passing its gradient down twice.
- Speed: on graphs with heavy sharing, the number of root-to-leaf paths can be exponential, while the number of nodes is linear.

## Complexity
$O(V + E)$ time and $O(V)$ memory. The recursion depth equals the longest path, so a very deep chain (thousands of nodes) needs an explicit stack or a raised recursion limit.

## Common bugs
- Appending *before* the recursive calls (pre-order), which puts parents before their operands.
- Marking a node visited only after its children are done. In a DAG that still terminates, but it re-explores shared nodes.
- Starting from the inputs instead of the output, which drags in unused lines (test 4).

## Interview follow-ups
- Implement the same order iteratively with an explicit stack. What extra state does each stack entry need?
- Kahn's algorithm produces a topological order by repeatedly removing nodes with no remaining dependencies. When would you prefer it to DFS?
"""
