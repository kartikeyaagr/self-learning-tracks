import numpy as np

META = dict(title='Interpret a program into a Value graph', tags=['Neural Networks'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:01.08'],
            related=['create-multiplication-value-node', 'trace-reachable-expression-graph', 'run-reverse-mode-autodiff'])

STATEMENT = r"""
Real models aren't written one operator at a time in a test file; they're programs. Here a tiny program describes a scalar expression, and you'll turn it into a graph of `Value` nodes.

## The program format
- `inputs`: a dict `{name: float}` of leaf values.
- `program`: a list of lines `[name, op, arg]` or `[name, op, arg1, arg2]`, run in order. Each argument is either the **name** (a string) of an input or an earlier line, or a **numeric constant**. Names are unique.

| op | meaning |
|---|---|
| `"add"` | `arg1 + arg2` |
| `"sub"` | `arg1 - arg2` |
| `"mul"` | `arg1 * arg2` |
| `"neg"` | `-arg` (its argument is always a name) |

A binary op has at least one named argument, so the constant may be on either side.

## Your job
1. Finish `Value`: `+` and `*` are given. Add `__neg__`, `__sub__` and `__rsub__` **without new primitive ops**: `-v` is `v * -1`, `a - b` is `a + (-b)`, and `c - v` (number on the left) is `c + (-v)`. So a `"neg"` line produces a `"*"` node and a `"sub"` line produces a `"+"` node.
2. Write `build_graph(inputs, program)`: create one `Value` per input, evaluate each line with the operators (a name refers to the **same** node object every time it is used, a constant is passed as a plain number), and return

```
{name: {"data": node.data, "op": node._op} for every input and every line, inputs first, then lines in order}
```

Inputs have op `""`.
"""

STARTER = '''def build_graph(inputs, program):
    """Evaluate the program with Value nodes; return {name: {"data", "op"}} for inputs and lines."""
    raise NotImplementedError


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

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # TODO: __neg__, __sub__, __rsub__, built from + and * only.
'''

SOLUTION = '''def build_graph(inputs, program):
    env = {name: Value(x) for name, x in inputs.items()}
    for name, op, *args in program:
        operands = [env[a] if isinstance(a, str) else a for a in args]   # names -> shared nodes
        env[name] = OPS[op](*operands)
    return {name: {"data": v.data, "op": v._op} for name, v in env.items()}


OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "neg": lambda a: -a,
}


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

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)      # works for a Value or a number on the right

    def __rsub__(self, other):
        return other + (-self)      # number - v; the + dispatches to (-v).__radd__
'''

HINTS = [
    'Keep a dict `env` from name to `Value`. Start it with the inputs; after each line, store the new node under the line\'s name.',
    'Resolve each argument with `env[a] if isinstance(a, str) else a`, then apply the operator. A small dict from op name to `lambda` keeps the loop to two lines.',
    '`__neg__` is `self * -1`, `__sub__` is `self + (-other)` and `__rsub__` is `other + (-self)`. Python routes `5 - v` to `v.__rsub__(5)`.',
]


def _prog(rng, names, n):
    pool, prog = list(names), []
    for i in range(n):
        op = ["add", "sub", "mul", "neg"][rng.integers(4)]
        a = pool[rng.integers(len(pool))]
        if op == "neg":
            prog.append([f"t{i}", op, a])
            pool.append(f"t{i}")
            continue
        b = pool[rng.integers(len(pool))]
        if rng.random() < 0.3:
            b = float(round(rng.uniform(-2, 2), 1))
            if rng.random() < 0.5:
                a, b = b, a
        prog.append([f"t{i}", op, a, b])
        pool.append(f"t{i}")
    return prog


_r = np.random.default_rng(31)
TESTS = [
    ({"a": 1.5, "b": -2.0}, [["m", "mul", "a", "b"], ["s", "add", "m", 4.0]]),
    ({"x": 4.0}, [["y", "neg", "x"], ["z", "sub", 1.0, "y"]]),
    ({"a": 1.5, "b": 0.5}, [["d", "sub", "a", "b"], ["e", "sub", "d", 2.0], ["f", "mul", "e", "e"]]),
    ({"x": 3.0}, [["y", "mul", 2.0, "x"], ["z", "add", 1.0, "y"], ["w", "sub", "z", "x"]]),
    ({"p": 0.0, "q": -1.0}, [["r", "neg", "q"], ["s", "mul", "p", "r"], ["t", "sub", "s", "s"]]),
    ({"x": 5.0}, []),
    ({"a": 0.7, "b": -1.2, "c": 2.0}, _prog(_r, ["a", "b", "c"], 10)),
]

EXPLANATION = r"""
## Graphs come from running code
The program never mentions nodes or edges. They appear because every operator call returns a new `Value` that points at its operands. This is "define-by-run", the model PyTorch uses: the graph is a by-product of ordinary evaluation, so loops, conditionals and helper functions all work.

## Why `neg` and `sub` reuse `*` and `+`
Each primitive op needs a backward rule. Defining $-v = v \cdot (-1)$ and $a - b = a + (-b)$ adds operators without adding rules, so there's less code to get wrong. The cost is a few extra nodes (a constant $-1$ and a multiply per negation). That's why the `"sub"` lines report `"+"` and the `"neg"` lines report `"*"`.

## Sharing is the point
Looking names up in `env` means `f = e * e` multiplies one node by itself, and a name used on several lines feeds several parents. Copying values instead would still give the right `data`, but the graph would no longer say that one quantity influences several others, and the gradients built on it later would be wrong.

## Complexity
$O(L)$ time and nodes for $L$ lines (each line adds at most three nodes: a constant, a $-1$ and the result).

## Common bugs
- Treating a constant as a name (or a name as a constant).
- Implementing `__rsub__` as `self - other`, which flips the sign.
- Returning only the program lines and forgetting the inputs.

## Interview follow-ups
- How would you support `"div"` with the same trick? (`a / b = a * b**-1`, which needs a power op.)
- What changes if programs may reassign a name, as in `x = x + 1`?
"""
