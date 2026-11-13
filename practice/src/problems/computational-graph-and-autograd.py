import numpy as np

META = dict(title='Tensor autograd with broadcasting', tags=['Neural Networks', 'Calculus'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:01.08', 'air-fanout-v1:02.03'],
            related=['backpropagation-through-computation-graph', 'backpropagation-single-hidden-layer',
                     'jvp-and-vjp-for-2-layer-network'])

STATEMENT = r"""
Real autograd engines work on tensors, and the subtle part is **broadcasting**: if a bias of shape `(3,)` was added to a `(4, 3)` matrix, its gradient must be summed back down to shape `(3,)`.

A graph is a list of nodes in topological order (each node refers only to earlier nodes by index). The **last node** is the output and is always a scalar.

| Node | Value |
|---|---|
| `['input', name]` | the array `inputs[name]` |
| `['const', c]` | the float or array `c` (no gradient needed) |
| `['add', a, b]`, `['sub', a, b]`, `['mul', a, b]` | elementwise, with NumPy broadcasting |
| `['matmul', a, b]` | matrix product of two **2-D** values |
| `['relu', a]`, `['exp', a]`, `['square', a]` | elementwise functions |
| `['sum', a]`, `['mean', a]` | reduce **all** elements to a scalar |

Write `backprop(nodes, inputs)` returning `(value, grads)`: the output as a Python `float`, and a dict mapping every key of `inputs` to $\partial\,\text{output}/\partial\,\text{inputs[name]}$, an array of the same shape as that input.

## Rules
- One forward pass, then one reverse pass accumulating gradients. A node can be used by several later nodes, and its gradient is the sum over all uses.
- Several `input` nodes may name the same input: their gradients add. Inputs the output does not depend on get zeros.
- Use $\mathrm{relu}'(0) = 0$.
"""

STARTER = '''import numpy as np

def backprop(nodes, inputs):
    """Return (output value as float, {name: gradient shaped like inputs[name]})."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _unbroadcast(g, shape):
    """Sum g down to `shape`, undoing NumPy broadcasting."""
    while g.ndim > len(shape):                       # dims that were prepended
        g = g.sum(axis=0)
    for ax, n in enumerate(shape):                   # dims that were stretched from 1
        if n == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g

def backprop(nodes, inputs):
    vals = []
    for op, *args in nodes:                          # forward
        v = [vals[i] for i in args] if op not in ('input', 'const') else None
        if op == 'input':
            vals.append(np.asarray(inputs[args[0]], dtype=float))
        elif op == 'const':
            vals.append(np.asarray(args[0], dtype=float))
        else:
            f = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b,
                 'matmul': lambda a, b: a @ b, 'relu': lambda a: np.maximum(a, 0.0), 'exp': np.exp,
                 'square': lambda a: a * a, 'sum': lambda a: a.sum(), 'mean': lambda a: a.mean()}[op]
            vals.append(np.asarray(f(*v), dtype=float))

    g = [np.zeros_like(x) for x in vals]
    g[-1] = np.ones_like(vals[-1])
    out = {k: np.zeros(np.shape(x)) for k, x in inputs.items()}
    for i in range(len(nodes) - 1, -1, -1):          # reverse
        op, *args = nodes[i]
        gi = g[i]
        if op == 'input':
            out[args[0]] = out[args[0]] + gi
            continue
        if op == 'const':
            continue
        x = [vals[j] for j in args]
        if op in ('add', 'sub', 'mul'):
            a, b = x
            ga = gi * b if op == 'mul' else gi
            gb = gi * a if op == 'mul' else (gi if op == 'add' else -gi)
            g[args[0]] = g[args[0]] + _unbroadcast(ga, a.shape)
            g[args[1]] = g[args[1]] + _unbroadcast(gb, b.shape)
        elif op == 'matmul':
            a, b = x
            g[args[0]] = g[args[0]] + gi @ b.T
            g[args[1]] = g[args[1]] + a.T @ gi
        else:
            a = x[0]
            local = {'relu': lambda: (a > 0) * gi, 'exp': lambda: vals[i] * gi, 'square': lambda: 2 * a * gi,
                     'sum': lambda: np.broadcast_to(gi, a.shape), 'mean': lambda: np.broadcast_to(gi / a.size, a.shape)}[op]()
            g[args[0]] = g[args[0]] + local
    return float(vals[-1]), {k: np.asarray(v, dtype=float) for k, v in out.items()}
'''

HINTS = [
    'Store every node\'s value in a list on the way forward. On the way back keep a gradient array per node, shaped like its value, start the output at 1 and walk indices in reverse, **adding** into the gradients of each node\'s parents.',
    'Local rules: `add` passes `g` to both sides, `sub` passes `-g` to the right, `mul` sends `g*b` and `g*a`, `matmul` sends `g @ B.T` and `A.T @ g`, `sum` broadcasts `g` to the input shape, `mean` does the same divided by the element count.',
    'After `add`/`sub`/`mul`, a parent may be smaller than `g`. Sum away the leading extra axes, then sum (with `keepdims=True`) every axis where the parent had size 1.',
]

_r = np.random.default_rng(23)
_lin = [['input', 'X'], ['input', 'W'], ['matmul', 0, 1], ['input', 'b'], ['add', 2, 3], ['relu', 4],
        ['input', 'Y'], ['sub', 5, 6], ['square', 7], ['mean', 8]]
TESTS = [
    ([['input', 'x'], ['input', 'y'], ['mul', 0, 1], ['sum', 2]], {'x': np.array([1.0, 2.0]), 'y': np.array([3.0, -1.0])}),
    ([['input', 'A'], ['input', 'b'], ['add', 0, 1], ['sum', 2]], {'A': np.ones((2, 3)), 'b': np.array([1.0, 2.0, 3.0])}),
    (_lin, {'X': _r.normal(size=(5, 3)), 'W': _r.normal(size=(3, 2)), 'b': _r.normal(size=2), 'Y': _r.normal(size=(5, 2))}),
    ([['input', 'a'], ['input', 'a'], ['mul', 0, 1], ['input', 'a'], ['mul', 2, 3], ['sum', 4]],
     {'a': np.array([1.0, -2.0, 0.5])}),
    ([['input', 'u'], ['input', 'v'], ['mul', 0, 1], ['exp', 2], ['const', 2.0], ['mul', 3, 4], ['mean', 5]],
     {'u': _r.normal(size=(3, 1)), 'v': _r.normal(size=(1, 4)), 'unused': np.ones(2)}),
    ([['input', 's'], ['input', 'M'], ['mul', 0, 1], ['relu', 2], ['input', 'M'], ['matmul', 3, 4], ['sum', 5]],
     {'s': np.array(2.0), 'M': np.array([[1.0, -1.0], [0.0, 3.0]])}),
    ([['input', 'X'], ['input', 'W1'], ['matmul', 0, 1], ['relu', 2], ['input', 'W2'], ['matmul', 3, 4],
      ['input', 'X'], ['input', 'Wskip'], ['matmul', 6, 7], ['add', 5, 8], ['const', np.array([[0.5, -0.5]])], ['add', 9, 10],
      ['square', 11], ['sum', 12]],
     {'X': _r.normal(size=(4, 3)), 'W1': _r.normal(size=(3, 5)), 'W2': _r.normal(size=(5, 2)), 'Wskip': _r.normal(size=(3, 2))}),
]

EXPLANATION = r"""
## Reverse mode in one pass
Every node gets one gradient buffer. Visiting nodes in reverse topological order guarantees that when you reach node $i$, every consumer of $i$ has already added its contribution, so $g_i$ is complete. That makes the cost linear in the number of nodes, however much the graph branches.

## Broadcasting is a hidden copy
`A + b` with `b` of shape `(3,)` behaves as if `b` were tiled to `(4, 3)`. The gradient of a copy is the sum of the copies' gradients, so the adjoint of broadcasting is a **sum** over the stretched axes. NumPy aligns shapes from the right, so first collapse prepended axes, then axes that were size 1 (test 5 broadcasts `(3, 1) * (1, 4)`). The same reasoning makes the gradient of a 0-d scalar that scaled a matrix (test 6) the sum over the whole matrix.

## Matmul
For $C = AB$, $\partial L/\partial A = G B^\top$ and $\partial L/\partial B = A^\top G$: the only arrangement whose shapes match.

## Complexity
Backward costs a small constant times the forward, with memory for every intermediate value.

## Common bugs
- Overwriting instead of adding gradients when a node is reused (test 4 uses `a` three times, the last test uses `X` twice).
- Returning a `(4, 3)` gradient for a `(3,)` bias.
- Forgetting that `mean` divides by the number of elements.

## Interview follow-ups
- How would you support `sum` over one axis with `keepdims=False`? (Re-insert the axis before broadcasting the gradient.)
- Why do frameworks free intermediate values as soon as backward has used them, and what does `retain_graph` change?
"""
