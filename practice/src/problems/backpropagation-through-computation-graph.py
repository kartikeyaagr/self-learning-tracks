import numpy as np

META = dict(title='Reverse-mode backprop over a computation graph', tags=['Neural Networks', 'Calculus'], diff='Hard',
            runtime='numpy', lessons=['air-fanout-v1:01.08', 'air-fanout-v1:01.07'],
            related=['forward-mode-autodiff-with-dual-numbers', 'chain-rule-through-3-layer-network',
                     'jvp-and-vjp-for-2-layer-network'])

STATEMENT = r"""
Autograd libraries record every operation as a node in a graph, then walk it backwards accumulating gradients. Build that engine for scalar graphs.

A graph is a list of nodes in **topological order** (every node only refers to earlier nodes). Node $i$ is a list whose first entry names the operation:

| Node | Value |
|---|---|
| `['var', j]` | the input `x[j]` |
| `['const', c]` | the float `c` |
| `['add', a, b]`, `['sub', a, b]`, `['mul', a, b]`, `['div', a, b]` | value of node `a` combined with value of node `b` |
| `['neg', a]`, `['exp', a]`, `['log', a]`, `['tanh', a]`, `['sigmoid', a]`, `['relu', a]` | a function of the value of node `a` |

`a` and `b` are integer indices of earlier nodes. The **last node is the output**.

Write `backprop(nodes, x)` that returns `(value, grad)`: the output value as a Python `float`, and $\partial\,\text{output}/\partial x$ as a float array with the same length as `x`.

## Rules
- One forward pass storing every node's value, then **one** reverse pass in reverse topological order. Don't recurse per path: a node can feed many later nodes (and a node can use the same input twice, as in `['mul', 3, 3]`), and its gradient is the **sum** over all its uses.
- Several `var` nodes may refer to the same `x[j]`; their gradients add. Inputs that the output doesn't depend on get gradient 0.
- Use $\mathrm{relu}'(0) = 0$. Test graphs keep `log` and `div` inside their domains.
"""

STARTER = '''import numpy as np

def backprop(nodes, x):
    """Evaluate a topologically sorted scalar graph; return (output value, d output / d x)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def backprop(nodes, x):
    x = np.asarray(x, dtype=float)
    val = []
    for node in nodes:                                   # forward pass: cache every value
        op, args = node[0], node[1:]
        if op == 'var':
            v = x[args[0]]
        elif op == 'const':
            v = float(args[0])
        else:
            a = val[args[0]]
            b = val[args[1]] if len(args) > 1 else None
            v = {'add': lambda: a + b, 'sub': lambda: a - b, 'mul': lambda: a * b, 'div': lambda: a / b,
                 'neg': lambda: -a, 'exp': lambda: np.exp(a), 'log': lambda: np.log(a), 'tanh': lambda: np.tanh(a),
                 'sigmoid': lambda: 1 / (1 + np.exp(-a)), 'relu': lambda: max(a, 0.0)}[op]()
        val.append(float(v))

    adj = [0.0] * len(nodes)                             # adjoints: d output / d node
    adj[-1] = 1.0
    grad = np.zeros(len(x))
    for i in range(len(nodes) - 1, -1, -1):              # reverse pass
        op, args, g, v = nodes[i][0], nodes[i][1:], adj[i], val[i]
        if op == 'var':
            grad[args[0]] += g
            continue
        if op == 'const':
            continue
        a = args[0]
        va = val[a]
        if op in ('add', 'sub', 'mul', 'div'):
            b = args[1]
            vb = val[b]
            if op == 'add':
                da, db = 1.0, 1.0
            elif op == 'sub':
                da, db = 1.0, -1.0
            elif op == 'mul':
                da, db = vb, va
            else:
                da, db = 1 / vb, -va / vb ** 2
            adj[a] += g * da                             # += so fan-out (and a == b) accumulates
            adj[b] += g * db
        else:                                            # unary: local derivative from the cached values
            if op == 'neg':
                local = -1.0
            elif op == 'exp':
                local = v
            elif op == 'log':
                local = 1 / va
            elif op == 'tanh':
                local = 1 - v * v
            elif op == 'sigmoid':
                local = v * (1 - v)
            else:
                local = 1.0 if va > 0 else 0.0           # relu, with relu'(0) = 0
            adj[a] += g * local
    return val[-1], grad
'''

HINTS = [
    'Forward: walk the list once and store each node\'s value. Backward: keep an "adjoint" per node, $\\bar v_i = \\partial\\,\\text{out}/\\partial v_i$, start the last one at 1 and walk the list in reverse.',
    'At node $i$, push its adjoint to its inputs: $\\bar v_a \\mathrel{+}= \\bar v_i\\, \\partial v_i/\\partial v_a$. Always `+=`, never `=`, so a node used several times collects every contribution. Because the order is topological, a node\'s adjoint is complete before you reach it.',
    'Local derivatives: `mul` gives $(v_b, v_a)$, `div` gives $(1/v_b, -v_a/v_b^2)$, `tanh` gives $1 - v_i^2$, `sigmoid` gives $v_i(1 - v_i)$, `exp` gives $v_i$. A `var` node adds its adjoint into `grad[j]`.',
]


def _random_graph(rng, n_vars, n_ops):
    """Alternate binary ops with squashing so values stay bounded."""
    nodes = [['var', j] for j in range(n_vars)] + [['const', 0.5]]
    for k in range(n_ops):
        a, b = (int(i) for i in rng.integers(0, len(nodes), size=2))
        nodes.append([['add', 'mul', 'sub'][k % 3], a, b])
        nodes.append([['tanh', 'sigmoid'][k % 2], len(nodes) - 1])
    return nodes


_r = np.random.default_rng(61)
TESTS = [
    ([['var', 0], ['var', 1], ['mul', 0, 1], ['add', 2, 0]], np.array([2.0, 3.0])),
    ([['var', 0], ['tanh', 0], ['mul', 1, 1]], np.array([0.5])),
    ([['var', 0], ['var', 0], ['mul', 0, 1], ['exp', 2]], np.array([0.7, 9.0])),
    ([['var', 0], ['const', 1.0], ['sub', 0, 1], ['relu', 2], ['var', 1], ['div', 3, 4]], np.array([1.0, 2.0])),
    ([['var', 0], ['var', 1], ['var', 2], ['mul', 0, 1], ['add', 3, 2], ['sigmoid', 4], ['log', 5], ['neg', 6]],
     np.array([1.5, -2.0, 0.5])),
    ([['var', 0], ['exp', 0], ['add', 1, 1], ['mul', 2, 1], ['div', 3, 0], ['log', 4]], np.array([1.2])),
    ([['var', 1], ['const', 3.0], ['mul', 0, 1]], np.array([5.0, 2.0, -1.0])),
    (_random_graph(_r, 4, 12), _r.normal(size=4)),
    (_random_graph(_r, 3, 20), _r.normal(size=3)),
]

EXPLANATION = r"""
## Why one reverse sweep is enough
For the output $o$, $\frac{\partial o}{\partial v_a} = \sum_{i \text{ uses } a} \frac{\partial o}{\partial v_i}\frac{\partial v_i}{\partial v_a}$: the multivariable chain rule sums over every path through each consumer. In reverse topological order, every consumer of $a$ is processed before $a$, so by the time you reach $a$ its adjoint has received all contributions. Each edge is touched once: the backward pass costs about the same as the forward pass, independent of the number of inputs. That's the "cheap gradient principle" behind training networks with billions of parameters.

## Fan-out is where naive implementations break
Test 2 computes $\tanh(x)^2$ by multiplying node 1 by itself, and test 3 reads the same input through two `var` nodes. With `=` instead of `+=`, the second contribution overwrites the first and the gradient comes out half as large. PyTorch's `.grad` accumulation (and the need to call `zero_grad`) is the same rule at the parameter level.

## Complexity
$O(|\text{nodes}|)$ time and memory for the cached values, which is why activation memory, not compute, often limits batch size. Gradient checkpointing trades recomputation for that memory.

## Common bugs
- Processing nodes in forward order during the backward pass.
- Using the input's value instead of the output's for `tanh`/`sigmoid`/`exp` local derivatives (both work if you're consistent, but $1 - \tanh(v_a)^2$ recomputes what's already stored).
- Forgetting that inputs never reached by the output must still appear in `grad`, as zeros (test 7).

## Interview follow-ups
- Extend the engine so values are arrays and `mul` is a matrix product. Which local derivatives need transposes?
- Why can't you free a node's forward value until the backward pass has passed it, and how does that interact with in-place operations?
"""
