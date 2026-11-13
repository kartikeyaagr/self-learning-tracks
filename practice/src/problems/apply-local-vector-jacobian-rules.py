import numpy as np

META = dict(title='Local backward rules for scalar ops', tags=['Neural Networks', 'Calculus'], diff='Easy',
            runtime='python', lessons=['air-fanout-v1:01.07', 'air-fanout-v1:01.06'],
            related=['differentiate-tanh-activation', 'run-reverse-mode-autodiff', 'derivatives-of-common-activation-functions'])

STATEMENT = r"""
Reverse-mode autodiff never forms a full Jacobian. Each operation only answers one question: *given the gradient $g$ of the final output with respect to my result, what gradient do I pass to each of my inputs?* For a scalar op $z = f(a, b)$ that's

$$\bar a = g \cdot \frac{\partial z}{\partial a}, \qquad \bar b = g \cdot \frac{\partial z}{\partial b}$$

Write `local_backward(nodes)`. Each entry of `nodes` is `[op, inputs, g]`; return a list with one entry per node: the list of input gradients, in input order.

| op | inputs | result $z$ | gradients to return |
|---|---|---|---|
| `"add"` | `[a, b]` | $a + b$ | $[\bar a, \bar b]$ |
| `"sub"` | `[a, b]` | $a - b$ | $[\bar a, \bar b]$ |
| `"mul"` | `[a, b]` | $a b$ | $[\bar a, \bar b]$ |
| `"div"` | `[a, b]` | $a / b$ | $[\bar a, \bar b]$ |
| `"neg"` | `[a]` | $-a$ | $[\bar a]$ |
| `"pow"` | `[a, k]` | $a^k$, $k$ a constant | $[\bar a]$ only (no gradient for the constant) |
| `"exp"` | `[a]` | $e^a$ | $[\bar a]$ |
| `"log"` | `[a]` | $\ln a$ | $[\bar a]$ |
| `"tanh"` | `[a]` | $\tanh a$ | $[\bar a]$ |
| `"relu"` | `[a]` | $\max(0, a)$ | $[\bar a]$, using slope 0 at $a = 0$ |

All values are floats. Tests keep every op inside its domain ($b \ne 0$, $a > 0$ for `log`, $a^{k-1}$ defined).
"""

STARTER = '''import math

def local_backward(nodes):
    """nodes: list of [op, inputs, g]. Return the list of input-gradient lists."""
    raise NotImplementedError
'''

SOLUTION = '''import math

RULES = {
    "add":  lambda a, b, g: [g, g],
    "sub":  lambda a, b, g: [g, -g],
    "mul":  lambda a, b, g: [g * b, g * a],               # each input gets the other one
    "div":  lambda a, b, g: [g / b, -g * a / (b * b)],
    "neg":  lambda a, g: [-g],
    "pow":  lambda a, k, g: [g * k * a ** (k - 1)],
    "exp":  lambda a, g: [g * math.exp(a)],
    "log":  lambda a, g: [g / a],
    "tanh": lambda a, g: [g * (1 - math.tanh(a) ** 2)],
    "relu": lambda a, g: [g if a > 0 else 0.0],
}

def local_backward(nodes):
    return [[float(v) for v in RULES[op](*inputs, g)] for op, inputs, g in nodes]
'''

HINTS = [
    'Every answer is the upstream gradient `g` times one ordinary derivative. Write the derivative of each op with respect to each input.',
    'A dict from op name to a small function keeps this tidy. The two you are most likely to get wrong: `mul` swaps the inputs ($\\bar a = g b$), and `div` gives $\\bar b = -g a / b^2$.',
    'For `pow`, $\\frac{d}{da} a^k = k a^{k-1}$ and only `a` gets a gradient. For `relu` return `g` when `a > 0` and `0.0` otherwise (including exactly 0).',
]

_r = np.random.default_rng(6)
_u = lambda: float(round(_r.uniform(0.2, 2.0), 3))
_s = lambda: float(round(_r.uniform(-2.0, 2.0), 3))
TESTS = [
    ([["add", [2.0, 3.0], 1.0], ["mul", [2.0, 3.0], 1.0]],),
    ([["tanh", [0.0], 2.0], ["relu", [-1.0], 5.0], ["relu", [1.0], 5.0]],),
    ([["sub", [1.0, 4.0], 0.5], ["neg", [7.0], -2.0], ["div", [3.0, 2.0], 1.0]],),
    ([["pow", [3.0, 2], 1.0], ["pow", [4.0, 0.5], 1.0], ["pow", [2.0, -1], 3.0], ["pow", [5.0, 0], 1.0]],),
    ([["exp", [0.0], 1.0], ["exp", [1.0], -1.0], ["log", [0.25], 2.0]],),
    ([["relu", [0.0], 1.0], ["mul", [0.0, 0.0], 9.0], ["add", [1.0, 2.0], 0.0]],),
    ([["tanh", [25.0], 1.0], ["div", [0.0, -4.0], 2.0]],),
    ([[op, [_s(), _u()] if op in ("add", "sub", "mul", "div") else [_u()], _s()]
      for op in ["add", "sub", "mul", "div", "neg", "exp", "log", "tanh", "relu"]],),
]

EXPLANATION = r"""
## Why only local rules
The chain rule says $\frac{\partial L}{\partial a} = \frac{\partial L}{\partial z}\frac{\partial z}{\partial a}$. The first factor is the $g$ that arrives from above; the second depends only on this op and its inputs. So each op can be written in isolation, and the engine just has to call them in the right order. For a vector op the second factor is a Jacobian matrix, and "multiply the upstream vector by the Jacobian" is the **vector-Jacobian product** (VJP) that gives this problem its name. With scalars the Jacobian is $1 \times 1$ or $1 \times 2$.

## Patterns worth remembering
- `add` copies $g$ to both inputs; `sub` copies it with a sign flip on the right.
- `mul` swaps: each input is scaled by the *other* input. This is why the forward values must be kept for the backward pass.
- `exp` and `tanh` can be written using the op's own *output* ($e^a$ and $1 - z^2$), so engines store outputs, not inputs.
- `relu` is not differentiable at 0. Any value in $[0, 1]$ is a valid subgradient; frameworks pick 0, as here.

## Complexity
$O(1)$ per node.

## Common bugs
- `div`: forgetting the minus sign or the square in $-a/b^2$.
- `pow`: returning a gradient for the exponent, or computing $k a^k$.
- Dropping the upstream $g$, which only shows up when $g \ne 1$.

## Interview follow-ups
- What's the VJP of matrix multiplication $Z = AB$ with respect to $A$ and $B$? ($\bar A = \bar Z B^\top$, $\bar B = A^\top \bar Z$.)
- If the exponent in `pow` were a variable too, what would its gradient be? ($g \cdot a^k \ln a$, defined for $a > 0$.)
"""
