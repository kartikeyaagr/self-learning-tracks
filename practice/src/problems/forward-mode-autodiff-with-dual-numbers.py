import numpy as np

META = dict(title='Forward-mode autodiff with dual numbers', tags=['Calculus'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:01.06', 'air-fanout-v1:01.07'],
            related=['backpropagation-through-computation-graph', 'jvp-and-vjp-for-2-layer-network',
                     'finite-difference-gradient-checking'])

STATEMENT = r"""
A **dual number** $a + a'\varepsilon$ with $\varepsilon^2 = 0$ carries a value and a derivative together. Arithmetic on duals applies the differentiation rules automatically:

$$(a + a'\varepsilon)(b + b'\varepsilon) = ab + (a'b + ab')\varepsilon, \qquad f(a + a'\varepsilon) = f(a) + f'(a)\,a'\varepsilon$$

Seed the input as $x + 1\varepsilon$, evaluate an expression, and the $\varepsilon$ part of the result is $f'(x)$: exact, with no step size.

Write `rpn_dual(tokens, x)` that evaluates an expression in **reverse Polish notation** with dual numbers and returns `(value, deriv)`.

## Tokens
| Token | Meaning |
|---|---|
| `'x'` | the input variable |
| a numeric string such as `'2'`, `'-0.5'`, `'1e-3'` | a constant |
| `'+'`, `'-'`, `'*'`, `'/'` | binary: pop $b$, then pop $a$, push $a \circ b$ |
| `'^'` | binary power $a^b$; the right operand $b$ is always a constant |
| `'neg'`, `'sin'`, `'cos'`, `'exp'`, `'log'`, `'tanh'`, `'sqrt'` | unary: pop $a$, push $f(a)$ |

For example `['x', '2', '^', 'sin']` is $\sin(x^2)$, and `['1', 'x', '-']` is $1 - x$.

## Inputs and output
- `tokens`: a valid RPN expression (list of strings) that leaves exactly one item on the stack.
- `x`: float array of any shape; evaluate at every element.
- Return two float arrays with the same shape as `x`, even when the expression doesn't use `'x'` (the derivative is then 0).
- Test inputs keep every operation inside its domain.
"""

STARTER = '''import numpy as np

def rpn_dual(tokens, x):
    """Evaluate an RPN expression on dual numbers. Return (f(x), f'(x)), each shaped like x."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

# unary op -> (f, f'); the chain rule multiplies f' by the incoming derivative
UNARY = {
    'neg': (np.negative, lambda v: -np.ones_like(v)),
    'sin': (np.sin, np.cos),
    'cos': (np.cos, lambda v: -np.sin(v)),
    'exp': (np.exp, np.exp),
    'log': (np.log, lambda v: 1 / v),
    'tanh': (np.tanh, lambda v: 1 - np.tanh(v) ** 2),
    'sqrt': (np.sqrt, lambda v: 0.5 / np.sqrt(v)),
}

def rpn_dual(tokens, x):
    x = np.asarray(x, dtype=float)
    stack = []                                       # each item is a dual number (value, derivative)
    for tok in tokens:
        if tok == 'x':
            stack.append((x, np.ones_like(x)))      # seed: dx/dx = 1
        elif tok in UNARY:
            f, df = UNARY[tok]
            v, d = stack.pop()
            stack.append((f(v), df(v) * d))
        elif tok in ('+', '-', '*', '/', '^'):
            (b, db), (a, da) = stack.pop(), stack.pop()
            if tok == '+':
                stack.append((a + b, da + db))
            elif tok == '-':
                stack.append((a - b, da - db))
            elif tok == '*':
                stack.append((a * b, da * b + a * db))              # product rule
            elif tok == '/':
                stack.append((a / b, (da * b - a * db) / b ** 2))   # quotient rule
            else:
                stack.append((a ** b, b * a ** (b - 1) * da))       # power rule, b constant
        else:
            stack.append((np.full_like(x, float(tok)), np.zeros_like(x)))   # constants have derivative 0
    return stack.pop()
'''

HINTS = [
    'Represent a dual number as a pair `(value, derivative)` of arrays. `x` is `(x, ones)` and a constant `c` is `(full(c), zeros)`. Then evaluate RPN with an ordinary stack.',
    'Each operator combines pairs with a calculus rule: sum rule for `+`/`-`, product rule for `*`, quotient rule for `/`, power rule for `^`, and the chain rule $f\'(a)\\,a\'$ for the unary functions.',
    'Pop order matters for the non-commutative operators: the first `pop()` is the **right** operand $b$. Keep a dict from unary token to `(f, f_prime)` to avoid a long if-chain.',
]

_x = np.array([0.5, 1.0, 2.0])
TESTS = [
    (['x', 'x', '*'], np.array([1.0, 2.0, 3.0])),
    (['x', 'sin', 'x', '*'], np.array([0.0, 1.0])),
    (['2', '3', '+'], _x),
    (['1', 'x', '-', 'x', '/'], _x),
    (['x', '2', '^', '1', '+', 'log'], np.array([[-1.0, 0.0], [1.0, 3.0]])),
    (['1', '1', 'x', 'neg', 'exp', '+', '/'], np.linspace(-4, 4, 9)),
    (['x', 'sqrt', 'x', '3', '^', '*', 'x', 'cos', '-'], _x),
    (['x', 'tanh', 'x', 'exp', '/', '0.5', '^'], np.array([0.25, 1.0, 1.5])),
    (['x', 'x', 'sin', 'cos', '*', 'x', '-1.5', '^', '+'], np.random.default_rng(59).uniform(0.5, 3.0, size=6)),
]

EXPLANATION = r"""
## Why dual numbers give exact derivatives
Expand any smooth $f$ around $a$: $f(a + a'\varepsilon) = f(a) + f'(a)a'\varepsilon + \tfrac12 f''(a)(a'\varepsilon)^2 + \dots$, and every term past the first order vanishes because $\varepsilon^2 = 0$. So each operation propagates exactly the first-order term, which is the chain rule. There's no truncation error and no step size, unlike finite differences: the derivative is as accurate as the function value.

## Forward mode in general
This is forward-mode automatic differentiation: one pass computes $f$ and the derivative along one input direction (a JVP). For $f:\mathbb R^n \to \mathbb R$ you'd need $n$ passes to get the gradient, which is why training uses reverse mode instead. Forward mode wins when inputs are few and outputs many, and for Hessian-vector products (forward-over-reverse).

## Complexity
$O(\text{len(tokens)})$ array operations, each costing $O(|x|)$, so about 2-3 times the cost of evaluating $f$ alone.

## Common bugs
- Popping operands in the wrong order, which turns $1 - x$ into $x - 1$ and $1/x$ into $x$.
- Writing the quotient rule as $(a'b + ab')/b^2$.
- Returning a scalar for constant expressions instead of an array shaped like `x`.

## Interview follow-ups
- Extend `'^'` to a non-constant exponent: $\frac{d}{dx}a^b = a^b\left(b'\log a + b\,a'/a\right)$.
- How would you compute second derivatives? (Hyper-dual numbers, or duals whose components are themselves duals.)
"""
