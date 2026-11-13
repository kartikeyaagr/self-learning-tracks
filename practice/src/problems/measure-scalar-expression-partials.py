import numpy as np

META = dict(title='Nudge each input of a small expression', tags=['Calculus'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:01.04'],
            related=['estimate-scalar-derivative', 'check-product-chain-gradient', 'build-scalar-expression-graph'])

STATEMENT = r"""
A partial derivative answers: if I nudge *one* input and hold the others fixed, how fast does the output move? This is the quantity every autograd engine computes, so it pays to measure it by hand first.

Take the four-input expression

$$L(a, b, c, f) = (a\,b + c)\cdot f$$

Write `nudge_partials(a, b, c, f, h)` that returns a dict

```
{"L": L, "numeric": {"a": .., "b": .., "c": .., "f": ..}, "exact": {"a": .., "b": .., "c": .., "f": ..}}
```

- `"L"`: the value of the expression at the given point.
- `"numeric"`: forward differences. For input $a$ this is $\dfrac{L(a+h, b, c, f) - L(a, b, c, f)}{h}$, and likewise for the others (nudge one input, keep the rest).
- `"exact"`: the analytic partials, derived by hand.

## Inputs
All five arguments are floats, $h > 0$ (tests use $h \ge 10^{-4}$). Return plain floats.
"""

STARTER = '''def nudge_partials(a, b, c, f, h):
    """Return {"L": float, "numeric": {name: slope}, "exact": {name: partial}} for L = (a*b + c) * f."""
    raise NotImplementedError
'''

SOLUTION = '''def nudge_partials(a, b, c, f, h):
    L = lambda a, b, c, f: (a * b + c) * f
    base = L(a, b, c, f)
    numeric = {
        "a": (L(a + h, b, c, f) - base) / h,
        "b": (L(a, b + h, c, f) - base) / h,
        "c": (L(a, b, c + h, f) - base) / h,
        "f": (L(a, b, c, f + h) - base) / h,
    }
    # d = a*b + c feeds L = d*f, so each partial is (local slope) x (slope of L w.r.t. d) = local x f.
    exact = {"a": b * f, "b": a * f, "c": f, "f": a * b + c}
    return {"L": base, "numeric": numeric, "exact": exact}
'''

HINTS = [
    'Wrap the expression in a little function of four arguments so you can re-evaluate it with one input shifted by `h`.',
    'For the exact partials, name the inner sum $d = ab + c$. Then $\\partial L/\\partial f = d$ and $\\partial L/\\partial d = f$.',
    'Chain rule: $\\partial L/\\partial a = \\partial L/\\partial d \\cdot \\partial d/\\partial a = f \\cdot b$. Likewise $\\partial L/\\partial b = f a$ and $\\partial L/\\partial c = f$.',
]

_r = np.random.default_rng(5)
_pt = lambda: [float(round(v, 2)) for v in _r.uniform(-4, 4, size=4)]
TESTS = [
    (2.0, -3.0, 10.0, -2.0, 0.001),
    (1.0, 1.0, 0.0, 1.0, 0.01),
    (0.0, 0.0, 0.0, 0.0, 0.001),
    (5.0, 4.0, -20.0, 3.0, 0.0001),
    (-1.5, 2.5, 0.75, 0.5, 0.1),
    (*_pt(), 0.001),
    (*_pt(), 0.0001),
]

EXPLANATION = r"""
## Why forward differences are (nearly) exact here
$L$ is linear in each input on its own: hold $b, c, f$ fixed and $L = (bf)\,a + cf$ is a straight line in $a$. A forward difference of a straight line is its slope exactly, so `numeric` and `exact` agree up to floating-point rounding. That's a special property of this expression. For curved functions you'd see an $O(h)$ gap, as in the polynomial problem.

## The chain rule in miniature
The exact partials show the pattern autograd automates. The intermediate $d = ab + c$ receives the gradient $\partial L / \partial d = f$ from above, and each of its inputs multiplies that by its own local slope ($b$, $a$ or $1$). An addition passes the incoming gradient through unchanged; a multiplication scales it by the *other* operand.

## Complexity
Five evaluations of the expression: one base value plus one per input. With $n$ inputs a finite-difference gradient costs $n + 1$ evaluations, which is why it doesn't scale to millions of parameters.

## Common bugs
- Nudging two inputs at once by reusing a variable you already shifted.
- Computing the base value after mutating an input.
- Mixing up $\partial L/\partial a = bf$ and $\partial L/\partial b = af$.

## Interview follow-ups
- What sign does `numeric["c"]` have, and what does that tell you about how to change $c$ to *decrease* $L$?
- How many evaluations would the central difference need for all four partials, and what would it buy you on this expression? (Eight; nothing, because $L$ is linear in each input.)
"""
