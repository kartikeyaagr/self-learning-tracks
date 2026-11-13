import numpy as np

META = dict(title='Finite-difference slopes of a polynomial', tags=['Calculus'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:01.02'],
            related=['measure-scalar-expression-partials', 'check-product-chain-gradient', 'finite-difference-gradient-checking'])

STATEMENT = r"""
Before building an autograd engine it helps to remember what a derivative *measures*: how much the output moves when you nudge the input by a tiny amount $h$. Three standard ways to turn that into a number are

$$\text{forward} = \frac{f(x+h) - f(x)}{h}, \qquad \text{backward} = \frac{f(x) - f(x-h)}{h}, \qquad \text{central} = \frac{f(x+h) - f(x-h)}{2h}$$

Here $f$ is a polynomial given by its coefficients, **lowest degree first**:

$$f(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_{n-1} x^{n-1}$$

Write `slopes(coeffs, x, h)` that returns a dict with four floats:

| key | value |
|---|---|
| `"forward"` | forward difference at `x` |
| `"backward"` | backward difference at `x` |
| `"central"` | central difference at `x` |
| `"exact"` | the true derivative $f'(x) = \sum_{k \ge 1} k\, c_k x^{k-1}$ |

## Inputs
- `coeffs`: list of floats, length $\ge 1$ (a single entry is a constant function).
- `x`: float. `h`: positive float (tests use $10^{-4} \le h \le 0.5$).

Plain Python only. Every value is compared with a relative tolerance of $10^{-6}$.
"""

STARTER = '''def slopes(coeffs, x, h):
    """Return {"forward", "backward", "central", "exact"} slopes of the polynomial at x."""
    raise NotImplementedError
'''

SOLUTION = '''def slopes(coeffs, x, h):
    def f(t):
        acc = 0.0
        for c in reversed(coeffs):          # Horner: c0 + t(c1 + t(c2 + ...))
            acc = acc * t + c
        return acc

    exact = sum(k * c * x ** (k - 1) for k, c in enumerate(coeffs) if k >= 1)
    fx, up, down = f(x), f(x + h), f(x - h)
    return {
        "forward": (up - fx) / h,
        "backward": (fx - down) / h,
        "central": (up - down) / (2 * h),
        "exact": float(exact),
    }
'''

HINTS = [
    'Write a small helper `f(t)` that evaluates the polynomial, then call it at `x - h`, `x` and `x + h`.',
    'Horner\'s rule evaluates the polynomial with one multiply and one add per coefficient: start from the highest coefficient and repeat `acc = acc * t + c`.',
    'The exact derivative drops $c_0$ and turns each $c_k x^k$ into $k c_k x^{k-1}$: `sum(k * c * x ** (k - 1) for k, c in enumerate(coeffs) if k >= 1)`. Make sure it is a float even for a constant polynomial.',
]

_r = np.random.default_rng(11)
TESTS = [
    ([1.0, -2.0, 0.0, 1.0], 2.0, 0.01),
    ([0.0, 0.0, 1.0], -1.5, 0.1),
    ([7.0], 3.0, 0.001),
    ([2.0, 3.0], 10.0, 0.5),
    ([1.0, 0.0, 0.0, 0.0, 0.0, 2.0], 0.0, 0.01),
    ([0.5, -1.0, 0.25, 0.0, -0.1], -2.0, 0.0001),
    ([float(round(c, 2)) for c in _r.uniform(-3, 3, size=4)], float(round(_r.uniform(-2, 2), 2)), 0.001),
    ([float(round(c, 2)) for c in _r.uniform(-1, 1, size=7)], float(round(_r.uniform(-1, 1), 2)), 0.05),
]

EXPLANATION = r"""
## Why the central difference wins
Expand $f$ around $x$ with Taylor's theorem: $f(x \pm h) = f(x) \pm h f'(x) + \tfrac{h^2}{2} f''(x) \pm \tfrac{h^3}{6} f'''(x) + \dots$

- Forward: $\frac{f(x+h) - f(x)}{h} = f'(x) + \tfrac{h}{2} f''(x) + O(h^2)$. The error shrinks like $h$.
- Backward: the same with the sign of the $h$ term flipped.
- Central: subtracting the two expansions cancels every even-order term, leaving $f'(x) + \tfrac{h^2}{6} f'''(x) + O(h^4)$. The error shrinks like $h^2$.

For a quadratic ($f''' = 0$) the central difference is exact up to rounding, which test 2 shows. Forward and backward always straddle it: their average *is* the central difference.

## Picking $h$
Too large and truncation error dominates. Too small and you subtract two nearly equal floats, so the rounding error (about $\varepsilon |f| / h$ with $\varepsilon \approx 10^{-16}$) blows up. For the central difference the sweet spot is roughly $h \approx \varepsilon^{1/3} \approx 10^{-5}$.

## Complexity
Three polynomial evaluations, each $O(n)$ with Horner's rule, plus $O(n)$ for the exact derivative.

## Common bugs
- Reading the coefficients highest-degree first.
- Dividing the central difference by $h$ instead of $2h$.
- Returning the integer `0` for the exact slope of a constant (harmless here, but sloppy).

## Interview follow-ups
- How would you estimate a second derivative from the same three evaluations? ($\frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$.)
- Why don't deep-learning frameworks use finite differences for training? (One extra forward pass *per parameter*, and limited precision. Reverse mode gets every partial for the cost of about one extra pass.)
"""
