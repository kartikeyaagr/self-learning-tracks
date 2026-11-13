import numpy as np

META = dict(title='Gradient check for a long product', tags=['Calculus'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:01.04', 'air-fanout-v1:01.07'],
            related=['measure-scalar-expression-partials', 'finite-difference-gradient-checking', 'gradient-check-autodiff-graph'])

STATEMENT = r"""
For a product of $n$ numbers

$$P(x) = x_0\, x_1 \cdots x_{n-1}, \qquad \frac{\partial P}{\partial x_i} = \prod_{j \ne i} x_j$$

someone has handed you a `claimed` gradient. Check it.

Write `check_product_gradient(xs, claimed, h)` that returns a tuple `(exact, numeric, bad)`:

- `exact`: list of floats, the true partials $\prod_{j \ne i} x_j$. **Do not divide by $x_i$**: inputs can contain zeros. An empty product is `1.0`, so for a single input the gradient is `[1.0]`.
- `numeric`: list of floats, the central differences $\dfrac{P(x + h e_i) - P(x - h e_i)}{2h}$, where $e_i$ shifts only entry $i$.
- `bad`: sorted list of the indices $i$ where the claim fails the check

$$|\text{claimed}_i - \text{numeric}_i| \le 10^{-4} \cdot \max(1, |\text{numeric}_i|)$$

A claimed entry that is `nan` or infinite always fails.

## Inputs
- `xs`: list of $n \ge 1$ floats of moderate size. `claimed`: list of $n$ floats (possibly `nan`/`inf`). `h`: positive float.
- Don't modify `xs`.
"""

STARTER = '''def check_product_gradient(xs, claimed, h):
    """Return (exact partials, central-difference partials, sorted list of failing indices)."""
    raise NotImplementedError
'''

SOLUTION = '''import math

def check_product_gradient(xs, claimed, h):
    n = len(xs)
    # prefix[i] = x0*...*x(i-1), suffix[i] = x(i+1)*...*x(n-1): product of the others without division.
    prefix, suffix = [1.0] * n, [1.0] * n
    for i in range(1, n):
        prefix[i] = prefix[i - 1] * xs[i - 1]
    for i in range(n - 2, -1, -1):
        suffix[i] = suffix[i + 1] * xs[i + 1]
    exact = [prefix[i] * suffix[i] for i in range(n)]

    def P(i, shift):                       # product with only entry i shifted
        out = 1.0
        for j, x in enumerate(xs):
            out *= x + shift if j == i else x
        return out

    numeric = [(P(i, h) - P(i, -h)) / (2 * h) for i in range(n)]
    # "not <=" also catches nan, for which every comparison is False.
    bad = [i for i in range(n)
           if not (math.isfinite(claimed[i]) and abs(claimed[i] - numeric[i]) <= 1e-4 * max(1.0, abs(numeric[i])))]
    return exact, numeric, bad
'''

HINTS = [
    'The obvious `P / xs[i]` breaks as soon as an input is 0. Think of "everything to the left of $i$" times "everything to the right of $i$".',
    'Build prefix products left to right and suffix products right to left, both starting from 1.0. Then `exact[i] = prefix[i] * suffix[i]`.',
    'For the check, `abs(nan - y) <= tol` is False, so write the test as "passes" and collect the indices where it does **not** pass, also guarding with `math.isfinite`.',
]

_r = np.random.default_rng(23)
_xs = [float(round(v, 2)) for v in _r.uniform(-2, 2, size=8)]
_ex = [float(np.prod(_xs[:i] + _xs[i + 1:])) for i in range(8)]
TESTS = [
    ([2.0, 3.0, 4.0], [12.0, 8.0, 6.0], 0.001),
    ([2.0, 0.0, 5.0], [0.0, 10.0, 0.0], 0.01),
    ([2.0, 0.0, 5.0], [float('nan'), float('inf'), float('nan')], 0.01),
    ([3.5], [1.0], 0.1),
    ([0.0, 0.0, 1.5, 2.0], [0.0, 0.0, 0.0, 0.0], 0.001),
    ([1.5, -2.0, 0.5, -1.0], [1.0, -0.75, 3.0, -1.5], 0.001),
    (_xs, _ex, 0.0001),
    (_xs, [g * 2 if i % 3 == 0 else g for i, g in enumerate(_ex)], 0.001),
]

EXPLANATION = r"""
## Why prefix and suffix products
$\prod_{j \ne i} x_j$ splits into the product of everything left of $i$ and everything right of $i$. Two linear sweeps give all $n$ partials in $O(n)$ time without a single division, so zeros are no problem. Test 2 has one zero (only its own partial survives) and test 5 has two (every partial is 0). Dividing $P$ by $x_i$ gives `nan` there, which is exactly what the bad claim in test 3 contains.

This is the same trick as the classic "product of array except self" interview question, and it's what reverse-mode autodiff does on a chain of multiplications: the forward pass builds the prefix, the backward pass carries the suffix.

## Why a relative tolerance
Partials can be huge or tiny depending on the inputs. The check $|g - \hat g| \le 10^{-4} \max(1, |\hat g|)$ is relative for large gradients and absolute near zero, so a true zero gradient isn't failed by a $10^{-13}$ rounding error.

## Complexity
`exact` is $O(n)$. The numeric gradient re-evaluates $P$ twice per coordinate, $O(n^2)$ in total. That gap, $n$ times more work, is the whole case for automatic differentiation.

## Common bugs
- `P / x[i]`, which fails on zeros.
- Shifting `xs[i]` in place and forgetting to restore it, so later coordinates see a modified input.
- `abs(c - n) > tol` as the failure test: it lets `nan` slip through.

## Interview follow-ups
- How would you compute the gradient of $\log |P|$ instead, and why is that numerically nicer for long products? ($\partial/\partial x_i = 1/x_i$, and no overflow.)
- When would you randomly sample a few coordinates instead of checking all $n$?
"""
