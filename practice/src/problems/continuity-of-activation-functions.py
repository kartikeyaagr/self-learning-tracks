import numpy as np

META = dict(title='Jumps, kinks and smooth joins in piecewise activations', tags=['Activation Functions', 'Calculus'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:01.02'],
            related=['derivatives-of-common-activation-functions', 'subgradients-for-non-smooth-functions'])

STATEMENT = r"""
Many activations and losses are glued together from polynomial pieces: ReLU, ReLU6, hard-tanh, the Huber loss. What happens at each glue point decides how gradient descent behaves there.

A piecewise polynomial is given by sorted breakpoints $b_0 < b_1 < \dots < b_{k-1}$ and $k+1$ pieces. Piece $p$ is $P_p(x) = \sum_i c_{p,i}\, x^i$ (coefficients in **increasing** powers) and applies on

- piece $0$: $x < b_0$
- piece $p$: $b_{p-1} \le x < b_p$
- piece $k$: $x \ge b_{k-1}$

At breakpoint $b_i$ the left piece is $P_i$ and the right piece is $P_{i+1}$. Write `classify_joins(breaks, coeffs)` that labels every breakpoint:

| Label | Condition at $b$ |
|---|---|
| `'jump'` | $\lvert P_i(b) - P_{i+1}(b)\rvert > 10^{-9}$ (not continuous) |
| `'kink'` | continuous, but $\lvert P_i'(b) - P_{i+1}'(b)\rvert > 10^{-9}$ |
| `'smooth'` | continuous and the one-sided derivatives agree |

## Input and output
- `breaks`: list of $k \ge 0$ floats, strictly increasing.
- `coeffs`: list of $k+1$ non-empty lists of floats; the lists can have different lengths.
- Return a Python list of $k$ strings, in breakpoint order (an empty list when $k = 0$).
"""

STARTER = '''import numpy as np

def classify_joins(breaks, coeffs):
    """Return ['jump' | 'kink' | 'smooth'] for each breakpoint."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from numpy.polynomial import polynomial as P

def classify_joins(breaks, coeffs, tol=1e-9):
    out = []
    for i, b in enumerate(breaks):
        left, right = coeffs[i], coeffs[i + 1]
        if abs(P.polyval(b, left) - P.polyval(b, right)) > tol:
            out.append('jump')
        elif abs(P.polyval(b, P.polyder(left)) - P.polyval(b, P.polyder(right))) > tol:
            out.append('kink')            # one-sided derivatives disagree: a corner
        else:
            out.append('smooth')
    return out
'''

HINTS = [
    'Only the two pieces that meet at a breakpoint matter: piece `i` on the left of `breaks[i]`, piece `i + 1` on the right.',
    'Check continuity first (compare the two values at $b$). Only if that passes does it make sense to compare the one-sided derivatives.',
    '`numpy.polynomial.polynomial.polyval(b, c)` evaluates a polynomial with increasing-power coefficients, and `polyder(c)` gives its derivative\'s coefficients.',
]


def _random_continuous(rng, k):
    """Integer pieces, with each constant term shifted so the pieces meet exactly."""
    breaks = sorted(rng.choice(np.arange(-5, 6), size=k, replace=False).tolist())
    coeffs = [rng.integers(-3, 4, size=3).astype(float).tolist() for _ in range(k + 1)]
    for i, b in enumerate(breaks):
        gap = np.polynomial.polynomial.polyval(b, coeffs[i]) - np.polynomial.polynomial.polyval(b, coeffs[i + 1])
        coeffs[i + 1][0] += float(gap)
    return [float(b) for b in breaks], coeffs


_r = np.random.default_rng(5)
TESTS = [
    ([0.0], [[0.0], [0.0, 1.0]]),                                    # ReLU
    ([0.0], [[0.0], [1.0]]),                                         # Heaviside step
    ([-1.0, 1.0], [[-1.0], [0.0, 1.0], [1.0]]),                      # hard tanh
    ([-1.0, 1.0], [[-0.5, -1.0], [0.0, 0.0, 0.5], [-0.5, 1.0]]),     # Huber, delta = 1
    ([0.0, 6.0], [[0.0], [0.0, 1.0], [6.0]]),                        # ReLU6
    ([0.0], [[0.0, 0.01], [0.0, 1.0]]),                              # leaky ReLU
    ([0.0, 1.0], [[0.0], [0.0, 0.0, 0.0, 1.0], [-2.0, 3.0]]),        # x^3, then its tangent line at 1
    ([], [[1.0, 2.0, 3.0]]),
    _random_continuous(_r, 4),
]

EXPLANATION = r"""
## Why two checks, in this order
A function is continuous at $b$ when both one-sided limits equal the value there. Each piece is a polynomial, so its limit at $b$ is just its value: compare $P_i(b)$ with $P_{i+1}(b)$. Differentiability needs continuity first, then equal one-sided derivatives. ReLU passes the first test and fails the second: that's a kink.

## What each label means for training
- **jump** (the step function): the derivative is 0 almost everywhere and undefined at the jump, so gradients carry no signal. That's why the perceptron's step was replaced by sigmoids.
- **kink** (ReLU, hard tanh, L1): gradient descent still works almost everywhere, and at the corner you pick a subgradient.
- **smooth** (Huber at $\pm\delta$): the gradient is continuous, so step sizes behave like on a smooth loss. Huber is built to be quadratic near 0 and linear in the tails while staying $C^1$.

## Complexity
$O(k \cdot \deg)$: two polynomial and two derivative evaluations per breakpoint.

## Common bugs
- Pairing the wrong pieces (piece $i$ with $i-1$), which shifts every label by one.
- Comparing coefficient lists instead of values. `[0.0]` and `[0.0, 0.0]` are the same polynomial.
- Exact `==` on floats: with real-valued coefficients the two sides can differ by $10^{-16}$. That's why the statement fixes a tolerance.

## Interview follow-ups
- Extend the classifier to report the order of smoothness: the largest $n$ such that the first $n$ derivatives all match ($C^n$).
- GELU and SiLU are smooth everywhere. Does that matter for training in practice, or only for second-order methods?
"""
