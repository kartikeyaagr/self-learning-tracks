import numpy as np

META = dict(title='Derivatives of six activations', tags=['Activation Functions', 'Calculus'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:01.06', 'air-fanout-v1:01.02'],
            related=['sigmoid-stable', 'continuity-of-activation-functions', 'chain-rule-through-3-layer-network'])

STATEMENT = r"""
Backprop through an activation layer multiplies the incoming gradient by $f'(x)$ elementwise. Write `activation_grad(x, name)` that returns $f'(x)$ for every element of `x`.

| `name` | $f(x)$ | Convention |
|---|---|---|
| `'relu'` | $\max(0, x)$ | $f'(0) = 0$ |
| `'leaky_relu'` | $x$ if $x > 0$, else $0.01x$ | $f'(0) = 0.01$ |
| `'elu'` | $x$ if $x > 0$, else $e^x - 1$ | |
| `'tanh'` | $\tanh x$ | |
| `'softplus'` | $\log(1 + e^x)$ | |
| `'silu'` | $x\,\sigma(x)$ | $\sigma$ is the logistic sigmoid |

## Requirements
- `x`: float array of any shape, with $|x| \le 50$. Return a float array of the same shape.
- Work out each derivative by hand and express it with NumPy; no finite differences.
- No Python loops over elements.
"""

STARTER = '''import numpy as np

def activation_grad(x, name):
    """Elementwise derivative of the named activation at x."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def activation_grad(x, name):
    x = np.asarray(x, dtype=float)
    sig = 1 / (1 + np.exp(-x))                          # safe for |x| <= 50
    if name == 'relu':
        return (x > 0).astype(float)
    if name == 'leaky_relu':
        return np.where(x > 0, 1.0, 0.01)
    if name == 'elu':
        return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0)))   # e^x on the left; min avoids e^50 waste
    if name == 'tanh':
        return 1 - np.tanh(x) ** 2
    if name == 'softplus':
        return sig                                       # d/dx log(1 + e^x) = e^x / (1 + e^x)
    if name == 'silu':
        return sig * (1 + x * (1 - sig))                 # product rule: sigma + x * sigma * (1 - sigma)
    raise ValueError(name)
'''

HINTS = [
    'The piecewise ones (`relu`, `leaky_relu`, `elu`) are an `np.where` on `x > 0`. Note that the convention at 0 follows the left piece.',
    'Softplus differentiates to the sigmoid. For tanh, use $\\tanh\' = 1 - \\tanh^2$.',
    'SiLU needs the product rule: $\\frac{d}{dx}[x\\sigma(x)] = \\sigma(x) + x\\,\\sigma(x)(1-\\sigma(x))$.',
]

_r = np.random.default_rng(21)
_x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
TESTS = [
    (_x, 'relu'),
    (_x, 'tanh'),
    (_x, 'leaky_relu'),
    (_x, 'elu'),
    (_x, 'softplus'),
    (_x, 'silu'),
    (np.array([[-50.0, 50.0], [-20.0, 20.0]]), 'silu'),
    (np.array([[-50.0, 50.0], [-20.0, 20.0]]), 'elu'),
    (_r.normal(size=(3, 4)) * 4, 'silu'),
    (_r.normal(size=(6,)) * 4, 'tanh'),
]

EXPLANATION = r"""
## The derivatives
- **ReLU**: 1 for $x > 0$, 0 for $x < 0$. At 0 it isn't differentiable; frameworks pick 0, a valid subgradient.
- **Leaky ReLU**: 1 or 0.01. The small slope keeps gradients flowing for negative inputs ("dead ReLU" fix).
- **ELU**: 1 for $x > 0$, $e^x$ for $x \le 0$. Both sides give 1 at 0, so ELU is $C^1$, unlike ReLU.
- **tanh**: $1 - \tanh^2 x$, peaks at 1 and vanishes in the tails.
- **Softplus**: $\sigma(x)$. Softplus is a smooth ReLU, and its derivative is a smooth step.
- **SiLU / Swish**: $\sigma(x)\bigl(1 + x(1 - \sigma(x))\bigr)$. It dips slightly below 0 around $x \approx -2.4$ and slightly above 1 for moderate positive $x$, which is why SiLU is non-monotonic.

## Reuse in backprop
Every one of these can be computed from the forward output or a single sigmoid, so the backward pass costs a few multiplies per element. That's one reason these activations won out over ones whose derivatives need extra transcendental calls.

## Common bugs
- `x >= 0` for ReLU, which gives $f'(0) = 1$, against the stated convention.
- Forgetting the product rule for SiLU and returning just $\sigma(x)$.
- `np.where(x > 0, 1, np.exp(x))` evaluates `exp` everywhere; fine here, but on unbounded inputs it overflows for large positive $x$ even though that branch is discarded.

## Interview follow-ups
- Derive the derivative of GELU, $x\,\Phi(x)$. ($\Phi(x) + x\,\varphi(x)$.)
- Why can a network with all-ReLU hidden layers still represent smooth-looking functions?
"""
