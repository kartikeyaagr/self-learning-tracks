import numpy as np

META = dict(title='Seven activations, overflow-free', tags=['Activation Functions'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:05.02'],
            related=['derivatives-of-common-activation-functions', 'sigmoid-stable', 'softmax-rows'])

STATEMENT = r"""
Write `activate(x, name)` that applies one of these activations elementwise.

| `name` | $f(x)$ |
|---|---|
| `'relu'` | $\max(0, x)$ |
| `'leaky_relu'` | $x$ if $x > 0$, else $0.01\,x$ |
| `'elu'` | $x$ if $x > 0$, else $e^{x} - 1$ |
| `'softplus'` | $\log(1 + e^{x})$ |
| `'silu'` | $x\,\sigma(x)$, with $\sigma(x) = 1/(1 + e^{-x})$ |
| `'gelu_tanh'` | $\tfrac12 x\left(1 + \tanh\!\left(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right)\right)$ |
| `'mish'` | $x \tanh(\mathrm{softplus}(x))$ |

## Requirements
- `x`: float array of any shape with $|x| \le 1000$. Return a float array of the same shape.
- The output must be finite everywhere, and NumPy must not emit overflow warnings: a naive `np.exp(1000)` is `inf`. Compute every branch in a form that cannot overflow.
- No Python loops over elements.
"""

STARTER = '''import numpy as np

def activate(x, name):
    """x: float array; name: one of relu, leaky_relu, elu, softplus, silu, gelu_tanh, mish."""
    x = np.asarray(x, dtype=float)
    if name == 'relu':
        return np.maximum(x, 0.0)
    raise NotImplementedError(name)
'''

SOLUTION = '''import numpy as np

def _softplus(x):
    # log(1 + e^x) = max(x, 0) + log1p(e^{-|x|}); the exponent is never positive
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

def _sigmoid(x):
    e = np.exp(-np.abs(x))                       # in (0, 1], never overflows
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

def activate(x, name):
    x = np.asarray(x, dtype=float)
    if name == 'relu':
        return np.maximum(x, 0.0)
    if name == 'leaky_relu':
        return np.where(x > 0, x, 0.01 * x)
    if name == 'elu':
        return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))   # clamp so the unused branch can't overflow
    if name == 'softplus':
        return _softplus(x)
    if name == 'silu':
        return x * _sigmoid(x)
    if name == 'gelu_tanh':
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))   # tanh saturates, no overflow
    if name == 'mish':
        return x * np.tanh(_softplus(x))
    raise ValueError(name)
'''

HINTS = [
    '`np.where(cond, a, b)` evaluates **both** `a` and `b` on the whole array, so the branch you discard can still overflow and warn.',
    'Softplus: $\\log(1 + e^x) = \\max(x, 0) + \\log(1 + e^{-|x|})$. Sigmoid: compute $e^{-|x|}$ once and pick $1/(1+e)$ or $e/(1+e)$ by sign.',
    'For ELU feed `np.minimum(x, 0)` into `np.expm1`, so positive inputs never reach the exponential. GELU and Mish only go through `tanh`, which saturates safely.',
]

_r = np.random.default_rng(3)
_x = np.array([-1000.0, -20.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 20.0, 1000.0])
_m = _r.normal(scale=4, size=(3, 4))
TESTS = [
    (np.array([-2.0, 0.0, 3.0]), 'leaky_relu'),
    (np.array([-1.0, 0.0, 1.0]), 'softplus'),
    (_x, 'elu'),
    (_x, 'softplus'),
    (_x, 'silu'),
    (_x, 'gelu_tanh'),
    (_x, 'mish'),
    (_m, 'relu'),
    (_m, 'gelu_tanh'),
    (_m, 'mish'),
]

EXPLANATION = r"""
## Where overflow hides
Most of these functions contain $e^x$ somewhere. For $x = 1000$, `np.exp(x)` is `inf`, and `inf/inf` or `0 * inf` turns into `nan`. The fix is always to rewrite the formula so the exponent is non-positive: the $\max(x,0) + \operatorname{log1p}(e^{-|x|})$ form of softplus, the two-branch sigmoid, and clamping the input of ELU's negative branch.

`np.where` is not lazy. It computes both branches on every element and only then selects, which is why the discarded branch has to be safe too.

## Small inputs
`np.expm1` and `np.log1p` keep full precision near 0, where `exp(x) - 1` and `log(1 + y)` lose digits to cancellation.

## Which activation where
- ReLU: cheap, sparse, but units can die. Leaky ReLU and ELU keep a gradient for $x < 0$.
- GELU (tanh form used by GPT-2 and BERT) and SiLU/Swish: smooth, slightly non-monotone, the defaults in transformers and modern CNNs.
- Mish: $x\tanh(\mathrm{softplus}(x))$, similar shape to SiLU, used in some detectors.

## Interview follow-ups
- Why is the tanh approximation of GELU used instead of the exact $x\,\Phi(x)$? (Historically faster than `erf`; today most frameworks offer both.)
- Which of these is not zero-centred, and why does that matter for the next layer's gradients?
"""
