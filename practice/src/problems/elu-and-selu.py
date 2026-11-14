import numpy as np

META = dict(title='ELU and SELU activations with their derivatives', tags=['Activation Functions'], diff='Easy', runtime='numpy',
            related=['leaky-relu', 'relu', 'gelu', 'swish'])

STATEMENT = r"""
Write `elu_selu(x, alpha=1.0)` returning a dict with four arrays of the same shape as `x`:

- `"elu"`: $\mathrm{ELU}(x) = x$ for $x > 0$, $\alpha(e^x - 1)$ otherwise;
- `"elu_grad"`: its derivative, $1$ for $x > 0$, $\alpha e^x$ otherwise;
- `"selu"`: $\lambda\,\mathrm{ELU}_{\alpha_s}(x)$ with the fixed constants $\lambda = 1.0507009873554805$ and $\alpha_s = 1.6732632423543772$ (ignore the `alpha` argument here);
- `"selu_grad"`: its derivative.

Use `np.expm1` for $e^x - 1$ and avoid overflow warnings for large positive $x$ (do not evaluate `exp` on large positive inputs).
"""

STARTER = '''import numpy as np

def elu_selu(x, alpha=1.0):
    """Return {"elu", "elu_grad", "selu", "selu_grad"}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

LAM, ALPHA_S = 1.0507009873554805, 1.6732632423543772

def elu_selu(x, alpha=1.0):
    x = np.asarray(x, dtype=float)
    neg = np.minimum(x, 0.0)                 # exp only ever sees values <= 0
    pos = x > 0
    elu = np.where(pos, x, alpha * np.expm1(neg))
    elu_g = np.where(pos, 1.0, alpha * np.exp(neg))
    selu = LAM * np.where(pos, x, ALPHA_S * np.expm1(neg))
    selu_g = LAM * np.where(pos, 1.0, ALPHA_S * np.exp(neg))
    return {"elu": elu, "elu_grad": elu_g, "selu": selu, "selu_grad": selu_g}
'''

HINTS = [
    '`np.where` evaluates both branches, so compute the exponential on `np.minimum(x, 0)` to keep it from overflowing.',
    'For x <= 0 the derivative of alpha * (e^x - 1) is alpha * e^x, which equals ELU(x) + alpha.',
    'SELU is ELU with a specific alpha, multiplied by a specific scale lambda.',
]

TESTS = [
    (np.array([-2.0, -0.5, 0.0, 0.5, 2.0]),),
    (np.array([-1.0, 1.0]), 0.5),
    (np.array([[1000.0, -1000.0], [1e-8, -1e-8]]),),
    (np.linspace(-5, 5, 11), 2.0),
    (np.array([0.0]),),
]

EXPLANATION = r"""
## Why ELU
ReLU outputs exactly zero for negative inputs, so the mean activation is positive and some units stop learning. ELU's smooth negative branch saturates at $-\alpha$, pushes mean activations towards zero and keeps a non-zero gradient for moderately negative inputs.

## Why SELU's constants
Klambauer et al. chose $\lambda$ and $\alpha$ so that, for standardised inputs and suitably initialised weights (LeCun normal), activations converge towards zero mean and unit variance through many layers: self-normalising networks. The property depends on that initialisation and on using alpha-dropout instead of ordinary dropout.

## At zero
With $\alpha = 1$, ELU is differentiable at 0 (both one-sided derivatives are 1). For other $\alpha$ the derivative jumps, and the convention here uses the left branch at exactly 0.

## Interview follow-ups
- Why did ELU and SELU lose ground to GELU and SiLU in transformers?
"""
