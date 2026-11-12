import numpy as np

META = dict(title='Sigmoid and its derivative', tags=['Activation Functions'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:01.02'], related=['softmax-rows'])

STATEMENT = r"""
The logistic sigmoid squashes any real number into $(0, 1)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \qquad \sigma'(x) = \sigma(x)\,\bigl(1 - \sigma(x)\bigr)$$

Write `sigmoid_and_grad(x)` that returns a tuple `(s, ds)`: the sigmoid of every element of `x` and its derivative at that element.

## Requirements
- `x` is a float array of any shape; both outputs have the same shape.
- Inputs as large as $\pm 1000$ must give finite results, with no `nan` and no overflow warnings. Use the two-branch form: for $x \ge 0$ use $1/(1+e^{-x})$, for $x < 0$ use $e^{x}/(1+e^{x})$.
"""

STARTER = '''import numpy as np

def sigmoid_and_grad(x):
    """x: float array. Return (sigmoid(x), sigmoid'(x)), each the same shape as x."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sigmoid_and_grad(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(-np.abs(x))                  # always in (0, 1]: never overflows
    s = np.where(x >= 0, 1 / (1 + e), e / (1 + e))
    return s, s * (1 - s)
'''

HINTS = [
    'For large negative x, `np.exp(-x)` overflows. Rewrite the formula so the exponent you compute is never positive.',
    '`np.exp(-np.abs(x))` is always between 0 and 1. Both branches can be written in terms of it.',
    'The derivative needs no new exponentials: once you have `s`, it is `s * (1 - s)`.',
]

TESTS = [
    (np.array([0.0]),),
    (np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),),
    (np.array([1000.0, -1000.0]),),
    (np.array([[30.0, -30.0], [0.5, -0.5]]),),
    (np.linspace(-8, 8, 17),),
    (np.array([1e-12, -1e-12]),),
]

EXPLANATION = r"""
## Why two branches
$1/(1+e^{-x})$ is fine for $x \ge 0$ because $e^{-x} \le 1$. For $x < 0$, multiply top and bottom by $e^{x}$ to get $e^{x}/(1+e^{x})$, where again the exponent is negative. Computing `e = exp(-|x|)` once covers both branches.

## The derivative
Differentiate $(1+e^{-x})^{-1}$: $\sigma'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = \sigma(x)(1-\sigma(x))$. In backprop you reuse the forward output, so the backward pass costs one multiply.

## Common bugs
- `np.where(x >= 0, 1/(1+np.exp(-x)), ...)` still evaluates **both** branches on every element, so the overflow warning comes back. Compute from `exp(-|x|)`.
- The derivative peaks at 0.25 (at $x = 0$) and vanishes for large $|x|$. That is the vanishing-gradient problem in deep sigmoid networks.

## Interview follow-ups
- Why did ReLU replace sigmoid in hidden layers? (No saturation for $x>0$, cheaper, sparse activations.)
- How is sigmoid related to softmax? It is a two-class softmax with logits $[x, 0]$.
"""
