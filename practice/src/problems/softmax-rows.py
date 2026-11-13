import numpy as np

META = dict(title='Numerically stable softmax', tags=['Activation Functions'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.02', 'ieng:00.03'], related=['sigmoid-stable', 'softmax-jacobian', 'cross-entropy-logits'])

STATEMENT = r"""
Softmax turns a vector of scores (logits) into a probability distribution:

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Write `softmax(z)` that applies it along the **last axis** of an array of any rank, so a batch of shape `(B, C)` gives one distribution per row.

## Requirements
- Return an array with the same shape as `z`; every slice along the last axis sums to 1.
- It must work for large logits. `[1000, 1001]` is a valid input and must not produce `nan`.
- Use NumPy only, no loops over rows.
"""

STARTER = '''import numpy as np

def softmax(z):
    """z: float array, any shape. Softmax over the last axis, same shape."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def softmax(z):
    z = np.asarray(z, dtype=float)
    shifted = z - z.max(axis=-1, keepdims=True)   # softmax(z) == softmax(z - c); largest entry becomes 0
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)
'''

HINTS = [
    'Adding the same constant to every logit in a row does not change the result. Which constant makes overflow impossible?',
    'Subtract the row maximum, `z.max(axis=-1, keepdims=True)`, before calling `np.exp`. The largest exponent is then e^0 = 1.',
    '`keepdims=True` keeps the reduced axis as size 1 so the subtraction and the division broadcast back over the last axis.',
]

_r = np.random.default_rng(7)
TESTS = [
    (np.array([1.0, 2.0, 3.0]),),
    (np.array([[1.0, 1.0], [0.0, 10.0]]),),
    (np.array([1000.0, 1001.0]),),
    (np.array([-1000.0, -1000.0, -1000.0]),),
    (np.array([[5.0]]),),
    (_r.normal(size=(4, 6)) * 20,),
    (_r.normal(size=(2, 3, 5)),),
]

EXPLANATION = r"""
## Why subtracting the max works
For any constant $c$, $\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c} e^{z_i}}{e^{-c}\sum_j e^{z_j}}$, so the result is unchanged. Picking $c = \max_j z_j$ makes every exponent $\le 0$: nothing overflows, and at least one term is exactly 1, so the denominator can't underflow to 0 either.

## Complexity
One pass for the max, one for `exp`, one for the sum: $O(n)$ time and $O(n)$ extra memory for the output.

## Common bugs
- Taking the max over the whole array instead of per row. That's still correct mathematically but a row far below the global max can underflow to all zeros, giving `0/0 = nan`.
- Forgetting `keepdims=True`, so `(B, C) - (B,)` fails to broadcast or broadcasts along the wrong axis.

## Interview follow-ups
- How does log-softmax avoid computing `log(softmax(z))` directly? (It's `z - logsumexp(z)`.)
- What does temperature do? Divide logits by $T$ first: $T \to 0$ approaches argmax, $T \to \infty$ approaches uniform.
"""
