import numpy as np

META = dict(title='Layer normalization (forward)', tags=['Neural Networks', 'Transformers'], diff='Easy', runtime='numpy',
            lessons=['ieng:13.05'], related=['scaled-dot-product-attention'])

STATEMENT = r"""
LayerNorm normalises each token's feature vector to zero mean and unit variance, then applies a learned scale $\gamma$ and shift $\beta$:

$$y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad \mu = \tfrac{1}{d}\textstyle\sum_i x_i,\quad \sigma^2 = \tfrac{1}{d}\sum_i (x_i - \mu)^2$$

Write `layer_norm(x, gamma, beta, eps=1e-5)`.

## Requirements
- `x` has shape `(..., d)`. Normalise over the **last** axis only. `gamma` and `beta` have shape `(d,)`.
- Use the population variance (divide by $d$, not $d-1$), as frameworks do.
- Same output shape as `x`.
"""

STARTER = '''import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    """x: (..., d); gamma, beta: (d,). Normalise over the last axis."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)          # ddof=0: population variance
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
'''

HINTS = [
    'Reduce over `axis=-1` with `keepdims=True`, so the mean and variance broadcast back over the features.',
    '`np.var` defaults to `ddof=0`, which is the population variance this asks for.',
    '`gamma` and `beta` have shape `(d,)`, which broadcasts against `(..., d)` with no reshaping.',
]

_r = np.random.default_rng(9)
d = 6
TESTS = [
    (np.array([[1.0, 2.0, 3.0]]), np.ones(3), np.zeros(3)),
    (np.array([[5.0, 5.0, 5.0, 5.0]]), np.ones(4), np.zeros(4)),
    (_r.normal(size=(3, d)) * 10 + 4, _r.normal(size=d), _r.normal(size=d)),
    (_r.normal(size=(2, 4, d)), np.ones(d) * 2, np.ones(d) * -1),
    (np.array([1.0, -1.0]), np.array([3.0, 3.0]), np.array([0.5, 0.5]), 1e-3),
    (_r.normal(size=(1, 8)) * 1e-4, np.ones(8), np.zeros(8)),
]

EXPLANATION = r"""
## LayerNorm vs BatchNorm
BatchNorm normalises each feature **across the batch**, so it depends on batch statistics and behaves differently in training and at inference. LayerNorm normalises each sample **across its features**. It works with batch size 1 and variable-length sequences, which is why Transformers use it.

## The role of $\epsilon$
A constant row (test 2) has $\sigma^2 = 0$. With $\epsilon$ the output is $\beta$ instead of $0/0$. For tiny-variance rows (last test), $\epsilon$ visibly damps the normalisation.

## RMSNorm
LLaMA-style models drop the mean subtraction and $\beta$: $y = \gamma \odot x / \sqrt{\overline{x^2} + \epsilon}$. It's cheaper and works as well in practice.

## Interview follow-ups
- Pre-norm vs post-norm Transformers: why does pre-norm train more stably at depth?
- Write the backward pass: what is $\partial L / \partial x$ given $\partial L / \partial y$?
"""
