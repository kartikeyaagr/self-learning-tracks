import numpy as np

META = dict(title='BatchNorm in train and eval mode', tags=['Neural Networks', 'Training', 'Convolutional Networks'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:01.14'],
            related=['layer-normalization', 'layer-norm', 'resnet-residual-block'])

STATEMENT = r"""
Batch normalization standardises each **channel** using statistics of the current mini-batch, then applies a learned scale and shift:

$$\hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma \hat x + \beta.$$

At inference there is no batch to average over, so the layer also keeps exponential moving averages of the statistics during training and uses those instead.

Write `batch_norm(x, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5)` that returns `(y, new_running_mean, new_running_var)`.

## Shapes
- `x` is `(N, C)` or `(N, C, H, W)`; the channel axis is always axis 1. `gamma`, `beta`, `running_mean`, `running_var` are `(C,)`.
- Statistics for channel $c$ are taken over **every axis except 1**: over $M = N$ values for 2-D input and $M = N H W$ for 4-D input.

## Training mode (`training=True`)
- Normalise with the batch mean $\mu_B$ and the **biased** variance $\sigma_B^2$ (divide by $M$).
- Update the running statistics with the **unbiased** variance (divide by $M - 1$; $M \ge 2$ in every test):
$$\text{rm} \leftarrow (1 - m)\,\text{rm} + m\,\mu_B, \qquad \text{rv} \leftarrow (1 - m)\,\text{rv} + m\,\tfrac{M}{M-1}\sigma_B^2.$$

## Eval mode (`training=False`)
- Normalise with `running_mean` and `running_var`; return them unchanged.

## Output
`y` with the shape of `x`, and the two `(C,)` running arrays. Do not modify the inputs in place.
"""

STARTER = '''import numpy as np

def batch_norm(x, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    """Return (y, new_running_mean, new_running_var)."""
    x = np.asarray(x, dtype=float)
    mu, var = x.mean(axis=0), x.var(axis=0)
    y = gamma * (x - mu) / np.sqrt(var + eps) + beta
    return y, running_mean, running_var
'''

SOLUTION = '''import numpy as np

def batch_norm(x, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    x = np.asarray(x, dtype=float)
    axes = (0,) + tuple(range(2, x.ndim))                 # every axis except channels
    shape = (1, -1) + (1,) * (x.ndim - 2)                 # (C,) -> broadcastable against x
    rm, rv = np.asarray(running_mean, dtype=float), np.asarray(running_var, dtype=float)
    if training:
        mu, var = x.mean(axis=axes), x.var(axis=axes)     # biased variance for normalising
        M = x.size // x.shape[1]
        rm = (1 - momentum) * rm + momentum * mu
        rv = (1 - momentum) * rv + momentum * var * M / (M - 1)   # unbiased for the running estimate
    else:
        mu, var, rm, rv = rm, rv, rm.copy(), rv.copy()
    x_hat = (x - mu.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)
    return np.reshape(gamma, shape) * x_hat + np.reshape(beta, shape), rm, rv
'''

HINTS = [
    'Reduce over `axis=(0, 2, 3)` for 4-D input and `axis=0` for 2-D input, then reshape the `(C,)` statistics to `(1, C, 1, 1)` (or `(1, C)`) so they broadcast.',
    'In training mode normalise with `x.var(axis=axes)` (biased) but feed `var * M / (M - 1)` into the running variance, where `M = x.size // C`.',
    'Eval mode is the same normalisation with the running statistics swapped in, and the running arrays come back unchanged. Don\'t forget `gamma` and `beta` need the same reshape.',
]

_r = np.random.default_rng(17)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 6.0]]), np.ones(2), np.zeros(2), np.zeros(2), np.ones(2), True),
    (np.array([[1.0, 2.0], [3.0, 6.0]]), np.array([2.0, 1.0]), np.array([0.0, 1.0]), np.array([2.0, 4.0]), np.array([1.0, 4.0]), False),
    (_r.normal(3, 2, size=(8, 4)), _r.normal(size=4), _r.normal(size=4), np.zeros(4), np.ones(4), True, 0.1),
    (_r.normal(size=(4, 3, 5, 5)) * 4 + 1, _r.normal(size=3), _r.normal(size=3), _r.normal(size=3), _r.uniform(0.5, 2, size=3), True, 0.3),
    (_r.normal(size=(2, 3, 2, 2)), np.ones(3), np.zeros(3), np.array([0.5, -1.0, 0.0]), np.array([2.0, 0.5, 1.0]), False),
    (np.full((3, 2), 7.0), np.ones(2), np.zeros(2), np.zeros(2), np.ones(2), True, 0.5),
    (_r.normal(size=(1, 2, 3, 3)), np.ones(2), np.zeros(2), np.zeros(2), np.ones(2), True, 1.0, 1e-3),
]

EXPLANATION = r"""
## Per-channel statistics
In a conv net every spatial position of a channel is produced by the same filter, so BatchNorm pools over batch **and** spatial axes, giving one $(\mu, \sigma^2)$ per channel. For a fully connected layer the "channels" are the features.

## Biased vs unbiased
Normalising uses the biased $\sigma_B^2$ because that is the variance of the batch actually being normalised. The running estimate is meant to approximate the population variance, so PyTorch feeds it the unbiased $\frac{M}{M-1}\sigma_B^2$. Mixing these up is invisible for large batches and obvious in test 1.

## Constant channels
Test 6 has zero variance: $\hat x = 0/\sqrt{\epsilon} = 0$, so $y = \beta$. That's why $\epsilon$ is inside the square root.

## Complexity
Two passes over the data, $O(NCHW)$, plus $O(C)$ state.

## Common bugs
- Normalising over axis 0 only for 4-D input (per pixel instead of per channel).
- Updating the running stats in eval mode, or using batch stats in eval mode.
- Note PyTorch's `momentum` is the weight on the **new** batch statistic, the opposite of the usual EMA "decay" convention.

## Interview follow-ups
- Derive the backward pass. Why does $\partial L/\partial x$ include terms from every sample in the batch?
- Why do transformers use LayerNorm rather than BatchNorm? (Variable sequence lengths, small per-device batches, and train/eval mismatch.)
"""
