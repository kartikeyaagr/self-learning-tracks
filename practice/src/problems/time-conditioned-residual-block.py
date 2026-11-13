import numpy as np

META = dict(title='Time-conditioned residual block (GroupNorm, SiLU)', tags=['U-Net', 'Conditioning', 'Neural Networks'], diff='Medium',
            runtime='numpy', lessons=[], related=['sinusoidal-timestep-embedding', 'tiny-u-net-forward-pass', 'layer-norm'])

STATEMENT = r"""
The workhorse of a diffusion U-Net is a residual block that injects the timestep embedding between two normalised convolutions. To keep this NumPy-sized, the convolutions here are **1x1** (pure channel mixing), but everything else matches the real block.

$$
\begin{aligned}
h &= W_1 \cdot \mathrm{SiLU}(\mathrm{GN}(x)) + b_1 \\
h &= h + \big(W_t\,\mathrm{SiLU}(e) + b_t\big) \quad\text{(broadcast over } H, W) \\
h &= W_2 \cdot \mathrm{SiLU}(\mathrm{GN}(h)) + b_2 \\
\text{out} &= x + h
\end{aligned}
$$

Write `res_block(x, temb, params, groups)`.

## Definitions
- $\mathrm{SiLU}(z) = z\,\sigma(z) = z / (1 + e^{-z})$.
- $\mathrm{GN}$ is GroupNorm **without** learned scale or shift: split the $C$ channels into `groups` contiguous groups of $C/G$ channels; for each sample and group, subtract the mean and divide by $\sqrt{\mathrm{var} + 10^{-5}}$, where mean and (population, `ddof=0`) variance are taken over that group's channels **and** all $H \times W$ positions.
- $W \cdot h$ is a 1x1 convolution: `out[b, o, y, x] = sum_i W[o, i] * h[b, i, y, x]`, plus a per-channel bias.

## Inputs
- `x`: `(B, C, H, W)`; `temb`: `(B, E)` timestep embeddings.
- `params`: dict with `W1` `(C, C)`, `b1` `(C,)`, `Wt` `(C, E)`, `bt` `(C,)`, `W2` `(C, C)`, `b2` `(C,)`.
- `groups`: a positive integer dividing $C$.

## Output
`(B, C, H, W)` float array.
"""

STARTER = '''import numpy as np

def res_block(x, temb, params, groups):
    """x: (B, C, H, W), temb: (B, E). Return x + residual branch, same shape as x."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def silu(z):
    return z / (1.0 + np.exp(-z))

def group_norm(h, groups, eps=1e-5):
    B, C, H, W = h.shape
    g = h.reshape(B, groups, C // groups, H, W)
    mu = g.mean(axis=(2, 3, 4), keepdims=True)
    var = g.var(axis=(2, 3, 4), keepdims=True)
    return ((g - mu) / np.sqrt(var + eps)).reshape(B, C, H, W)

def conv1x1(h, W, b):
    return np.einsum('oi,bihw->bohw', W, h) + b[None, :, None, None]

def res_block(x, temb, params, groups):
    x, temb = np.asarray(x, dtype=float), np.asarray(temb, dtype=float)
    p = {k: np.asarray(v, dtype=float) for k, v in params.items()}
    h = conv1x1(silu(group_norm(x, groups)), p['W1'], p['b1'])
    t = silu(temb) @ p['Wt'].T + p['bt']               # (B, C): one shift per sample and channel
    h = h + t[:, :, None, None]
    h = conv1x1(silu(group_norm(h, groups)), p['W2'], p['b2'])
    return x + h
'''

HINTS = [
    'Write three helpers first: `silu`, `group_norm` and a 1x1 `conv` (an `einsum` over the channel axis). The block is then four lines.',
    'For GroupNorm, reshape `(B, C, H, W)` to `(B, G, C//G, H, W)` and take mean and variance over axes `(2, 3, 4)`. Reshape back afterwards.',
    'The time projection `silu(temb) @ Wt.T + bt` is `(B, C)`. Index it as `[:, :, None, None]` so it adds the same value at every pixel of a channel.',
]

_r = np.random.default_rng(31)

def _params(C, E, zero_out=False):
    p = {'W1': _r.normal(size=(C, C)) / np.sqrt(C), 'b1': _r.normal(size=C) * 0.1,
         'Wt': _r.normal(size=(C, E)) / np.sqrt(E), 'bt': _r.normal(size=C) * 0.1,
         'W2': _r.normal(size=(C, C)) / np.sqrt(C), 'b2': _r.normal(size=C) * 0.1}
    if zero_out:
        p['W2'], p['b2'] = np.zeros((C, C)), np.zeros(C)
    return p

_eye = {'W1': np.eye(2), 'b1': np.zeros(2), 'Wt': np.zeros((2, 1)), 'bt': np.zeros(2), 'W2': np.eye(2), 'b2': np.zeros(2)}
TESTS = [
    (np.array([[[[1.0, -1.0]], [[2.0, 0.0]]]]), np.array([[0.0]]), _eye, 1),
    (np.array([[[[1.0, -1.0]], [[2.0, 0.0]]]]), np.array([[0.0]]), _eye, 2),
    (_r.normal(size=(2, 4, 3, 3)), _r.normal(size=(2, 5)), _params(4, 5, zero_out=True), 2),
    (_r.normal(size=(2, 4, 3, 3)), _r.normal(size=(2, 5)), _params(4, 5), 2),
    (_r.normal(size=(1, 6, 2, 4)) * 3 + 1, _r.normal(size=(1, 3)), _params(6, 3), 3),
    (_r.normal(size=(3, 4, 2, 2)), _r.normal(size=(3, 8)), _params(4, 8), 1),
    (_r.normal(size=(2, 4, 2, 2)), _r.normal(size=(2, 8)), _params(4, 8), 4),
]

EXPLANATION = r"""
## Why it is built this way
- **GroupNorm** normalises within each sample, so it behaves the same for batch size 1 at sampling time as during training (BatchNorm does not). `groups = 1` is LayerNorm over $(C, H, W)$; `groups = C` is InstanceNorm.
- **Adding the time projection** shifts each channel by a step-dependent amount, so the same weights can behave differently at high and low noise. Stronger variants predict a scale and a shift (FiLM / AdaGN).
- **The residual path** lets the block start as the identity. Real implementations zero-initialise the last convolution, which is test 3: the output equals `x` exactly.

## Complexity
$O(B\,C^2\,H W + B\,C\,E)$ for the two channel mixes and the time projection.

## Common bugs
- Normalising over $H, W$ only (that is InstanceNorm) or over the whole batch.
- Grouping channels in strided order (`c % G`) instead of contiguous blocks.
- Broadcasting the `(B, C)` time vector against `(B, C, H, W)` without adding the two trailing axes, which raises or, worse, broadcasts along the wrong axes when shapes coincide.

## Interview follow-ups
- When $C_{in} \ne C_{out}$, what goes on the skip path? (A learned 1x1 convolution.)
- Why is the SiLU applied to the embedding before the projection, and why does it matter that it is smooth?
"""
