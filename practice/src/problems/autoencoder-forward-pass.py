import numpy as np

META = dict(title='Autoencoder with optional tied weights', tags=['Generative Models', 'Neural Networks', 'Loss Functions'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:05.03'],
            related=['variational-autoencoder-forward-pass', 'multi-layer-perceptron-forward-pass', 'log-loss-binary-cross-entropy'])

STATEMENT = r"""
An autoencoder squeezes inputs through a narrow code $z$ and tries to reconstruct them. For inputs in $[0, 1]$ (pixel intensities) the decoder outputs logits and the reconstruction loss is binary cross-entropy. A common trick is **tied weights**: each decoder layer reuses the transpose of the mirrored encoder layer, halving the parameter count.

Write `autoencoder(X, enc, dec)` that returns a dict with keys `'z'`, `'recon'` and `'loss'`.

## Network
- `X`: float `(N, D)` with entries in $[0, 1]$.
- `enc`: list of $L \ge 1$ pairs `(W, b)`, `W` of shape `(in, out)`, applied as `a @ W + b`. Every encoder layer except the last is followed by ReLU; the **last** is linear and its output is the code `z`.
- `dec`: list of pairs `(W, b)`, same convention. If `W` is `None` in position $k$ (0-based), that layer is tied: it uses `enc[L - 1 - k][0].T` (its bias `b` is always its own). Every decoder layer except the last is followed by ReLU; the last produces logits $s$ of shape `(N, D)`.

## Output
- `'z'`: `(N, code)`. `'recon'`: $\sigma(s)$, `(N, D)`.
- `'loss'`: a Python float, the BCE **summed over the $D$ features and averaged over the $N$ samples**:
$$\frac1N \sum_{n}\sum_{j} \Big[-X_{nj}\log\sigma(s_{nj}) - (1 - X_{nj})\log(1 - \sigma(s_{nj}))\Big].$$
Compute it from the logits so that saturated outputs don't produce `inf` or `nan`.
"""

STARTER = '''import numpy as np

def autoencoder(X, enc, dec):
    """enc, dec: lists of (W, b); a dec W of None means tied. Return {'z', 'recon', 'loss'}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def autoencoder(X, enc, dec):
    X = np.asarray(X, dtype=float)
    a = X
    for k, (W, b) in enumerate(enc):
        a = a @ np.asarray(W, dtype=float) + b
        if k < len(enc) - 1:
            a = np.maximum(a, 0.0)
    z = a
    for k, (W, b) in enumerate(dec):
        W = np.asarray(enc[len(enc) - 1 - k][0], dtype=float).T if W is None else np.asarray(W, dtype=float)
        a = a @ W + b
        if k < len(dec) - 1:
            a = np.maximum(a, 0.0)
    s = a                                                      # logits
    e = np.exp(-np.abs(s))
    recon = np.where(s >= 0, 1 / (1 + e), e / (1 + e))
    # -x log sig(s) - (1-x) log(1-sig(s)) = softplus(s) - x s, with softplus(s) = max(s,0) + log1p(e^-|s|)
    bce = np.maximum(s, 0.0) + np.log1p(e) - X * s
    return {'z': z, 'recon': recon, 'loss': float(bce.sum(axis=1).mean())}
'''

HINTS = [
    'Run two loops of `a = a @ W + b` with ReLU after every layer but the last of each stack. The encoder\'s final output is `z`; the decoder\'s is the logit matrix.',
    'For a tied layer `k`, the weight is `enc[len(enc) - 1 - k][0].T`: the first decoder layer mirrors the last encoder layer, so its shape `(code, hidden)` lines up automatically.',
    'Stable BCE from logits: $\\ell = \\max(s, 0) + \\log(1 + e^{-|s|}) - x s$. Sum over features (axis 1), then average over samples.',
]

_r = np.random.default_rng(107)
def _layers(sizes, s=0.5):
    return [(_r.normal(size=(i, o)) * s, _r.normal(size=o) * 0.1) for i, o in zip(sizes, sizes[1:])]
_X = _r.random((5, 6))
_enc = _layers([6, 4, 2])
TESTS = [
    (np.array([[1.0, 0.0]]), [(np.array([[1.0], [-1.0]]), np.zeros(1))], [(np.array([[2.0, -2.0]]), np.zeros(2))]),
    (np.array([[1.0, 0.0]]), [(np.array([[1.0], [-1.0]]), np.zeros(1))], [(None, np.array([0.0, 0.5]))]),
    (_X, _enc, _layers([2, 4, 6])),
    (_X, _enc, [(None, _r.normal(size=4) * 0.1), (None, _r.normal(size=6) * 0.1)]),
    (_X, _enc, [(None, np.zeros(4)), (_r.normal(size=(4, 6)), np.zeros(6))]),
    ((_r.random((3, 4)) > 0.5).astype(float), _layers([4, 3], s=60.0), [(None, np.zeros(4))]),
    (_r.random((4, 8)), _layers([8, 6, 5, 3]), _layers([3, 5, 6, 8])),
]

EXPLANATION = r"""
## Why BCE for pixels
Treating each pixel as a Bernoulli probability makes $-\log p(x \mid z)$ the binary cross-entropy, which is also well defined for grey values in $[0, 1]$ (it is then minimised at $\sigma(s) = x$, not at 0). Summing over features rather than averaging matters once you add other terms: the VAE's KL is per sample, and averaging the reconstruction over $D$ pixels would shrink it by $D$.

## Stable from logits
$-x\log\sigma(s) - (1-x)\log(1 - \sigma(s)) = \operatorname{softplus}(s) - xs$. Computing $\log$ of a sigmoid that rounded to exactly 0 or 1 gives `-inf` (test 6 has logits around $\pm 100$).

## Tied weights
With $W_\text{dec} = W_\text{enc}^\top$ the model has half the weights, and a linear tied autoencoder trained with squared error recovers the PCA subspace. Ties also act as a regulariser: the decoder can't undo an encoder that blows up some direction.

## Complexity
One forward pass of both stacks: $O(N \sum_k n_{k-1} n_k)$.

## Common bugs
- A ReLU on the code layer, which forces $z \ge 0$ and wastes half the code space.
- Tying to `enc[k]` instead of the mirrored `enc[L-1-k]`.
- Averaging the BCE over all $N D$ entries.

## Interview follow-ups
- What stops an overcomplete autoencoder (code larger than input) from learning the identity? (Sparsity penalties, denoising, contractive penalties.)
- Why can't you sample new data from a plain autoencoder the way you can from a VAE?
"""
