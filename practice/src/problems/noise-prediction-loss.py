import numpy as np

META = dict(title='Noise-prediction loss for a toy denoiser', tags=['Diffusion Models', 'Loss Functions'], diff='Easy',
            runtime='numpy', lessons=['ieng:26.08'],
            related=['forward-diffusion-sample', 'toy-diffusion-training-step', 'min-snr-loss-weighting'])

STATEMENT = r"""
The standard DDPM objective is plain regression: noise a clean example to a random step, ask the network which noise was added, and score it with mean squared error.

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad \mathcal L = \operatorname{mean}\big(\lVert \hat\varepsilon_\theta(x_t, t) - \varepsilon \rVert^2\big)$$

The "network" here is a toy linear model with a learned per-step bias, so you can focus on the objective:

$$\hat\varepsilon_\theta(x_t, t) = x_t\,W + E_t$$

where $E_t$ is row $t$ of a bias table $E$.

Write `eps_loss(x0, t, noise, alpha_bars, W, E)`.

## Inputs
- `x0`: `(B, D)` clean data; `noise`: `(B, D)` the $\varepsilon$ to add; `t`: `(B,)` integer steps (0-indexed).
- `alpha_bars`: `(T,)`; `W`: `(D, D)`, applied as `x_t @ W`; `E`: `(T, D)`.

## Output
A Python `float`: the mean of the squared error over **all** $B \times D$ entries.
"""

STARTER = '''import numpy as np

def eps_loss(x0, t, noise, alpha_bars, W, E):
    """Noise x0 to step t, predict eps = x_t @ W + E[t], return the mean squared error (float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def eps_loss(x0, t, noise, alpha_bars, W, E):
    x0, noise, W, E = (np.asarray(a, dtype=float) for a in (x0, noise, W, E))
    ab = np.asarray(alpha_bars, dtype=float)[t][:, None]     # (B, 1): one noise level per row
    x_t = np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * noise
    pred = x_t @ W + E[t]                                    # E[t] gathers one bias row per example
    return float(np.mean((pred - noise) ** 2))
'''

HINTS = [
    'Two stages: build `x_t` with the forward-diffusion formula, then run the model on it. The target is the **noise**, not `x0`.',
    '`alpha_bars[t][:, None]` gives a `(B, 1)` column so each row gets its own noise level. `E[t]` gives a `(B, D)` array of bias rows.',
    '`float(np.mean((x_t @ W + E[t] - noise) ** 2))`.',
]

_r = np.random.default_rng(13)
_ab = np.cumprod(1 - np.linspace(1e-4, 0.02, 100))
TESTS = [
    (np.array([[1.0, 0.0]]), np.array([0]), np.array([[0.0, 1.0]]), np.array([0.25]), np.zeros((2, 2)), np.zeros((1, 2))),
    (np.array([[2.0]]), np.array([1]), np.array([[1.0]]), np.array([0.5, 0.0]), np.eye(1), np.array([[0.0], [0.0]])),
    (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([0, 1]), np.array([[1.0, -1.0], [2.0, 0.0]]), np.array([0.0, 0.0]), np.eye(2), np.zeros((2, 2))),
    (_r.normal(size=(5, 3)), np.array([0, 50, 99, 50, 10]), _r.normal(size=(5, 3)), _ab, _r.normal(size=(3, 3)) * 0.3, _r.normal(size=(100, 3)) * 0.1),
    (_r.normal(size=(8, 4)), _r.integers(0, 100, size=8), _r.normal(size=(8, 4)), _ab, np.eye(4), np.zeros((100, 4))),
    (_r.normal(size=(3, 2)), np.array([2, 2, 2]), _r.normal(size=(3, 2)), _ab, np.zeros((2, 2)), _r.normal(size=(100, 2))),
]

EXPLANATION = r"""
## Why predicting noise is enough
The DDPM variational bound breaks into per-step KL terms between Gaussians. If the reverse mean is written as $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\varepsilon_\theta\big)$, each KL becomes a weighted squared error on $\hat\varepsilon$. Dropping the weights gives this "simple" loss, which trains better in practice because it puts relatively more weight on the hard, noisy steps.

Test 2 is the sanity check: with $\bar\alpha = 0$, $x_t = \varepsilon$ exactly, and an identity model is perfect (loss $0$). Test 5 runs the same identity model at mixed noise levels, where it is wrong by $\sqrt{\bar\alpha_t}x_0 + (\sqrt{1-\bar\alpha_t} - 1)\varepsilon$.

## Complexity
$O(BD^2)$ for the matrix product; the rest is $O(BD)$.

## Common bugs
- Regressing onto `x0` or `x_t` instead of the noise.
- Using one $t$ for the whole batch, or broadcasting `alpha_bars[t]` of shape `(B,)` against `(B, D)` without a trailing axis.
- Summing over $D$ and averaging over $B$: a different scale from the mean over all entries (it changes the effective learning rate).

## Interview follow-ups
- Why does the toy model need the bias table $E$ at all? What would a model with no access to $t$ learn?
- What changes if you regress on $x_0$ or $v$ instead? (See the parameterization and Min-SNR problems.)
"""
