import numpy as np

META = dict(title='DDIM step with eta and x0 clipping', tags=['Sampling', 'Diffusion Models', 'ODE Solvers'], diff='Hard',
            runtime='numpy', lessons=['ieng:26.09', 'ieng:26.08'],
            related=['ddpm-ancestral-step', 'prediction-parameterization', 'classifier-free-guidance'])

STATEMENT = r"""
DDIM reuses a trained DDPM but can **skip steps**: it jumps from step $t$ straight to any earlier step $t'$. It first estimates the clean sample, then re-noises it to the level of $t'$ along the predicted noise direction:

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon}{\sqrt{\bar\alpha_t}}$$

$$\sigma = \eta\,\sqrt{\frac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}}\,\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t'}}}$$

$$x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t'}-\sigma^2}\;\hat\varepsilon + \sigma\,z$$

$\eta = 0$ is the deterministic DDIM sampler; $\eta = 1$ with consecutive steps recovers DDPM's posterior-variance sampler.

Write `ddim_step(x_t, eps_pred, t, t_prev, alpha_bars, eta, noise, clip_x0=False)` and return `(x_prev, x0_pred)`.

## Inputs
- `x_t`, `eps_pred`, `noise`: float arrays of one shape (a batch at one shared step).
- `t`: int step index into `alpha_bars` `(T,)`. `t_prev`: int with `-1 <= t_prev < t`; `t_prev == -1` means "jump to clean data", where $\bar\alpha_{t'} = 1$.
- `eta`: float $\ge 0$.

## Clipping
If `clip_x0` is true, clip $\hat x_0$ to $[-1, 1]$ **and then recompute** $\hat\varepsilon$ to be consistent with the clipped value, $\hat\varepsilon = (x_t - \sqrt{\bar\alpha_t}\,\hat x_0)/\sqrt{1-\bar\alpha_t}$, before using it in the direction term. `x0_pred` is the clipped value.

## Numerics
Take $\max(0,\ 1-\bar\alpha_{t'}-\sigma^2)$ under the square root, since rounding can make it a hair negative.
"""

STARTER = '''import numpy as np

def ddim_step(x_t, eps_pred, t, t_prev, alpha_bars, eta, noise, clip_x0=False):
    """Jump from step t to t_prev (-1 = clean). Return (x_prev, x0_pred)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ddim_step(x_t, eps_pred, t, t_prev, alpha_bars, eta, noise, clip_x0=False):
    x_t, eps, z = (np.asarray(a, dtype=float) for a in (x_t, eps_pred, noise))
    ab = np.asarray(alpha_bars, dtype=float)
    a_t = ab[t]
    a_p = ab[t_prev] if t_prev >= 0 else 1.0           # never ab[-1] for "clean"
    x0 = (x_t - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)
    if clip_x0:
        x0 = np.clip(x0, -1.0, 1.0)
        eps = (x_t - np.sqrt(a_t) * x0) / np.sqrt(1.0 - a_t)   # keep (x0, eps) consistent with x_t
    sigma = eta * np.sqrt((1.0 - a_p) / (1.0 - a_t)) * np.sqrt(1.0 - a_t / a_p)
    direction = np.sqrt(max(0.0, 1.0 - a_p - sigma ** 2)) * eps
    return np.sqrt(a_p) * x0 + direction + sigma * z, x0
'''

HINTS = [
    'Look up $\\bar\\alpha_t$ and $\\bar\\alpha_{t\'}$ first, mapping `t_prev == -1` to `1.0` explicitly (indexing `-1` would give the noisiest step).',
    'Write $x_{t\'}$ as three pieces: the clean estimate rescaled to the new level, the predicted noise rescaled to the remaining deterministic noise budget $1 - \\bar\\alpha_{t\'} - \\sigma^2$, and fresh noise $\\sigma z$.',
    'With clipping, the direction term must use the noise implied by the **clipped** $\\hat x_0$: `eps = (x_t - sqrt(a_t) * x0) / sqrt(1 - a_t)`. Otherwise the pair $(\\hat x_0, \\hat\\varepsilon)$ no longer reconstructs $x_t$.',
]

_r = np.random.default_rng(53)
_ab = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))
_x, _e, _z = _r.normal(size=(2, 6)), _r.normal(size=(2, 6)), _r.normal(size=(2, 6))
TESTS = [
    (np.array([1.0]), np.array([1.0]), 1, 0, np.array([0.75, 0.5]), 0.0, np.array([0.0])),
    (np.array([0.5, -0.5]), np.array([1.0, 1.0]), 0, -1, np.array([0.25]), 0.0, np.array([0.0, 0.0])),
    (_x, _e, 999, 979, _ab, 0.0, _z),
    (_x, _e, 999, 979, _ab, 1.0, _z),
    (_x, _e, 500, 250, _ab, 0.5, _z),
    (_x * 3, _e, 999, 949, _ab, 0.0, _z, True),
    (_x * 3, _e, 300, 100, _ab, 1.0, _z, True),
    (_x, _e, 20, -1, _ab, 1.0, _z),
]

EXPLANATION = r"""
## Why it works
DDIM defines a family of non-Markovian forward processes that all have the **same marginals** $q(x_t\mid x_0)$ as DDPM. A model trained with the noise-prediction loss only ever sees those marginals, so it is valid for every member of the family. Choosing $\sigma = 0$ makes the reverse process deterministic: the same noise $\hat\varepsilon$ that explains $x_t$ is reused at level $t'$. Because nothing in the update requires $t' = t - 1$, you can sample with 50 or even 10 evaluations.

The deterministic case is an Euler step of the probability-flow ODE in the variables $x/\sqrt{\bar\alpha}$ and $\sqrt{(1-\bar\alpha)/\bar\alpha}$, which is why it shares the ODE Solvers tag.

## The clipping subtlety
Clipping $\hat x_0$ but keeping the original $\hat\varepsilon$ produces a pair that no longer reconstructs $x_t$: the deterministic update then drifts. Recomputing $\hat\varepsilon$ from the clipped $\hat x_0$ keeps $x_t = \sqrt{\bar\alpha_t}\hat x_0 + \sqrt{1-\bar\alpha_t}\hat\varepsilon$ exact. Tests 6 and 7 scale `x_t` up so clipping actually triggers.

## Complexity
$O(n)$ per step; the network call dominates in practice.

## Common bugs
- `alpha_bars[t_prev]` with `t_prev = -1` silently uses the noisiest level.
- Forgetting $-\sigma^2$ in the direction coefficient when $\eta > 0$, which adds too much noise.
- Using $\beta$-based formulas: DDIM is written entirely in $\bar\alpha$ and does not need the per-step betas of the skipped steps.

## Interview follow-ups
- Why can deterministic DDIM encode a real image into a latent noise and reconstruct it? What is that used for?
- How do you choose the subsequence of timesteps ("linspace", "leading", "trailing"), and why does it matter with zero terminal SNR?
"""
