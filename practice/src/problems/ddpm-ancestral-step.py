import numpy as np

META = dict(title='One DDPM ancestral sampling step', tags=['Sampling', 'Diffusion Models'], diff='Medium', runtime='numpy',
            lessons=['ieng:26.08'], related=['posterior-mean-and-variance', 'ddim-step', 'toy-ddpm-sampler'])

STATEMENT = r"""
To generate, DDPM walks the chain backwards one step at a time. Given the network's noise estimate $\hat\varepsilon$ at step $t$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\hat\varepsilon\right), \qquad x_{t-1} = \mu_\theta + \sigma_t\,z$$

with fresh noise $z \sim \mathcal N(0, I)$ added at every step **except the last** ($t = 0$), where the mean is returned as the final sample.

Two standard choices for the variance:
- `"beta"`: $\sigma_t^2 = \beta_t$
- `"posterior"`: $\sigma_t^2 = \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$, with $\bar\alpha_{t-1} = 1$ when $t = 0$

Write `ddpm_step(x_t, eps_pred, t, betas, noise, var_type="beta")` and return `(x_prev, x0_pred)`.

## Inputs
- `x_t`, `eps_pred`, `noise`: float arrays of the same shape (a batch at one shared step).
- `t`: Python `int`, 0-indexed step. `betas`: `(T,)`. Use $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s\le t}\alpha_s$.

## Output
- `x_prev`: the sample at step $t - 1$ (the final sample when `t == 0`; `noise` is ignored then).
- `x0_pred`: the implied clean estimate $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon)/\sqrt{\bar\alpha_t}$, often shown as a preview while sampling.
"""

STARTER = '''import numpy as np

def ddpm_step(x_t, eps_pred, t, betas, noise, var_type="beta"):
    """Return (x_{t-1}, x0_pred). No noise is added at t == 0."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ddpm_step(x_t, eps_pred, t, betas, noise, var_type="beta"):
    x_t, eps, z, b = (np.asarray(a, dtype=float) for a in (x_t, eps_pred, noise, betas))
    ab = np.cumprod(1.0 - b)
    beta, alpha, abar = b[t], 1.0 - b[t], ab[t]
    abar_prev = ab[t - 1] if t > 0 else 1.0
    mean = (x_t - beta / np.sqrt(1.0 - abar) * eps) / np.sqrt(alpha)
    x0 = (x_t - np.sqrt(1.0 - abar) * eps) / np.sqrt(abar)
    if t == 0:
        return mean, x0                                  # last step: return the mean, no noise
    var = beta if var_type == 'beta' else beta * (1.0 - abar_prev) / (1.0 - abar)
    return mean + np.sqrt(var) * z, x0
'''

HINTS = [
    'Compute $\\alpha_t$, $\\bar\\alpha_t$ and $\\bar\\alpha_{t-1}$ as scalars from `betas` first. Careful: `ab[t - 1]` at `t == 0` is `ab[-1]`, the last entry, not 1.',
    'The mean removes a scaled copy of the predicted noise, then divides by $\\sqrt{\\alpha_t}$ (one step\'s signal shrink), not by $\\sqrt{\\bar\\alpha_t}$.',
    'Return early at `t == 0` with the mean. Otherwise add `sqrt(var) * noise`, with `var` either `beta_t` or the posterior $\\tilde\\beta_t$.',
]

_r = np.random.default_rng(43)
_lin = np.linspace(1e-4, 0.02, 1000)
_x, _e, _z = _r.normal(size=(3, 4)), _r.normal(size=(3, 4)), _r.normal(size=(3, 4))
TESTS = [
    (np.array([1.0]), np.array([0.0]), 1, np.array([0.19, 0.36]), np.array([1.0])),
    (np.array([1.0]), np.array([1.0]), 0, np.array([0.19, 0.36]), np.array([5.0])),
    (_x, _e, 999, _lin, _z),
    (_x, _e, 999, _lin, _z, 'posterior'),
    (_x, _e, 500, _lin, _z, 'posterior'),
    (_x, _e, 0, _lin, _z, 'posterior'),
    (_r.normal(size=(2, 1, 3, 3)), _r.normal(size=(2, 1, 3, 3)), 3, np.array([0.1, 0.2, 0.3, 0.4]), _r.normal(size=(2, 1, 3, 3))),
]

EXPLANATION = r"""
## Why the mean looks like that
The true reverse step given $x_0$ is the posterior $\tilde\mu_t(x_t, x_0)$. Substituting the network's guess $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\hat\varepsilon)/\sqrt{\bar\alpha_t}$ into $\tilde\mu_t$ and simplifying gives the formula above, with a single $\hat\varepsilon$ term.

## Choosing the variance
$\beta_t$ and $\tilde\beta_t$ are the upper and lower bounds on the optimal reverse variance (for data that is pure noise versus a single point). They give almost identical samples at $T = 1000$ but differ noticeably with few steps. At $t = T-1$ (test 3 vs 4) they are nearly equal; at small $t$, $\tilde\beta_t$ is much smaller.

## Why no noise at the end
$x_{-1}$ would be noise added on top of the finished sample. The last step outputs the mean so the result is the model's best estimate, not a noisy one.

## Complexity
$O(n)$ per step in the sample size (plus $O(T)$ for the cumulative product, which a real sampler precomputes once).

## Common bugs
- `ab[t - 1]` at `t = 0` wraps to the last entry.
- Dividing by $\sqrt{\bar\alpha_t}$ in the mean instead of $\sqrt{\alpha_t}$.
- Adding noise at $t = 0$, which leaves visible grain in images.

## Interview follow-ups
- Why does DDPM need around 1000 steps while DDIM can use 50?
- How would you clip $\hat x_0$ to $[-1, 1]$ and still produce a consistent mean? (Recompute the mean from the posterior with the clipped $\hat x_0$.)
"""
