import numpy as np

META = dict(title='Forward-process posterior: mean and variance', tags=['Diffusion Models', 'Probability'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:11.25'],
            related=['diagonal-gaussian-kl-divergence', 'ddpm-ancestral-step', 'linear-beta-schedule'])

STATEMENT = r"""
If you know the clean example $x_0$, the step backwards from $x_t$ is a Gaussian with a closed form. This **posterior** is what the reverse model is trained to match:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal N\!\left(\tilde\mu_t,\ \tilde\beta_t I\right)$$
$$\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t, \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t$$

Write `q_posterior(x0, x_t, t, betas)` and return `(mean, var, log_var)`.

## Conventions
- 0-indexed: $\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s \le t}\alpha_s$, and $\bar\alpha_{t-1}$ for $t = 0$ is defined as $1$ (no noise before the first step).
- At $t = 0$ the posterior variance is exactly $0$, so its log is $-\infty$. Models need a finite log-variance, so `log_var` uses the **clipped** variance: $\log\tilde\beta_1$ at $t = 0$, and $\log\tilde\beta_t$ otherwise.

## Inputs
- `x0`, `x_t`: `(B, D)`; `t`: `(B,)` integer steps, each example its own; `betas`: `(T,)` with $T \ge 2$.

## Output
- `mean`: `(B, D)`
- `var`: `(B,)`, the unclipped $\tilde\beta_t$ (exactly `0.0` where `t == 0`)
- `log_var`: `(B,)`, the clipped log-variance
"""

STARTER = '''import numpy as np

def q_posterior(x0, x_t, t, betas):
    """Return (mean (B, D), var (B,), log_var_clipped (B,)) of q(x_{t-1} | x_t, x0)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def q_posterior(x0, x_t, t, betas):
    x0, x_t, b = (np.asarray(a, dtype=float) for a in (x0, x_t, betas))
    a = 1.0 - b
    ab = np.cumprod(a)
    ab_prev = np.concatenate([[1.0], ab[:-1]])
    var_all = b * (1.0 - ab_prev) / (1.0 - ab)                  # var_all[0] == 0
    logv_all = np.log(np.concatenate([var_all[1:2], var_all[1:]]))  # clip step 0 to step 1
    c0 = np.sqrt(ab_prev) * b / (1.0 - ab)
    ct = np.sqrt(a) * (1.0 - ab_prev) / (1.0 - ab)
    mean = c0[t][:, None] * x0 + ct[t][:, None] * x_t
    return mean, var_all[t], logv_all[t]
'''

HINTS = [
    'Precompute every coefficient for all $T$ steps as arrays, then gather them with `[t]`. `ab_prev` is `ab` shifted right by one with a `1.0` in front.',
    'Check $t = 0$ by hand: the $x_0$ coefficient becomes $\\beta_0/\\beta_0 = 1$ and the $x_t$ coefficient $0$, so the mean is exactly $x_0$ and the variance is $0$.',
    'Build the clipped variance table as `concatenate([var[1:2], var[1:]])` before taking the log, and add a trailing axis (`[:, None]`) to the coefficients before multiplying the `(B, D)` inputs.',
]

_r = np.random.default_rng(23)
_lin = np.linspace(1e-4, 0.02, 1000)
TESTS = [
    (np.array([[1.0]]), np.array([[3.0]]), np.array([1]), np.array([0.5, 0.5])),
    (np.array([[1.0, 2.0]]), np.array([[5.0, 5.0]]), np.array([0]), np.array([0.2, 0.3])),
    (np.array([[1.0], [1.0], [1.0]]), np.array([[0.0], [0.0], [0.0]]), np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3])),
    (_r.normal(size=(4, 3)), _r.normal(size=(4, 3)), np.array([0, 1, 500, 999]), _lin),
    (_r.normal(size=(6, 2)), _r.normal(size=(6, 2)), _r.integers(0, 1000, size=6), _lin),
    (_r.normal(size=(2, 5)), _r.normal(size=(2, 5)), np.array([3, 3]), np.array([0.05, 0.1, 0.2, 0.4, 0.8])),
]

EXPLANATION = r"""
## Where it comes from
By Bayes, $q(x_{t-1}\mid x_t, x_0) \propto q(x_t \mid x_{t-1})\,q(x_{t-1}\mid x_0)$. Both factors are Gaussian in $x_{t-1}$: one with precision $\alpha_t/\beta_t$ centred at $x_t/\sqrt{\alpha_t}$, the other with precision $1/(1-\bar\alpha_{t-1})$ centred at $\sqrt{\bar\alpha_{t-1}}x_0$. Multiplying Gaussians adds precisions and precision-weights the means, which gives exactly $\tilde\beta_t$ and $\tilde\mu_t$.

The two mean coefficients interpolate: early steps trust $x_t$ (little noise has been added), late steps lean on $x_0$.

## Why clip the log-variance
Models that learn the variance interpolate in log space between $\log\beta_t$ and $\log\tilde\beta_t$. At $t = 0$, $\log\tilde\beta_0 = -\infty$ would poison that interpolation, so implementations substitute the next step's value.

## Complexity
$O(T + BD)$.

## Common bugs
- Using $\bar\alpha_t$ where $\bar\alpha_{t-1}$ is needed, or forgetting to define $\bar\alpha_{-1} = 1$.
- $\sqrt{\bar\alpha_t}$ instead of $\sqrt{\alpha_t}$ in the $x_t$ coefficient.
- Taking `np.log(var)` directly and returning `-inf` at $t = 0$.

## Interview follow-ups
- Show that $\tilde\beta_t \le \beta_t$. Which choice of reverse variance gives the better bound at large $T$?
- Rewrite $\tilde\mu_t$ in terms of $x_t$ and $\varepsilon$ instead of $x_0$. (That is the DDPM sampling mean.)
"""
