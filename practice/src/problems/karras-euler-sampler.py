import numpy as np

META = dict(title='Karras sigma schedule with an Euler / Heun sampler', tags=['ODE Solvers', 'Sampling', 'Noise Schedules'],
            diff='Hard', runtime='numpy', lessons=['ieng:26.09'], related=['ddim-step', 'annealed-langevin-dynamics'])

STATEMENT = r"""
In the "noise level as time" view, a sample at noise level $\sigma$ is $x = x_0 + \sigma\varepsilon$, and generation integrates the probability-flow ODE from $\sigma_{\max}$ down to $0$:

$$\frac{dx}{d\sigma} = \frac{x - D(x, \sigma)}{\sigma}$$

where $D(x, \sigma)$ is a denoiser's estimate of $x_0$. Two ingredients decide quality per step: **where** you place the noise levels, and **how** you step between them.

## Noise levels
For $n \ge 2$ steps, place levels evenly in $\sigma^{1/\rho}$ and append a final $0$:

$$\sigma_i = \left(\sigma_{\max}^{1/\rho} + \tfrac{i}{n-1}\big(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\big)\right)^{\rho}, \quad i = 0, \dots, n-1, \qquad \sigma_n = 0$$

## Denoiser
So the problem has an exact answer, the data is Gaussian, $x_0 \sim \mathcal N(\mu, s^2 I)$, for which the optimal denoiser is

$$D(x, \sigma) = \frac{s^2\,x + \sigma^2\,\mu}{s^2 + \sigma^2}$$

## Sampler
Start at $x = \sigma_0 \cdot$ `x_init` (`x_init` is standard-normal noise you are given). For $i = 0, \dots, n-1$:
1. $d = (x - D(x, \sigma_i)) / \sigma_i$ and the Euler proposal $\tilde x = x + (\sigma_{i+1} - \sigma_i)\,d$.
2. If `heun` is true **and** $\sigma_{i+1} > 0$: $d' = (\tilde x - D(\tilde x, \sigma_{i+1}))/\sigma_{i+1}$ and $x \leftarrow x + (\sigma_{i+1}-\sigma_i)\,\tfrac{d + d'}{2}$. Otherwise $x \leftarrow \tilde x$.

Write `karras_sample(x_init, n, sigma_min, sigma_max, rho, mu, s, heun=False)` and return `(sigmas, x)`.

## Inputs and output
- `x_init`: `(N, D)`; `mu`: `(D,)`; `s`: float data std; `rho`: float (7 is typical).
- `sigmas`: `(n + 1,)` float array ending in `0.0`; `x`: the final `(N, D)` sample.
"""

STARTER = '''import numpy as np

def karras_sample(x_init, n, sigma_min, sigma_max, rho, mu, s, heun=False):
    """Return (sigmas (n+1,), final x (N, D)) from the Karras schedule and an Euler/Heun ODE sampler."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def karras_sample(x_init, n, sigma_min, sigma_max, rho, mu, s, heun=False):
    mu = np.asarray(mu, dtype=float)
    lo, hi = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    sig = (hi + np.arange(n) / (n - 1) * (lo - hi)) ** rho
    sig = np.append(sig, 0.0)

    def D(x, sg):                                     # optimal denoiser for N(mu, s^2 I) data
        return (s * s * x + sg * sg * mu) / (s * s + sg * sg)

    x = np.asarray(x_init, dtype=float) * sig[0]
    for i in range(n):
        cur, nxt = sig[i], sig[i + 1]
        d = (x - D(x, cur)) / cur
        x_new = x + (nxt - cur) * d                   # Euler
        if heun and nxt > 0:                          # trapezoid correction; skipped on the last step
            d2 = (x_new - D(x_new, nxt)) / nxt
            x_new = x + (nxt - cur) * (d + d2) / 2
        x = x_new
    return sig, x
'''

HINTS = [
    'Build the schedule with `np.arange(n) / (n - 1)` in $\\sigma^{1/\\rho}$ space, raise to the power $\\rho$, then `np.append(..., 0.0)`.',
    'The ODE slope is "noise estimate per unit sigma": $(x - D(x,\\sigma))/\\sigma$. An Euler step moves by $(\\sigma_{i+1} - \\sigma_i)$ times that slope (a negative step, since sigma decreases).',
    'Heun evaluates the slope again at the Euler proposal and averages the two slopes. Skip it on the final step, where $\\sigma_{i+1} = 0$ would divide by zero. Remember to start from `x_init * sigmas[0]`, not from `x_init`.',
]

_r = np.random.default_rng(59)
_x = _r.normal(size=(4, 2))
_mu = np.array([1.0, -2.0])
TESTS = [
    (np.array([[1.0]]), 2, 1.0, 4.0, 1.0, np.array([0.0]), 1.0),
    (np.array([[1.0]]), 2, 1.0, 4.0, 1.0, np.array([0.0]), 1.0, True),
    (_x, 10, 0.002, 80.0, 7.0, _mu, 0.5),
    (_x, 10, 0.002, 80.0, 7.0, _mu, 0.5, True),
    (_x, 3, 0.1, 10.0, 3.0, _mu, 2.0, True),
    (_r.normal(size=(5, 3)), 25, 0.01, 50.0, 7.0, np.array([0.5, 0.0, -0.5]), 1.0, True),
]

EXPLANATION = r"""
## Why the schedule
With $\rho = 1$ the levels are evenly spaced in $\sigma$; larger $\rho$ crowds them towards $\sigma_{\min}$, where the ODE curves most and errors are most visible. $\rho = 7$ was found to balance the truncation error across the range.

## Why it works
For Gaussian data the ODE has a closed form: $x(\sigma) = \mu + (x(\sigma_0) - \mu)\sqrt{\tfrac{s^2+\sigma^2}{s^2+\sigma_0^2}}$, so the exact sample is $\mu + (\sigma_{\max}x_{init} - \mu)\frac{s}{\sqrt{s^2 + \sigma_{\max}^2}}$. Euler is first order, so its error shrinks linearly with the step size; Heun is second order and pulls ahead as $n$ grows, at the cost of about twice the denoiser calls ($2n - 1$). Tests 3 and 4 run both with 10 steps; try $n = 200$ with Heun and the gap to the exact answer drops to around $10^{-4}$.

## Complexity
$O(nND)$ time, $O(ND)$ memory, with $n$ (Euler) or $2n-1$ (Heun) denoiser evaluations.

## Common bugs
- Starting from unit-variance noise instead of $\sigma_{\max}$-scaled noise.
- Applying the Heun correction on the last step and dividing by $\sigma = 0$.
- Using $\sigma_{i} - \sigma_{i+1}$ (positive) as the step, which runs the ODE backwards and adds noise.

## Interview follow-ups
- Add "churn": temporarily raise $\sigma$ by a factor and add matching fresh noise before each step. What does it trade off?
- How do you convert a DDPM model's $\bar\alpha_t$ into this $\sigma$ parameterization? ($\sigma_t = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$, input scaled by $\sqrt{\bar\alpha_t}$.)
"""
