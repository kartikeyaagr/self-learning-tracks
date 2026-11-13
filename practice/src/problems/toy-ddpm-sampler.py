import numpy as np

META = dict(title='Full DDPM sampling loop with a toy denoiser', tags=['Sampling', 'Diffusion Models'], diff='Hard', runtime='numpy',
            lessons=['ieng:26.08'], related=['ddpm-ancestral-step', 'posterior-mean-and-variance', 'toy-diffusion-training-step'])

STATEMENT = r"""
Generate samples end to end: start from pure noise at the last step and walk the chain back to $t = 0$, calling the toy denoiser $\hat\varepsilon = x\,W + E_t$ at every step.

For $t = T-1, T-2, \dots, 0$ (0-indexed, with $\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s \le t}\alpha_s$ and $\bar\alpha_{-1} = 1$):

1. $\hat\varepsilon = x\,W + E_t$.
2. $\hat x_0 = (x - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon)/\sqrt{\bar\alpha_t}$. If `clip` is true, clip $\hat x_0$ to $[-1, 1]$.
3. Posterior mean from $\hat x_0$:
$$\mu = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\hat x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x$$
4. If $t > 0$: $x \leftarrow \mu + \sqrt{\tilde\beta_t}\,z_t$ with $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$. If $t = 0$: $x \leftarrow \mu$.

Without clipping, step 3 equals the familiar $\frac{1}{\sqrt{\alpha_t}}\big(x - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\varepsilon\big)$; going through $\hat x_0$ is what makes clipping possible.

Write `ddpm_sample(x_T, z, betas, W, E, clip=False)` and return the final samples.

## Inputs
- `x_T`: `(B, D)` starting noise.
- `z`: `(T, B, D)` noise for the updates; `z[t]` is used at step $t$ (so `z[0]` is never used).
- `betas`: `(T,)`; `W`: `(D, D)`, applied as `x @ W`; `E`: `(T, D)`; `clip`: bool.

## Output
`(B, D)` float array.
"""

STARTER = '''import numpy as np

def ddpm_sample(x_T, z, betas, W, E, clip=False):
    """Run the reverse chain from t = T-1 down to 0 with eps = x @ W + E[t]. Return the final (B, D) samples."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ddpm_sample(x_T, z, betas, W, E, clip=False):
    b = np.asarray(betas, dtype=float)
    z, W, E = (np.asarray(a, dtype=float) for a in (z, W, E))
    a = 1.0 - b
    ab = np.cumprod(a)
    ab_prev = np.concatenate([[1.0], ab[:-1]])
    x = np.asarray(x_T, dtype=float)
    for t in range(len(b) - 1, -1, -1):
        eps = x @ W + E[t]
        x0 = (x - np.sqrt(1.0 - ab[t]) * eps) / np.sqrt(ab[t])
        if clip:
            x0 = np.clip(x0, -1.0, 1.0)
        mean = (np.sqrt(ab_prev[t]) * b[t] * x0 + np.sqrt(a[t]) * (1.0 - ab_prev[t]) * x) / (1.0 - ab[t])
        if t > 0:
            var = b[t] * (1.0 - ab_prev[t]) / (1.0 - ab[t])
            x = mean + np.sqrt(var) * z[t]
        else:
            x = mean                                    # final step: no noise
    return x
'''

HINTS = [
    'Precompute `alphas`, `alpha_bars` and `alpha_bars_prev` (shifted right with a leading `1.0`) once, outside the loop.',
    'Loop with `range(T - 1, -1, -1)`. Inside: predict noise, form $\\hat x_0$, optionally clip, then the posterior mean. The model always sees the **current** `x`.',
    'Add `sqrt(posterior_var) * z[t]` only when `t > 0`. At `t == 0` the posterior mean is exactly $\\hat x_0$ (its $x$ coefficient is 0), which is why clipping also bounds the final output.',
]

_r = np.random.default_rng(89)
_b8 = np.linspace(0.05, 0.4, 8)
_x, _z = _r.normal(size=(4, 2)), _r.normal(size=(8, 4, 2))
_W, _E = _r.normal(size=(2, 2)) * 0.3, _r.normal(size=(8, 2)) * 0.3
TESTS = [
    (np.array([[1.0]]), np.zeros((1, 1, 1)), np.array([0.5]), np.zeros((1, 1)), np.zeros((1, 1))),
    (np.array([[1.0]]), np.ones((2, 1, 1)), np.array([0.5, 0.5]), np.eye(1), np.zeros((2, 1))),
    (_x, _z, _b8, _W, _E),
    (_x, _z, _b8, _W, _E, True),
    (_x * 2, _z, _b8, np.eye(2), np.zeros((8, 2)), True),
    (_r.normal(size=(3, 3)), _r.normal(size=(20, 3, 3)), np.linspace(0.01, 0.2, 20), _r.normal(size=(3, 3)) * 0.2, _r.normal(size=(20, 3)) * 0.2),
]

EXPLANATION = r"""
## Why it works
Each iteration samples $x_{t-1} \sim q(x_{t-1}\mid x_t, \hat x_0)$, the exact backward step **if** the model's guess of $x_0$ were right. Early on $\hat x_0$ is a blurry average; as noise is removed the guess sharpens, and the chain ends at $t = 0$, where the posterior collapses onto $\hat x_0$ itself.

Test 1 is a single step from pure noise with an all-zero model: $\hat\varepsilon = 0$ means $\hat x_0 = x/\sqrt{\bar\alpha_0}$. Test 5 uses $W = I$: the model claims everything is noise, so $\hat x_0 = x(1 - \sqrt{1-\bar\alpha})/\sqrt{\bar\alpha}$, clipped.

## Clipping
Images live in $[-1, 1]$. Clipping $\hat x_0$ (not $x_t$) keeps the chain on plausible images and avoids over-saturated colours, at the price of a slight bias. It only works in the $\hat x_0$ form of the mean, which is why the loop goes through step 2.

## Complexity
$O(T\,(BD^2))$: one model evaluation per step, $T$ sequential steps. That sequential depth is the cost DDIM and distillation attack.

## Common bugs
- Looping forwards in time, or off by one (starting at $T$ or stopping at $1$).
- Adding noise at $t = 0$, or using `z[t-1]` instead of `z[t]`.
- Clipping $x_t$ instead of $\hat x_0$.

## Interview follow-ups
- Swap this loop for DDIM with 4 of the 8 steps. Which parts change?
- How would you plug classifier-free guidance into step 1?
"""
