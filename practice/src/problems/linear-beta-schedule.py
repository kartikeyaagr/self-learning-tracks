import numpy as np

META = dict(title='Linear beta schedule and its cumulative products', tags=['Noise Schedules', 'Diffusion Models'], diff='Easy',
            runtime='numpy', lessons=['ieng:26.08'], related=['cosine-noise-schedule', 'forward-diffusion-sample'])

STATEMENT = r"""
A DDPM corrupts data over $T$ steps. Step $t$ adds Gaussian noise of variance $\beta_t$ and shrinks the signal by $\sqrt{1-\beta_t}$. Everything else in the model is built from three arrays:

$$\beta_t, \qquad \alpha_t = 1 - \beta_t, \qquad \bar\alpha_t = \prod_{s \le t} \alpha_s$$

$\bar\alpha_t$ is the fraction of signal variance left after step $t$. The classic schedule spaces $\beta$ evenly between a small start and a larger end value.

Write `linear_schedule(T, beta_start=1e-4, beta_end=0.02)` and return `(betas, alphas, alpha_bars)`.

## Conventions (used throughout this study plan)
- Arrays are **0-indexed**: entry `t` for `t = 0 .. T-1`. Entry `0` is the first, least noisy step.
- `betas` are `T` evenly spaced values from `beta_start` to `beta_end` **inclusive** (so `betas[0] == beta_start` and `betas[-1] == beta_end`).
- If `T == 1`, `betas` is just `[beta_start]`.

## Output
Three float arrays of shape `(T,)`.
"""

STARTER = '''import numpy as np

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Return (betas, alphas, alpha_bars), each of shape (T,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T)     # inclusive of both ends; T == 1 gives [beta_start]
    alphas = 1.0 - betas
    return betas, alphas, np.cumprod(alphas)
'''

HINTS = [
    '`np.linspace(a, b, n)` returns `n` evenly spaced values including both endpoints.',
    '$\\bar\\alpha_t$ is a running product, not a single product: each entry multiplies all the $\\alpha$ values up to and including $t$.',
    '`np.cumprod(1 - betas)`.',
]

TESTS = [
    (3, 0.1, 0.3),
    (2, 0.5, 0.5),
    (1, 0.01, 0.9),
    (5,),
    (1000,),
    (10, 0.0, 1.0),
]

EXPLANATION = r"""
## Why the cumulative product matters
Composing Gaussian steps gives a Gaussian: after $t$ steps, $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$. So $\bar\alpha_t$ lets you jump straight to any noise level without simulating the chain. With the defaults and $T = 1000$, $\bar\alpha_{T-1} \approx 4\times10^{-5}$: the last step is almost pure noise.

## Complexity
$O(T)$.

## Common bugs
- `np.arange(beta_start, beta_end, step)` drops the end point and is sensitive to float rounding in the step.
- Taking `np.prod` (a scalar) instead of `np.cumprod`.
- Off-by-one between 0-indexed arrays and the 1-indexed $t$ in papers. Pick one convention and keep it everywhere.

## Interview follow-ups
- Why does the linear schedule destroy information too quickly at low resolutions? (Most of $\bar\alpha$ is near zero for the last half of the steps.)
- If you change $T$, how should you rescale `beta_start` and `beta_end` to keep the same continuous-time process? (Multiply by $1000/T$.)
"""
