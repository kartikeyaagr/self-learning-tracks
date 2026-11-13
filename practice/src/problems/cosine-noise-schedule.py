import numpy as np

META = dict(title='Cosine noise schedule with beta clipping', tags=['Noise Schedules', 'Diffusion Models'], diff='Medium',
            runtime='numpy', lessons=['ieng:26.09'], related=['linear-beta-schedule', 'rescale-to-zero-terminal-snr'])

STATEMENT = r"""
The linear schedule wastes many steps near pure noise. The **cosine schedule** defines the signal level $\bar\alpha$ directly, so that it falls slowly at both ends:

$$f(u) = \cos^2\!\left(\frac{u/T + s}{1 + s}\cdot\frac{\pi}{2}\right), \qquad \tilde\alpha_u = \frac{f(u)}{f(0)}, \quad u = 0, 1, \dots, T$$

The small offset $s$ stops $\beta$ from being too tiny near $u = 0$. Per-step betas come from consecutive ratios and are then **clipped** so no step destroys the signal completely:

$$\beta_i = \min\!\left(1 - \frac{\tilde\alpha_{i+1}}{\tilde\alpha_i},\ \beta_{\max}\right), \qquad i = 0, \dots, T-1$$

Write `cosine_schedule(T, s=0.008, max_beta=0.999)` and return `(betas, alpha_bars)`.

## Output
- `betas`: `(T,)`, computed as above (0-indexed, entry `i` is the step from level $i$ to level $i+1$).
- `alpha_bars`: `(T,)`, **recomputed** from the clipped betas as `cumprod(1 - betas)`. Do not return $\tilde\alpha$ itself: after clipping the two differ, and the sampler must use values consistent with its betas.

Floating point makes $f(T)$ a tiny positive number rather than exactly $0$, so the last raw beta is just below $1$. Clipping handles it either way.
"""

STARTER = '''import numpy as np

def cosine_schedule(T, s=0.008, max_beta=0.999):
    """Return (betas (T,), alpha_bars (T,)) for the cosine schedule."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def cosine_schedule(T, s=0.008, max_beta=0.999):
    u = np.arange(T + 1, dtype=float)
    f = np.cos((u / T + s) / (1 + s) * np.pi / 2) ** 2
    abar = f / f[0]                                   # T + 1 levels, abar[0] == 1
    betas = np.minimum(1.0 - abar[1:] / abar[:-1], max_beta)
    return betas, np.cumprod(1.0 - betas)             # consistent with the clipped betas
'''

HINTS = [
    'Evaluate $f$ on all $T+1$ integer levels $0..T$ at once with `np.arange(T + 1)`, then divide by `f[0]`.',
    'Betas are one minus the ratio of consecutive levels: `1 - abar[1:] / abar[:-1]` has exactly `T` entries.',
    'Clip with `np.minimum(..., max_beta)`, then rebuild `alpha_bars = np.cumprod(1 - betas)` rather than slicing the unclipped levels.',
]

TESTS = [
    (2, 0.0, 0.999),
    (4,),
    (1,),
    (10, 0.008, 0.5),
    (1000,),
    (50, 0.1, 0.9),
]

EXPLANATION = r"""
## Why the shape helps
Near $u = 0$ the cosine is flat, so early steps add very little noise and fine detail survives longer. Near $u = T$ it is also flat in $\bar\alpha$, which avoids spending dozens of steps on samples that are already pure noise. The result is that $\log$ SNR falls roughly linearly in the middle of the range, which spreads the training signal more evenly across noise levels.

## Why clip, and why recompute
At $u = T$ the raw ratio is almost $0$, so $\beta_{T-1} \approx 1$. A beta of $1$ means $\sqrt{\alpha_t} = 0$, and the DDPM update divides by $\sqrt{\alpha_t}$. Clipping at $0.999$ keeps that finite. Once you clip, $\prod(1-\beta)$ no longer equals $\tilde\alpha$, so the cumulative product has to be rebuilt from the betas the sampler will actually use.

## Complexity
$O(T)$.

## Common bugs
- Using $T$ levels instead of $T+1$, which leaves you one beta short.
- Forgetting to divide by $f(0)$, so $\bar\alpha$ starts below $1$.
- Returning the unclipped $\tilde\alpha_{1:T}$ (test 4 has a tight `max_beta` that exposes it).

## Interview follow-ups
- Express the cosine schedule in terms of log-SNR. What does it look like as a function of $u/T$?
- Why do high-resolution images need the schedule shifted towards more noise?
"""
