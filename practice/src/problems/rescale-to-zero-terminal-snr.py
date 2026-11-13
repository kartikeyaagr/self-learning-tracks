import numpy as np

META = dict(title='Rescale a schedule to zero terminal SNR', tags=['Noise Schedules', 'Diffusion Models'], diff='Medium',
            runtime='numpy', lessons=['ieng:26.09'], related=['linear-beta-schedule', 'cosine-noise-schedule', 'prediction-parameterization'])

STATEMENT = r"""
Common schedules never reach pure noise: with the default linear schedule, $\bar\alpha_{T-1} \approx 4\times10^{-5}$, so $\sqrt{\bar\alpha_{T-1}} \approx 0.006$ of the image survives. The model is trained on that leaked signal, but at sampling time it is started from pure noise, a mismatch that shows up as images that can't be very bright or very dark.

The fix rescales the **signal scale** $a_t = \sqrt{\bar\alpha_t}$ with an affine map that keeps the first value and sends the last to exactly zero:

$$a'_t = (a_t - a_{T-1})\,\frac{a_0}{a_0 - a_{T-1}}$$

then converts back: $\bar\alpha'_t = (a'_t)^2$, $\alpha'_0 = \bar\alpha'_0$, $\alpha'_t = \bar\alpha'_t / \bar\alpha'_{t-1}$ for $t \ge 1$, and $\beta'_t = 1 - \alpha'_t$.

Write `rescale_zero_terminal_snr(betas)` and return the new betas.

## Inputs
`betas`: `(T,)` floats in $(0, 1)$, with $T \ge 2$ and a strictly decreasing $\bar\alpha$ (so $a_0 > a_{T-1}$).

## Output
`(T,)` float array. By construction `result[0] == betas[0]` (up to rounding) and `result[-1] == 1.0`.
"""

STARTER = '''import numpy as np

def rescale_zero_terminal_snr(betas):
    """Return betas whose final cumulative alpha_bar is exactly 0 and whose first beta is unchanged."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def rescale_zero_terminal_snr(betas):
    a = np.sqrt(np.cumprod(1.0 - np.asarray(betas, dtype=float)))   # signal scale per step
    a0, aT = a[0], a[-1]
    a = (a - aT) * a0 / (a0 - aT)            # shift last to 0, then rescale so the first is unchanged
    abar = a ** 2
    alphas = np.concatenate([abar[:1], abar[1:] / abar[:-1]])        # undo the cumulative product
    return 1.0 - alphas
'''

HINTS = [
    'Work in $\\sqrt{\\bar\\alpha}$ space: compute `a = sqrt(cumprod(1 - betas))`, rescale it, square it.',
    'The map is a shift by $-a_{T-1}$ followed by a scale $a_0/(a_0 - a_{T-1})$. Check: $t = 0$ maps to $a_0$, $t = T-1$ maps to $0$.',
    'Undo the cumulative product with ratios of neighbours, `abar[1:] / abar[:-1]`, and put `abar[0]` in front. The final beta comes out as exactly `1.0`.',
]

TESTS = [
    (np.array([0.5, 0.5]),),
    (np.array([0.1, 0.2, 0.3]),),
    (np.linspace(1e-4, 0.02, 10),),
    (np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2,),
    (np.linspace(1e-4, 0.02, 1000),),
    (np.array([0.01, 0.99]),),
]

EXPLANATION = r"""
## Why it works
The affine map is monotone, so the rescaled $a'$ is still decreasing and every intermediate $\alpha'_t \in (0, 1)$. Fixing $a'_0 = a_0$ keeps the least-noisy step unchanged, while $a'_{T-1} = 0$ means $\bar\alpha'_{T-1} = 0$: SNR $\bar\alpha/(1-\bar\alpha)$ is exactly zero at the final step, so training sees the same pure noise the sampler starts from. Test 4 is the "scaled linear" schedule used by latent diffusion models, where the leak is large ($a_{T-1} \approx 0.068$).

## The catch
With $\bar\alpha'_{T-1} = 0$, the noise prediction carries no information about $x_0$ at the last step ($x_t = \varepsilon$ exactly), so $\varepsilon$-prediction becomes degenerate there. Zero terminal SNR is paired with $v$-prediction, which at the last step reduces to predicting $-x_0$. Samplers must also start at the true final step (trailing timestep spacing).

## Complexity
$O(T)$.

## Common bugs
- Rescaling $\bar\alpha$ instead of $\sqrt{\bar\alpha}$. That gives a different, non-standard schedule.
- Forgetting to put $\bar\alpha'_0$ in front when undoing the cumulative product, which leaves you with $T-1$ betas.

## Interview follow-ups
- Why does a nonzero terminal SNR stop a model from generating a fully black image?
- Which DDIM timestep spacing ("leading", "linspace", "trailing") actually visits the final step $T-1$?
"""
