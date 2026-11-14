import numpy as np

META = dict(title='Capstone: sample 2-D data with a respaced DDPM', tags=['Diffusion Models', 'Sampling'], diff='Hard', runtime='numpy',
            related=['cosine-noise-schedule', 'forward-diffusion-sample', 'ddpm-ancestral-step', 'respaced-ddpm-schedule'])

STATEMENT = r"""
Sample from a 2-D dataset with DDPM, using the exact optimal denoiser for that dataset in place of a trained network, and see how sample quality depends on the number of steps.

For a dataset of points $\{x^{(m)}\}$ and noisy input $x_t$, the optimal estimate of the clean point is a softmax-weighted average:
$$w_m \propto \exp\!\Big(-\frac{\lVert x_t - \sqrt{\bar\alpha_t}\,x^{(m)}\rVert^2}{2(1-\bar\alpha_t)}\Big),\quad \hat x_0 = \sum_m w_m x^{(m)},\quad \hat\epsilon = \frac{x_t - \sqrt{\bar\alpha_t}\,\hat x_0}{\sqrt{1-\bar\alpha_t}}.$$

Write `sample(data, betas, K, x_T, noise)`:

1. Respace `betas` to `K` steps as in the respacing step, giving timesteps, $\bar\alpha_i$ and $\beta'_i$ (with $\alpha'_i = 1 - \beta'_i$).
2. Start at `x_T (n, 2)`. For $i = K'-1$ down to 0 ($K'$ = number of kept steps):
   $x \leftarrow \frac{1}{\sqrt{\alpha'_i}}\Big(x - \frac{\beta'_i}{\sqrt{1-\bar\alpha_i}}\hat\epsilon\Big) + \sigma_i z_i$, with $\sigma_i^2 = \frac{1-\bar\alpha_{i-1}}{1-\bar\alpha_i}\beta'_i$ and $z_i$ = `noise[i]`. At $i = 0$, add no noise.
3. Score the samples against the data. For each sample find its nearest data point (lowest index on ties). Report `nn_dist`, the mean distance to that point; `coverage`, the fraction of data points that are the nearest point of at least one sample; and `max_share`, the largest fraction of samples sharing one nearest point.

Return a dict with `samples` (list of points), `nn_dist`, `coverage`, `max_share` and `steps` ($K'$).
"""

STARTER = '''import numpy as np

def sample(data, betas, K, x_T, noise):
    """Ancestral sampling over a respaced schedule with the exact dataset denoiser."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _respace(b, K):
    ab = np.cumprod(1.0 - b)
    t = np.round(np.linspace(0, len(b) - 1, K)).astype(int)
    _, first = np.unique(t, return_index=True)
    t = t[np.sort(first)]
    a = ab[t]
    return a, 1.0 - a / np.r_[1.0, a[:-1]]

def _eps(x, data, ab):
    d2 = ((x[:, None, :] - np.sqrt(ab) * data[None]) ** 2).sum(2)
    logw = -d2 / (2 * (1 - ab))
    w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    x0 = w @ data
    return (x - np.sqrt(ab) * x0) / np.sqrt(1 - ab)

def sample(data, betas, K, x_T, noise):
    data, b = np.asarray(data, float), np.asarray(betas, float)
    ab, bp = _respace(b, K)
    x = np.asarray(x_T, float).copy()
    noise = np.asarray(noise, float)
    n = len(ab)
    for i in range(n - 1, -1, -1):
        e = _eps(x, data, ab[i])
        x = (x - bp[i] / np.sqrt(1 - ab[i]) * e) / np.sqrt(1 - bp[i])
        if i > 0:
            var = (1 - ab[i - 1]) / (1 - ab[i]) * bp[i]
            x = x + np.sqrt(var) * noise[i]
    D = np.sqrt(((x[:, None, :] - data[None]) ** 2).sum(2))
    j = D.argmin(1)
    counts = np.bincount(j, minlength=len(data))
    return {"samples": x.tolist(), "nn_dist": float(D.min(1).mean()), "coverage": float((counts > 0).mean()),
            "max_share": float(counts.max() / len(x)), "steps": n}
'''

HINTS = [
    'Compute the weights in log space and subtract the row maximum before exponentiating; at low noise the distances are huge relative to 1 - alpha_bar.',
    'The respaced betas, not the original ones, go into both the mean update and the posterior variance.',
    'Index noise by the kept-step index i, not by the original timestep.',
]

def _cos(T, s=0.008):
    f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    t = np.arange(T + 1)
    return np.clip(1 - f(t[1:]) / f(t[:-1]), 0, 0.999)

_t = np.linspace(0.5, 3 * np.pi, 40)
SPIRAL = np.stack([_t * np.cos(_t), _t * np.sin(_t)], 1) / 5
RING = np.stack([np.cos(np.linspace(0, 2 * np.pi, 24, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False))], 1)

def _case(seed, data, T, K, n):
    g = np.random.default_rng(seed)
    return (data, _cos(T), K, g.normal(size=(n, 2)), g.normal(size=(K, n, 2)))

TESTS = [
    _case(1, SPIRAL, 200, 50, 80),
    _case(1, SPIRAL, 200, 5, 80),
    _case(1, SPIRAL, 200, 2, 80),
    _case(2, RING, 100, 20, 48),
    _case(2, RING, 100, 2, 48),
]

EXPLANATION = r"""
## Why the exact denoiser
A trained network approximates $\mathbb{E}[x_0 \mid x_t]$. For a finite dataset this expectation has a closed form, so the sampler can be tested in isolation: any bad sample is the sampler's fault, not the network's.

## What the tests show
- **Every sample is a training point.** The exact denoiser memorises, so with enough steps each sample lands on a data point and `nn_dist` is tiny. A real network generalises because it cannot represent this sharp function, which is one explanation for why diffusion models produce new images instead of copies.
- **Few steps lose diversity.** With 50 or 5 steps the samples cover over 80% of the spiral's points, about what 80 random draws from 40 points would give. With 2 steps the first jump goes straight from pure noise to the lowest noise level. At pure noise the best estimate of $x_0$ is close to the data mean, so most samples are pushed to the same place and then snap to the few data points nearest it: coverage collapses and `max_share` jumps.
- **Distance alone can mislead.** On the spiral, 2 steps still gives a small `nn_dist`. Only coverage shows the collapse. On the ring the 2-step samples still cover most points but sit visibly off the ring (an `nn_dist` about 70 times larger). Sample-quality metrics need to measure both fidelity and diversity, which is what precision and recall for generative models, or FID, try to do.

## Interview follow-ups
- Replace the ancestral step with the deterministic DDIM update. How do you expect `nn_dist` at 5 steps to change, and why?
"""
