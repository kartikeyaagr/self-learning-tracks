import numpy as np

META = dict(title='Annealed Langevin dynamics', tags=['Score Matching', 'Sampling'], diff='Medium', runtime='numpy',
            lessons=[], related=['gaussian-mixture-score', 'denoising-score-matching-loss', 'karras-euler-sampler'])

STATEMENT = r"""
Langevin dynamics samples from a density using only its score. Repeating

$$x \leftarrow x + \frac{\alpha}{2}\,\nabla_x\log p(x) + \sqrt{\alpha}\,z, \qquad z \sim \mathcal N(0, I)$$

converges to $p$ as the step size shrinks. Starting from noise, a single small noise level mixes badly, so **annealed** Langevin runs a few steps at each of a decreasing sequence of noise levels $\sigma_1 > \dots > \sigma_L$, using the score of the data **perturbed** at that level. The step size shrinks with the level:

$$\alpha_i = \epsilon\cdot\frac{\sigma_i^2}{\sigma_L^2}$$

For a closed-form score, the data is $\mathcal N(\mu, s^2 I)$. Perturbed with noise of scale $\sigma$ it is $\mathcal N(\mu, (s^2+\sigma^2) I)$, whose score is

$$\nabla_x\log p_\sigma(x) = \frac{\mu - x}{s^2 + \sigma^2}$$

Write `annealed_langevin(x_init, sigmas, eps, noise, mu, s)`.

## Inputs
- `x_init`: `(N, D)` starting points (used as given).
- `sigmas`: `(L,)` strictly decreasing positive noise levels; $\sigma_L$ is the last entry.
- `eps`: float base step size $\epsilon$.
- `noise`: `(L, K, N, D)` standard-normal draws. Level $i$ runs $K$ steps; step $k$ at level $i$ uses `noise[i, k]`.
- `mu`: `(D,)`; `s`: float data std.

## Output
The final `(N, D)` samples after all $L \times K$ updates, in order (all of level 0, then level 1, ...).
"""

STARTER = '''import numpy as np

def annealed_langevin(x_init, sigmas, eps, noise, mu, s):
    """Run K Langevin steps at each noise level (noise has shape (L, K, N, D)). Return final x."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def annealed_langevin(x_init, sigmas, eps, noise, mu, s):
    x = np.asarray(x_init, dtype=float).copy()
    sig, z, mu = np.asarray(sigmas, dtype=float), np.asarray(noise, dtype=float), np.asarray(mu, dtype=float)
    for i, sg in enumerate(sig):
        alpha = eps * sg ** 2 / sig[-1] ** 2               # larger steps at larger noise
        for k in range(z.shape[1]):
            score = (mu - x) / (s * s + sg * sg)
            x = x + 0.5 * alpha * score + np.sqrt(alpha) * z[i, k]
    return x
'''

HINTS = [
    'Two nested loops: outer over noise levels, inner over the `K` steps. `K` is `noise.shape[1]`.',
    'At level $i$, the step size is `eps * sigmas[i]**2 / sigmas[-1]**2` and the score uses variance $s^2 + \\sigma_i^2$.',
    'Recompute the score from the **current** `x` at every inner step, then `x = x + 0.5 * alpha * score + sqrt(alpha) * noise[i, k]`.',
]

_r = np.random.default_rng(67)
_sig = np.geomspace(10.0, 0.1, 5)
TESTS = [
    (np.array([[2.0]]), np.array([1.0]), 1.0, np.zeros((1, 1, 1, 1)), np.array([0.0]), 1.0),
    (np.array([[0.0]]), np.array([2.0, 1.0]), 0.5, np.ones((2, 1, 1, 1)), np.array([1.0]), 0.0),
    (np.zeros((3, 2)), _sig, 1e-3, _r.normal(size=(5, 4, 3, 2)), np.array([1.0, -1.0]), 0.5),
    (_r.normal(size=(4, 2)) * 10, _sig, 2e-4, _r.normal(size=(5, 10, 4, 2)), np.array([3.0, 0.0]), 1.0),
    (_r.normal(size=(2, 3)), np.array([1.0, 0.5, 0.25]), 0.01, np.zeros((3, 6, 2, 3)), np.array([0.0, 1.0, 2.0]), 0.2),
    (_r.normal(size=(6, 1)), np.array([5.0, 1.0]), 0.05, _r.normal(size=(2, 3, 6, 1)), np.array([-2.0]), 0.3),
]

EXPLANATION = r"""
## Why it works
Langevin dynamics is a discretised diffusion whose stationary distribution is $p$: the score term drifts towards high density and the noise term keeps it from collapsing onto the mode. At large $\sigma$ the perturbed density is broad and smooth, so samples travel far and find every mode; as $\sigma$ shrinks, the perturbed density sharpens towards the data. Scaling $\alpha_i \propto \sigma_i^2$ keeps the signal-to-noise ratio of each update roughly the same at every level.

With zero injected noise (test 5) the update is gradient ascent on $\log p_\sigma$, and every point slides deterministically towards $\mu$.

## Complexity
$O(LKND)$ time, $O(ND)$ memory beyond the noise array.

## Common bugs
- Using the clean-data score $(\mu - x)/s^2$ at every level, which ignores the annealing (and divides by $0$ when $s = 0$, test 2).
- Normalising the step by $\sigma_1$ instead of $\sigma_L$, so the smallest level gets $\alpha = \epsilon\,\sigma_L^2/\sigma_1^2$ instead of $\epsilon$.
- Forgetting the $\tfrac12$ on the drift: then the chain targets $p^2$ (a sharper distribution) instead of $p$.

## Interview follow-ups
- How does this relate to predictor-corrector sampling in score-based SDEs?
- What happens to the sample variance if $\epsilon$ is too large? (Discretisation inflates it; a Metropolis correction would fix it.)
"""
