import numpy as np

META = dict(title='Classifier-free guidance with rescaling', tags=['Conditional Generation', 'Sampling'], diff='Medium',
            runtime='numpy', lessons=['ieng:26.09'], related=['ddim-step', 'ddpm-ancestral-step'])

STATEMENT = r"""
A text-to-image model is trained to work both with a prompt and, some of the time, with the prompt dropped. At sampling time you run it both ways and **extrapolate** away from the unconditional prediction:

$$\hat\varepsilon_{g} = \hat\varepsilon_{u} + w\,(\hat\varepsilon_{c} - \hat\varepsilon_{u})$$

$w = 1$ is the plain conditional model, $w = 0$ unconditional, and $w > 1$ pushes harder towards the prompt. Large $w$ also inflates the prediction's scale, which over-saturates images. **Guidance rescale** pulls each example's standard deviation back towards the conditional one and blends:

$$\hat\varepsilon_{r} = \hat\varepsilon_{g}\cdot\frac{\mathrm{std}(\hat\varepsilon_{c})}{\mathrm{std}(\hat\varepsilon_{g})}, \qquad \text{out} = \phi\,\hat\varepsilon_{r} + (1-\phi)\,\hat\varepsilon_{g}$$

Write `cfg(eps_cond, eps_uncond, scale, rescale=0.0)`.

## Inputs
- `eps_cond`, `eps_uncond`: float arrays of the same shape `(B, ...)`, batch first.
- `scale`: guidance weight $w$ (float). `rescale`: $\phi \in [0, 1]$.

## Details
- `std` is the **population** standard deviation (`ddof=0`) of one example, taken over all its non-batch axes.
- When `rescale == 0` skip the rescaling entirely (no std computed). Tests with `rescale > 0` never have $\mathrm{std}(\hat\varepsilon_g) = 0$.

## Output
Float array with the same shape as the inputs.
"""

STARTER = '''import numpy as np

def cfg(eps_cond, eps_uncond, scale, rescale=0.0):
    """Return the guided (and optionally rescaled) noise prediction, same shape as the inputs."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def cfg(eps_cond, eps_uncond, scale, rescale=0.0):
    c, u = np.asarray(eps_cond, dtype=float), np.asarray(eps_uncond, dtype=float)
    g = u + scale * (c - u)
    if rescale == 0:
        return g
    axes = tuple(range(1, c.ndim))                       # everything but the batch axis
    ratio = c.std(axis=axes, keepdims=True) / g.std(axis=axes, keepdims=True)
    return rescale * (g * ratio) + (1.0 - rescale) * g
'''

HINTS = [
    'The guided prediction is one line: `u + scale * (c - u)`. Check it gives `c` at scale 1 and `u` at scale 0.',
    'Per-example std means reducing over every axis except 0: `axes = tuple(range(1, x.ndim))` with `keepdims=True` so it broadcasts back.',
    'Rescaled is `g * std(c) / std(g)`; the output is a convex blend `phi * rescaled + (1 - phi) * g`.',
]

_r = np.random.default_rng(47)
_c, _u = _r.normal(size=(2, 3, 4, 4)), _r.normal(size=(2, 3, 4, 4))
TESTS = [
    (np.array([[1.0, 2.0]]), np.array([[0.0, 0.0]]), 3.0),
    (np.array([[1.0, -1.0]]), np.array([[0.0, 0.0]]), 2.0, 1.0),
    (_c, _u, 1.0),
    (_c, _u, 0.0),
    (_c, _u, 7.5),
    (_c, _u, 7.5, 0.7),
    (_r.normal(size=(3, 5)), _r.normal(size=(3, 5)) * 0.5, 4.0, 1.0),
]

EXPLANATION = r"""
## Why it works
With Bayes, $\nabla\log p(x\mid c) = \nabla\log p(x) + \nabla\log p(c\mid x)$. The conditional minus unconditional prediction estimates the "classifier" gradient $\nabla\log p(c\mid x)$ (up to a $-\sigma$ factor), with no separate classifier. Scaling it by $w$ samples from a sharpened distribution $\propto p(x)\,p(c\mid x)^w$: closer to the prompt, less diverse.

## Why rescale
At $w = 7.5$ the guided prediction has a much larger standard deviation than anything seen in training, and the decoded image comes out washed-out or over-contrasted. Matching the per-example std of the conditional prediction fixes the scale while keeping the guided direction; $\phi \approx 0.7$ blends in some of the original to avoid overly flat images. Test 2 shows full rescaling at $\phi = 1$: the guided vector is doubled, then scaled back to the conditional vector's std.

## Complexity
$O(n)$, but the real cost is the **two** network evaluations per step (usually batched together as a batch of $2B$).

## Common bugs
- Swapping the arguments, which guides away from the prompt.
- Taking std over the whole batch, which couples different images.
- Using `ddof=1`: harmless in direction but changes the numbers.

## Interview follow-ups
- How do you train a single network to give both predictions? (Drop the condition with probability around 10% and replace it with a null embedding.)
- How would you guide with a negative prompt?
"""
