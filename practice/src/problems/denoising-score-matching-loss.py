import numpy as np

META = dict(title='Denoising score matching loss', tags=['Score Matching', 'Loss Functions'], diff='Medium', runtime='numpy',
            lessons=[], related=['gaussian-mixture-score', 'annealed-langevin-dynamics', 'noise-prediction-loss'])

STATEMENT = r"""
You cannot regress a network onto the score $\nabla\log p(x)$ of the data: nobody knows it. **Denoising score matching** sidesteps this. Perturb each example with Gaussian noise of scale $\sigma$; the score of the perturbation kernel is known exactly,

$$\tilde x = x + \sigma z, \qquad \nabla_{\tilde x}\log q_\sigma(\tilde x \mid x) = -\frac{\tilde x - x}{\sigma^2} = -\frac{z}{\sigma}$$

and regressing onto it gives (in expectation) the same minimiser as regressing onto the true noisy-data score.

The model here is a toy noise-conditional score network $s_\theta(\tilde x, \sigma) = (\tilde x W + b)/\sigma$. With a different $\sigma$ per example, the loss weights each one by $\lambda(\sigma) = \sigma^2$ so all noise levels contribute on a similar scale:

$$\mathcal L = \frac{1}{B}\sum_{b=1}^{B} \frac{\sigma_b^2}{2}\,\Big\lVert s_\theta(\tilde x_b, \sigma_b) + \frac{z_b}{\sigma_b} \Big\rVert^2$$

Write `dsm_loss(x, sigmas, noise, W, b)`.

## Inputs
- `x`: `(B, D)` clean data; `sigmas`: `(B,)` positive noise scales; `noise`: `(B, D)` the $z$.
- `W`: `(D, D)` applied as `x_tilde @ W`; `b`: `(D,)`.

## Output
A Python `float`: the loss above (squared norm summed over $D$, averaged over $B$).
"""

STARTER = '''import numpy as np

def dsm_loss(x, sigmas, noise, W, b):
    """sigma^2-weighted denoising score matching loss for s(x~, sigma) = (x~ @ W + b) / sigma."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dsm_loss(x, sigmas, noise, W, b):
    x, z, W, b = (np.asarray(a, dtype=float) for a in (x, noise, W, b))
    sg = np.asarray(sigmas, dtype=float)[:, None]
    x_tilde = x + sg * z
    score = (x_tilde @ W + b) / sg
    target = -z / sg                                   # score of N(x, sigma^2 I) at x_tilde
    per_example = 0.5 * (sg[:, 0] ** 2) * ((score - target) ** 2).sum(axis=1)
    return float(per_example.mean())
'''

HINTS = [
    'Build `x_tilde = x + sigma * z` with sigma as a `(B, 1)` column, then evaluate the model and the target $-z/\\sigma$.',
    'The weight is $\\sigma^2/2$ per example; the norm is a sum over features, and the batch reduction is a mean.',
    'Sanity check: $\\sigma^2\\lVert s_\\theta + z/\\sigma\\rVert^2 = \\lVert \\tilde x W + b + z\\rVert^2$, so a model that outputs $-z$ exactly scores 0.',
]

_r = np.random.default_rng(61)
TESTS = [
    (np.array([[0.0]]), np.array([1.0]), np.array([[1.0]]), np.array([[0.0]]), np.array([0.0])),
    (np.array([[1.0, 0.0]]), np.array([2.0]), np.array([[0.5, 0.5]]), np.zeros((2, 2)), np.array([1.0, 0.0])),
    (np.zeros((2, 1)), np.array([1.0, 3.0]), np.array([[1.0], [1.0]]), np.array([[-1.0]]), np.array([0.0])),
    (_r.normal(size=(5, 3)), np.array([0.01, 0.1, 1.0, 10.0, 50.0]), _r.normal(size=(5, 3)), _r.normal(size=(3, 3)) * 0.2, _r.normal(size=3) * 0.1),
    (_r.normal(size=(8, 2)), np.exp(_r.uniform(np.log(0.01), np.log(10), size=8)), _r.normal(size=(8, 2)), -np.eye(2), np.zeros(2)),
    (_r.normal(size=(4, 4)), np.full(4, 0.5), _r.normal(size=(4, 4)), np.zeros((4, 4)), np.zeros(4)),
]

EXPLANATION = r"""
## Why it works
Expand $\mathbb E\lVert s_\theta(\tilde x) - \nabla\log q_\sigma(\tilde x\mid x)\rVert^2$. The cross term $\mathbb E[s_\theta(\tilde x)^\top \nabla\log q_\sigma(\tilde x\mid x)]$, averaged over $x$, equals $\mathbb E[s_\theta(\tilde x)^\top \nabla\log q_\sigma(\tilde x)]$ by integrating against the marginal. So the denoising objective and the (intractable) explicit score-matching objective differ by a constant that does not depend on $\theta$. Their minimiser is the true score of the noisy data distribution.

## The weighting and the link to $\varepsilon$-prediction
The target $-z/\sigma$ has magnitude about $1/\sigma$, which explodes for small $\sigma$. Multiplying by $\lambda(\sigma) = \sigma^2$ makes every level's loss $O(1)$. With the model parameterised as $(\cdot)/\sigma$, the weighted loss is $\lVert \tilde xW + b + z\rVert^2$: the network is predicting $-z$, the (negated) noise. That is exactly DDPM's noise-prediction loss in a different coordinate system.

Test 3 makes this concrete: with $x = 0$ and $W = -1$ the network outputs $-\tilde x = -\sigma z$, which equals $-z$ only at $\sigma = 1$. That example scores 0; the $\sigma = 3$ one does not.

## Complexity
$O(BD^2)$ for the matrix product.

## Common bugs
- Target with the wrong sign: the score points from $\tilde x$ back **towards** $x$.
- Weighting by $\sigma$ instead of $\sigma^2$, or forgetting the weight, which lets the smallest $\sigma$ dominate (test 4).
- Dividing by $\sigma$ of shape `(B,)` against `(B, D)` without a trailing axis.

## Interview follow-ups
- Why does score matching at a single small $\sigma$ give poor estimates in low-density regions?
- What is sliced score matching, and when would you use it instead?
"""
