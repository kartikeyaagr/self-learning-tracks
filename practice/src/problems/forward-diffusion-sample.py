import numpy as np

META = dict(title='Jump to any noise level: forward diffusion sample', tags=['Diffusion Models', 'Noise Schedules'], diff='Easy',
            runtime='numpy', lessons=['ieng:26.08'], related=['linear-beta-schedule', 'reparameterized-gaussian-sample', 'noise-prediction-loss'])

STATEMENT = r"""
Because each forward step is Gaussian, you never need to run the noising chain. The marginal at step $t$ is

$$q(x_t \mid x_0) = \mathcal N\!\left(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t) I\right) \quad\Longrightarrow\quad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$$

During training each example in a batch gets its **own** random step.

Write `q_sample(x0, t, alpha_bars, noise)`.

## Inputs
- `x0`: float array of shape `(B, ...)`: a batch of clean examples of any trailing shape (vectors, images).
- `t`: integer array of shape `(B,)`, 0-indexed steps into `alpha_bars`.
- `alpha_bars`: `(T,)` cumulative products.
- `noise`: same shape as `x0`.

## Output
`x_t`, same shape as `x0`. Example `b` uses `alpha_bars[t[b]]`.
"""

STARTER = '''import numpy as np

def q_sample(x0, t, alpha_bars, noise):
    """Return sqrt(abar_t) * x0 + sqrt(1 - abar_t) * noise with a per-example t."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def q_sample(x0, t, alpha_bars, noise):
    x0, noise = np.asarray(x0, dtype=float), np.asarray(noise, dtype=float)
    ab = np.asarray(alpha_bars, dtype=float)[np.asarray(t)]
    ab = ab.reshape((-1,) + (1,) * (x0.ndim - 1))    # (B, 1, 1, ...) so it broadcasts over each example
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * noise
'''

HINTS = [
    'Gather the per-example coefficients first: `alpha_bars[t]` has shape `(B,)`.',
    'A `(B,)` array does not broadcast against `(B, C, H, W)`: NumPy aligns trailing axes. Reshape it to `(B, 1, 1, 1)`.',
    '`ab.reshape((-1,) + (1,) * (x0.ndim - 1))` works for any trailing shape, then apply the formula.',
]

_r = np.random.default_rng(8)
_ab = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))
TESTS = [
    (np.array([[1.0, 2.0]]), np.array([0]), np.array([0.25]), np.array([[0.0, 1.0]])),
    (np.array([[1.0], [1.0]]), np.array([0, 1]), np.array([0.64, 0.36]), np.array([[1.0], [1.0]])),
    (np.ones((2, 2)), np.array([1, 1]), np.array([0.9, 0.0]), np.full((2, 2), 3.0)),
    (_r.normal(size=(4, 3)), np.array([0, 999, 500, 10]), _ab, _r.normal(size=(4, 3))),
    (_r.normal(size=(3, 2, 4, 4)), np.array([5, 200, 900]), _ab, _r.normal(size=(3, 2, 4, 4))),
    (np.array([[2.0, 2.0, 2.0]]), np.array([2]), np.array([1.0, 0.5, 1.0]), np.array([[-1.0, 0.0, 1.0]])),
]

EXPLANATION = r"""
## Why it works
One step is $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\varepsilon_t$. Substituting the previous step in, the two noise terms are independent Gaussians whose variances add: $\alpha_t\beta_{t-1} + \beta_t = 1 - \alpha_t\alpha_{t-1}$. By induction the signal coefficient is $\sqrt{\bar\alpha_t}$ and the total noise variance is $1 - \bar\alpha_t$. The coefficients' squares always sum to $1$, so unit-variance data stays unit-variance at every $t$ (variance preserving).

## Complexity
$O(n)$ in the batch size times the example size.

## Common bugs
- Broadcasting `alpha_bars[t]` of shape `(B,)` against `(B, D)` when `B == D`: it silently scales **columns** instead of rows. Always reshape.
- Using $\bar\alpha_t$ and $1 - \bar\alpha_t$ without square roots: those are variances, not scales.
- Indexing with a 1-based $t$ into a 0-based array.

## Interview follow-ups
- Why sample $t$ uniformly per example rather than once per batch?
- Write the same marginal for a variance-exploding process $x_t = x_0 + \sigma_t\varepsilon$. How do the two relate? ($x_t^{VE} = x_t^{VP}/\sqrt{\bar\alpha_t}$ with $\sigma_t^2 = (1-\bar\alpha_t)/\bar\alpha_t$.)
"""
