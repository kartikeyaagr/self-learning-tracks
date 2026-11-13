import numpy as np

META = dict(title='KL divergence between diagonal Gaussians', tags=['Generative Models', 'Probability', 'Loss Functions'], diff='Easy',
            runtime='numpy', lessons=['air-fanout-v1:01.12'],
            related=['kl-divergence-discrete-and-continuous', 'reparameterized-gaussian-sample', 'posterior-mean-and-variance'])

STATEMENT = r"""
The variational bound of a diffusion model (and the KL term of a VAE) is a sum of KL divergences between Gaussians with **diagonal** covariance. Because the dimensions are independent, the divergence is a sum of one-dimensional terms:

$$D_{\mathrm{KL}}\big(\mathcal N(\mu_1, \mathrm{diag}\,e^{v_1})\,\Vert\,\mathcal N(\mu_2, \mathrm{diag}\,e^{v_2})\big) = \frac12 \sum_{d} \left( v_{2,d} - v_{1,d} + \frac{e^{v_{1,d}} + (\mu_{1,d} - \mu_{2,d})^2}{e^{v_{2,d}}} - 1 \right)$$

where $v = \log\sigma^2$ is the log-variance.

Write `diag_gaussian_kl(mu1, logvar1, mu2, logvar2)`.

## Inputs
- Four float arrays. They broadcast against each other to a common shape `(..., D)`; the last axis is the event dimension $D$.
- A common case is `mu2 = 0`, `logvar2 = 0` passed as scalars-shaped arrays (KL to a standard normal).

## Output
Float array with the broadcast shape **minus the last axis** (the sum over $D$), in nats. For 1-D inputs that is a 0-d result; return it as a Python `float`.
"""

STARTER = '''import numpy as np

def diag_gaussian_kl(mu1, logvar1, mu2, logvar2):
    """KL(N1 || N2) summed over the last axis. Inputs are means and log-variances."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def diag_gaussian_kl(mu1, logvar1, mu2, logvar2):
    mu1, lv1, mu2, lv2 = (np.asarray(a, dtype=float) for a in (mu1, logvar1, mu2, logvar2))
    # exp(lv1 - lv2) instead of exp(lv1) / exp(lv2): no overflow for large log-variances
    terms = lv2 - lv1 + np.exp(lv1 - lv2) + (mu1 - mu2) ** 2 * np.exp(-lv2) - 1.0
    kl = 0.5 * terms.sum(axis=-1)
    return float(kl) if kl.ndim == 0 else kl
'''

HINTS = [
    'Each dimension contributes its own univariate Gaussian KL; the total is the sum over the last axis.',
    'Work with log-variances throughout: the variance ratio is `exp(lv1 - lv2)` and $1/\\sigma_2^2$ is `exp(-lv2)`.',
    'Let NumPy broadcast the four inputs, build the bracketed term elementwise, then `0.5 * terms.sum(axis=-1)`. Convert a 0-d result to `float`.',
]

_r = np.random.default_rng(5)
TESTS = [
    (np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0]), np.array([0.0])),
    (np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])),
    (np.array([0.0]), np.array([np.log(4.0)]), np.array([0.0]), np.array([0.0])),
    (np.array([[1.0, -1.0], [0.0, 2.0]]), np.zeros((2, 2)), np.zeros(2), np.zeros(2)),
    (_r.normal(size=(4, 3)), _r.normal(size=(4, 3)), _r.normal(size=(4, 3)), _r.normal(size=(4, 3))),
    (np.array([0.5, 0.5]), np.array([800.0, 800.0]), np.array([0.5, 0.5]), np.array([800.0, 800.0])),
    (_r.normal(size=(2, 3, 5)), _r.normal(size=(2, 3, 5)) - 2, _r.normal(size=(5,)), np.zeros(5)),
]

EXPLANATION = r"""
## Why it works
For independent dimensions the density factorises, the log-ratio is a sum, and so is its expectation. Each one-dimensional term is $\log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12$; writing $\log\frac{\sigma_2}{\sigma_1} = \frac12(v_2 - v_1)$ gives the formula. It is $0$ exactly when the two Gaussians match, and positive otherwise.

## Where diffusion uses it
Each term $L_{t-1}$ of the DDPM bound is $D_{\mathrm{KL}}(q(x_{t-1}\mid x_t, x_0)\,\Vert\,p_\theta(x_{t-1}\mid x_t))$ between two diagonal Gaussians. With a fixed shared variance it reduces to a scaled squared error between means, which is where the simple noise-prediction loss comes from.

## Complexity
$O(n)$ over the broadcast elements.

## Common bugs
- `np.exp(lv1) / np.exp(lv2)` overflows to `inf / inf = nan` for large log-variances (test 6). Subtract first.
- Averaging over $D$ instead of summing.
- Mixing up the direction: KL is not symmetric; the first distribution is the one the expectation is taken under.

## Interview follow-ups
- Derive the gradient of this KL with respect to $\mu_1$ and $v_1$.
- Why does a VAE's KL term to $\mathcal N(0, I)$ cause posterior collapse, and what is KL annealing?
"""
