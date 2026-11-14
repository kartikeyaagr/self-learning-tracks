import numpy as np

META = dict(title='Reparameterized Gaussian sample', tags=['Generative Models', 'Probability'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:11.25'], related=['diagonal-gaussian-kl-divergence', 'forward-diffusion-sample'])

STATEMENT = r"""
You cannot backpropagate through "draw a sample from $\mathcal N(\mu, \sigma^2)$". The **reparameterization trick** moves the randomness into a fixed input $\varepsilon \sim \mathcal N(0, 1)$ and writes the sample as a deterministic, differentiable function of the parameters:

$$z = \mu + \sigma \odot \varepsilon, \qquad \sigma = \exp\!\left(\tfrac12 \log\sigma^2\right)$$

Networks usually predict the **log-variance** rather than $\sigma$, because it is unconstrained. The same trick is what makes a diffusion forward sample $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ cheap and differentiable.

Write `reparam_sample(mu, log_var, eps)` and return `(z, dz_dlogvar)`.

## Inputs
- `mu`, `log_var`: float arrays of the same shape `S` (any number of dimensions).
- `eps`: standard-normal noise of shape `S`, or of shape `(K,) + S` to draw `K` samples per distribution at once. The noise is given to you; do not generate any.

## Output
- `z`: array with the shape of `eps`.
- `dz_dlogvar`: the elementwise derivative $\partial z / \partial \log\sigma^2$, same shape as `eps`.
"""

STARTER = '''import numpy as np

def reparam_sample(mu, log_var, eps):
    """Return (z, dz/dlog_var), both shaped like eps."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def reparam_sample(mu, log_var, eps):
    mu, log_var, eps = (np.asarray(a, dtype=float) for a in (mu, log_var, eps))
    sigma = np.exp(0.5 * log_var)            # broadcasts over a leading sample axis in eps
    z = mu + sigma * eps
    return z, 0.5 * sigma * eps              # d/dlv of exp(lv/2) * eps
'''

HINTS = [
    'Turn the log-variance into a standard deviation first: $\\sigma = e^{\\log\\sigma^2 / 2}$, not $\\sqrt{\\cdot}$ of the log.',
    'Broadcasting already handles the optional leading `K` axis: `mu + sigma * eps` works for `eps` of shape `S` or `(K,) + S`.',
    'Differentiate $\\mu + e^{v/2}\\varepsilon$ with respect to $v$: you get $\\tfrac12 e^{v/2}\\varepsilon = \\tfrac12\\sigma\\varepsilon$.',
]

_r = np.random.default_rng(11)
TESTS = [
    (np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, -1.0])),
    (np.array([2.0]), np.array([np.log(4.0)]), np.array([0.5])),
    (np.array([[0.0, 0.0]]), np.array([[-2.0, 2.0]]), np.zeros((1, 2))),
    (np.array([1.0, -1.0]), np.array([0.0, np.log(9.0)]), np.array([[1.0, 1.0], [-1.0, 2.0], [0.0, -0.5]])),
    (_r.normal(size=(3, 4)), _r.normal(size=(3, 4)), _r.normal(size=(3, 4))),
    (_r.normal(size=(2, 3)), _r.normal(size=(2, 3)) - 3.0, _r.normal(size=(5, 2, 3))),
]

EXPLANATION = r"""
## Why it works
If $\varepsilon \sim \mathcal N(0, 1)$ then $\mu + \sigma\varepsilon \sim \mathcal N(\mu, \sigma^2)$: shifting and scaling a Gaussian keeps it Gaussian. The sample now depends on $\mu$ and $\log\sigma^2$ through ordinary arithmetic, so gradients flow: $\partial z/\partial\mu = 1$ and $\partial z/\partial \log\sigma^2 = \tfrac12\sigma\varepsilon$. The expectation of a loss over $z$ becomes an expectation over a fixed $\varepsilon$, and a Monte-Carlo average of the gradient is unbiased.

## Complexity
$O(n)$ in the number of elements of `eps`.

## Common bugs
- Using $\sqrt{\log\sigma^2}$ or $\log\sigma^2$ directly as the standard deviation.
- Forgetting the $\tfrac12$ in the derivative (it comes from the chain rule through $e^{v/2}$).
- Generating fresh noise inside the function. Passing `eps` in keeps training reproducible and makes the function testable.

## Interview follow-ups
- Why can't the score-function (REINFORCE) estimator compete with this one on variance for continuous latents?
- How would you reparameterize a full-covariance Gaussian? (Use a Cholesky factor: $z = \mu + L\varepsilon$.)
"""
