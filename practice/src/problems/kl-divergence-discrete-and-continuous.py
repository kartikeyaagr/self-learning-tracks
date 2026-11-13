import numpy as np

META = dict(title='KL divergence: categorical and Gaussian', tags=['Probability', 'Loss Functions'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:01.12', 'air-fanout-v1:01.11'],
            related=['expected-value-via-numerical-integration', 'cross-entropy-logits'])

STATEMENT = r"""
The Kullback-Leibler divergence measures how much extra information you need when you use $Q$ to encode samples from $P$:

$$D_{\mathrm{KL}}(P\,\Vert\,Q) = \sum_k p_k \log\frac{p_k}{q_k} \quad\text{(discrete)}, \qquad D_{\mathrm{KL}}(P\,\Vert\,Q) = \int p(x)\log\frac{p(x)}{q(x)}\,dx \quad\text{(continuous)}$$

Write `kl_divergence(p, q, kind)` that computes it row by row for a batch of pairs. Use the natural log.

## `kind == 'discrete'`
- `p`, `q`: float arrays of shape `(N, K)`; each row is a probability vector (non-negative, sums to 1).
- Conventions: a term with $p_k = 0$ contributes $0$ (even if $q_k = 0$). A term with $p_k > 0$ and $q_k = 0$ makes that row's divergence `inf`.
- No runtime warnings.

## `kind == 'gaussian'`
- `p`, `q`: float arrays of shape `(N, 2)`; row $n$ is `[mean, std]` of a univariate normal, with `std > 0`.
- Use the closed form for $D_{\mathrm{KL}}\bigl(\mathcal N(\mu_1, \sigma_1^2)\,\Vert\,\mathcal N(\mu_2, \sigma_2^2)\bigr)$. Deriving it is part of the exercise.

## Output
A float array of shape `(N,)`.
"""

STARTER = '''import numpy as np

def kl_divergence(p, q, kind):
    """Row-wise KL(p || q). kind is 'discrete' ((N, K) probabilities) or 'gaussian' ((N, 2) rows of [mean, std])."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def kl_divergence(p, q, kind):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if kind == 'discrete':
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0)   # 0 log 0 = 0; p > 0, q = 0 -> inf
        return terms.sum(axis=1)
    (m1, s1), (m2, s2) = p.T, q.T
    return np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5
'''

HINTS = [
    'Discrete: mask the terms with $p_k = 0$ using `np.where(p > 0, ..., 0.0)`, and silence the warnings that the masked-out branch would raise with `np.errstate`.',
    'Gaussian: $\\log\\frac{p(x)}{q(x)} = \\log\\frac{\\sigma_2}{\\sigma_1} - \\frac{(x-\\mu_1)^2}{2\\sigma_1^2} + \\frac{(x-\\mu_2)^2}{2\\sigma_2^2}$. Take its expectation under $P$ using $E[(x-\\mu_1)^2] = \\sigma_1^2$ and $E[(x - \\mu_2)^2] = \\sigma_1^2 + (\\mu_1 - \\mu_2)^2$.',
    'The result is $\\log\\frac{\\sigma_2}{\\sigma_1} + \\frac{\\sigma_1^2 + (\\mu_1-\\mu_2)^2}{2\\sigma_2^2} - \\frac12$, applied to whole columns at once.',
]

_r = np.random.default_rng(9)
TESTS = [
    (np.array([[0.5, 0.5]]), np.array([[0.9, 0.1]]), 'discrete'),
    (np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 2.0], [0.0, 1.0]]), 'gaussian'),
    (np.array([[0.2, 0.3, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.0]]),
     np.array([[0.2, 0.3, 0.5], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]), 'discrete'),
    (np.array([[1.0, 0.0, 0.0]]), np.array([[0.1, 0.0, 0.9]]), 'discrete'),
    (np.array([[3.0, 1.0], [0.0, 1e-3]]), np.array([[3.0, 1.0], [0.0, 1.0]]), 'gaussian'),
    (_r.dirichlet(np.ones(6), size=5), _r.dirichlet(np.ones(6), size=5), 'discrete'),
    (np.c_[_r.normal(size=6) * 3, _r.uniform(0.1, 3, size=6)], np.c_[_r.normal(size=6), _r.uniform(0.1, 3, size=6)], 'gaussian'),
]

EXPLANATION = r"""
## The conventions come from limits
$\lim_{p\to 0^+} p\log p = 0$, so missing outcomes under $P$ cost nothing. But if $P$ puts mass where $Q$ has none, the log ratio is infinite: $Q$ says the event is impossible and $P$ produces it. That asymmetry is the whole personality of KL: $D_{\mathrm{KL}}(P\Vert Q)$ punishes $Q$ for missing $P$'s mass (mass-covering), and the reverse direction punishes $Q$ for putting mass where $P$ has none (mode-seeking).

## The Gaussian closed form
Taking $E_P$ of the log ratio, the $(x-\mu_1)^2/(2\sigma_1^2)$ term averages to $\tfrac12$ and $(x-\mu_2)^2$ averages to $\sigma_1^2 + (\mu_1-\mu_2)^2$. So

$$D_{\mathrm{KL}} = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12$$

It's 0 iff the parameters match. This is the KL term in a VAE's loss, with $Q = \mathcal N(0, 1)$.

## Complexity
$O(NK)$ for the discrete case, $O(N)$ for the Gaussian case.

## Common bugs
- `np.sum(p * np.log(p / q))`: `0 * log(0/0)` is `nan`, which poisons the row.
- Mixing up the direction: the expectation is under $P$, the first argument.
- Using variances where the formula wants standard deviations in the log term (that's an off-by-a-factor-of-2 bug: $\log\frac{\sigma_2}{\sigma_1} = \tfrac12\log\frac{\sigma_2^2}{\sigma_1^2}$).

## Interview follow-ups
- Show that cross-entropy $H(P, Q) = H(P) + D_{\mathrm{KL}}(P\Vert Q)$, so minimising cross-entropy over $Q$ minimises KL.
- Extend the Gaussian formula to $d$-dimensional normals with full covariances.
"""
