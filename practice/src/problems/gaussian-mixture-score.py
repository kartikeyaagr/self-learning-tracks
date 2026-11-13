import numpy as np

META = dict(title='Score of an isotropic Gaussian mixture', tags=['Score Matching', 'Probability'], diff='Medium', runtime='numpy',
            lessons=[], related=['gradient-of-multivariate-gaussian-log-pdf', 'gaussian-mixture-models-em', 'annealed-langevin-dynamics'])

STATEMENT = r"""
The **score** of a density is the gradient of its log: $s(x) = \nabla_x \log p(x)$. Score-based diffusion models learn it, and a Gaussian mixture is the one non-trivial distribution where you can write it down exactly, which makes it the standard test bed.

For an isotropic mixture

$$p(x) = \sum_{k=1}^{K} w_k\,\mathcal N\!\left(x;\ \mu_k,\ s_k^2 I\right)$$

the score is a responsibility-weighted pull towards each mean:

$$s(x) = \sum_k r_k(x)\,\frac{\mu_k - x}{s_k^2}, \qquad r_k(x) = \frac{w_k\,\mathcal N(x; \mu_k, s_k^2 I)}{\sum_j w_j\,\mathcal N(x; \mu_j, s_j^2 I)}$$

Write `gmm_score(x, weights, means, stds)`.

## Inputs
- `x`: `(N, D)` query points.
- `weights`: `(K,)`, positive, summing to 1.
- `means`: `(K, D)`.
- `stds`: `(K,)`, the per-component standard deviation $s_k > 0$.

## Output
`(N, D)` float array of scores.

## Requirements
- Compute the responsibilities in log space (log-sum-exp). A point far from every mean must still get a finite, correct score: plain densities underflow to `0 / 0` there.
- Remember the normalising constant $(2\pi s_k^2)^{-D/2}$: it differs between components when their `stds` differ.
"""

STARTER = '''import numpy as np

def gmm_score(x, weights, means, stds):
    """x: (N, D), weights: (K,), means: (K, D), stds: (K,). Return grad_x log p(x), shape (N, D)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gmm_score(x, weights, means, stds):
    x, w, mu, s = (np.asarray(a, dtype=float) for a in (x, weights, means, stds))
    D = x.shape[1]
    diff = mu[None, :, :] - x[:, None, :]                    # (N, K, D): x -> mu_k
    sq = (diff ** 2).sum(-1)                                 # (N, K) squared distances
    var = s ** 2
    logp = np.log(w) - 0.5 * D * np.log(2 * np.pi * var) - 0.5 * sq / var   # log w_k N(x; mu_k, s_k^2 I)
    logp -= logp.max(axis=1, keepdims=True)                  # log-sum-exp shift
    r = np.exp(logp)
    r /= r.sum(axis=1, keepdims=True)                        # responsibilities (N, K)
    return np.einsum('nk,nkd->nd', r / var, diff)
'''

HINTS = [
    'Each component alone has score $(\\mu_k - x)/s_k^2$. The mixture score averages these with weights $r_k(x)$, the posterior probability of component $k$ given $x$.',
    'Build the `(N, K)` matrix of $\\log w_k + \\log\\mathcal N(x_n; \\mu_k, s_k^2 I)$, subtract each row\'s max, exponentiate and normalise. That is a softmax over components.',
    'With `diff = means[None] - x[:, None]` of shape `(N, K, D)`, the answer is `einsum("nk,nkd->nd", r / stds**2, diff)`. Do not forget the $-\\tfrac D2\\log(2\\pi s_k^2)$ term in the log-density.',
]

_r = np.random.default_rng(21)
TESTS = [
    (np.array([[1.0, 0.0]]), np.array([1.0]), np.array([[0.0, 0.0]]), np.array([1.0])),
    (np.array([[0.0], [3.0]]), np.array([0.5, 0.5]), np.array([[-1.0], [1.0]]), np.array([1.0, 1.0])),
    (np.array([[0.0, 0.0]]), np.array([0.5, 0.5]), np.array([[-2.0, 0.0], [2.0, 0.0]]), np.array([1.0, 1.0])),
    (np.array([[500.0, 0.0], [-500.0, 1.0]]), np.array([0.3, 0.7]), np.array([[-1.0, 0.0], [1.0, 0.0]]), np.array([0.5, 0.5])),
    (np.array([[0.5], [4.0]]), np.array([0.5, 0.5]), np.array([[0.0], [0.0]]), np.array([0.5, 3.0])),
    (_r.normal(size=(6, 3)) * 2, np.array([0.2, 0.3, 0.5]), _r.normal(size=(3, 3)) * 2, np.array([0.5, 1.0, 1.5])),
    (_r.normal(size=(4, 5)) * 20, np.array([0.6, 0.4]), _r.normal(size=(2, 5)), np.array([0.3, 0.8])),
]

EXPLANATION = r"""
## Why it works
$\nabla_x \log \sum_k f_k(x) = \sum_k \frac{f_k(x)}{\sum_j f_j(x)} \nabla_x \log f_k(x)$. Each $f_k = w_k \mathcal N(\cdot)$ has $\nabla \log f_k = (\mu_k - x)/s_k^2$, and the fraction is the responsibility $r_k(x)$. Near one mean the score points at that mean; between two equal components (test 3) the pulls cancel to $0$.

## Stability
At $x = 500$ every density is about $e^{-125000}$, which underflows to $0$ and gives `nan`. Subtracting the row max in log space leaves the ratios unchanged and keeps the largest term at $e^0 = 1$. Far from the data the score is dominated by the widest or closest component, as test 7 checks.

## Complexity
$O(NKD)$ time and memory for the `(N, K, D)` difference tensor.

## Common bugs
- Dropping the $-\tfrac D2\log(2\pi s_k^2)$ term. It cancels only when all `stds` are equal (test 5 has different ones).
- Normalising responsibilities over points instead of components (wrong axis).
- Computing $x - \mu_k$ and getting the sign of the score backwards.

## Interview follow-ups
- If you add Gaussian noise of variance $\sigma^2$ to samples from this mixture, what is the score of the noisy distribution? (Same mixture with $s_k^2 \to s_k^2 + \sigma^2$.)
- Why does score-based sampling with a single noise level mix poorly between well-separated modes?
"""
