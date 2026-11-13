import numpy as np

META = dict(title='Gaussian mixture model fitted with EM', tags=['Clustering', 'Probability'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:12.02', 'mlmaths:11.05'], related=['k-means-clustering', 'gaussian-naive-bayes'])

STATEMENT = r"""
A Gaussian mixture models data as coming from $K$ Gaussians with mixing weights $\pi_k$, means $\mu_k$ and full covariances $\Sigma_k$:

$$p(x) = \sum_{k=1}^K \pi_k\, \mathcal N(x;\ \mu_k, \Sigma_k).$$

Expectation-maximisation alternates two steps:

- **E-step**: responsibilities $r_{ik} = \dfrac{\pi_k\,\mathcal N(x_i;\mu_k,\Sigma_k)}{\sum_j \pi_j\,\mathcal N(x_i;\mu_j,\Sigma_j)}$.
- **M-step**: with $N_k = \sum_i r_{ik}$: $\pi_k = N_k / n$, $\mu_k = \frac{1}{N_k}\sum_i r_{ik} x_i$, $\Sigma_k = \frac{1}{N_k}\sum_i r_{ik}(x_i - \mu_k)(x_i - \mu_k)^\top + \lambda I$, using the **new** $\mu_k$.

Write `gmm_em(X, means_init, n_iter, reg)` that returns `(weights, means, covs, avg_log_lik)`.

## Procedure (follow exactly)
1. Initialise $\pi_k = 1/K$, $\mu_k$ = rows of `means_init`, $\Sigma_k = I$.
2. Repeat `n_iter` times: one E-step, then one M-step (with $\lambda$ = `reg` added to every diagonal).
3. `avg_log_lik` is $\frac1n\sum_i \log p(x_i)$ under the **final** parameters.

## Input / output
- `X`: float `(n, d)`; `means_init`: float `(K, d)`; `n_iter >= 0`; `reg >= 0`.
- Return `weights` `(K,)`, `means` `(K, d)`, `covs` `(K, d, d)` as float arrays, and `avg_log_lik` as a Python `float`.
- Compute log densities in log space (via Cholesky or `slogdet`) and normalise responsibilities with log-sum-exp, so far-away points don't underflow to `0/0`.
"""

STARTER = '''import numpy as np

def gmm_em(X, means_init, n_iter, reg):
    """EM for a full-covariance GMM. Return (weights (K,), means (K, d), covs (K, d, d), avg_log_lik)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def log_gauss(X, mu, cov):
    """log N(x; mu, cov) for every row of X, via a Cholesky factor."""
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mu).T)                         # L z = (x - mu), so |z|^2 is the Mahalanobis term
    log_det = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (X.shape[1] * np.log(2 * np.pi) + log_det + (z ** 2).sum(axis=0))

def log_joint(X, w, mu, cov):
    return np.column_stack([np.log(w[k]) + log_gauss(X, mu[k], cov[k]) for k in range(len(w))])   # (n, K)

def logsumexp_rows(A):
    m = A.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(axis=1, keepdims=True)))[:, 0]

def gmm_em(X, means_init, n_iter, reg):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    mu = np.array(means_init, dtype=float)
    K = len(mu)
    w, cov = np.full(K, 1.0 / K), np.array([np.eye(d)] * K)
    for _ in range(n_iter):
        lj = log_joint(X, w, mu, cov)
        R = np.exp(lj - logsumexp_rows(lj)[:, None])             # E-step, rows sum to 1
        Nk = R.sum(axis=0)
        w = Nk / n                                               # M-step
        mu = (R.T @ X) / Nk[:, None]
        for k in range(K):
            D = X - mu[k]
            cov[k] = (R[:, k, None] * D).T @ D / Nk[k] + reg * np.eye(d)
    return w, mu, cov, float(logsumexp_rows(log_joint(X, w, mu, cov)).mean())
'''

HINTS = [
    'Write `log_gauss(X, mu, cov)` first. With `L = cholesky(cov)` and `z = solve(L, (X - mu).T)`, the log density is $-\\tfrac12(d\\log 2\\pi + 2\\sum\\log L_{jj} + \\lVert z\\rVert^2)$.',
    'E-step in log space: build the `(n, K)` matrix $\\log\\pi_k + \\log\\mathcal N$, subtract its row-wise log-sum-exp, and exponentiate. The same log-sum-exp, averaged, is the log-likelihood.',
    'M-step: `Nk = R.sum(0)`, `w = Nk / n`, `mu = R.T @ X / Nk[:, None]`, and for each component `((R[:, k, None] * D).T @ D) / Nk[k] + reg * I` with `D = X - mu[k]` using the new mean.',
]

_r = np.random.default_rng(101)
_A = np.vstack([_r.multivariate_normal([0, 0], [[1.0, 0.6], [0.6, 0.8]], size=40),
                _r.multivariate_normal([4, 3], [[0.5, -0.2], [-0.2, 0.4]], size=30)])
_B = np.concatenate([_r.normal(-2, 0.5, size=25), _r.normal(1, 1.0, size=35), _r.normal(5, 0.7, size=20)])[:, None]
TESTS = [
    (np.array([[0.0], [0.2], [5.0], [5.2]]), np.array([[0.0], [5.0]]), 1, 1e-6),
    (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [6.0, 6.0], [7.0, 6.0], [6.0, 7.0]]), np.array([[1.0, 1.0], [5.0, 5.0]]), 5, 1e-3),
    (np.array([[1.0], [2.0], [3.0]]), np.array([[2.0]]), 0, 1e-6),
    (np.array([[1.0], [2.0], [3.0]]), np.array([[0.0]]), 2, 0.0),
    (np.array([[0.0], [0.5], [1.0], [100.0], [101.0]]), np.array([[0.0], [100.0]]), 10, 1e-6),
    (_A, np.array([[1.0, 1.0], [3.0, 2.0]]), 30, 1e-6),
    (_B, np.array([[-1.0], [0.0], [3.0]]), 40, 1e-6),
]

EXPLANATION = r"""
## Why EM increases the likelihood
For any responsibilities $r$, Jensen's inequality gives $\log p(X) \ge \sum_{i,k} r_{ik}\log\frac{\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)}{r_{ik}}$. The E-step chooses $r$ equal to the true posterior, which makes the bound tight. The M-step maximises the bound over the parameters in closed form (weighted MLEs). So the log-likelihood never decreases, and EM converges to a stationary point, usually a local maximum, depending on `means_init`.

## Why reg
If a component collapses onto a single point, $\Sigma_k \to 0$ and the likelihood goes to $+\infty$: the MLE of a GMM is degenerate. Adding $\lambda I$ keeps covariances positive definite (test 5 has a two-point cluster that would otherwise get very tight, and test 4 shows that $\lambda = 0$ is fine when nothing collapses).

## Relationship to k-means
With shared covariance $\sigma^2 I$, equal weights, and $\sigma \to 0$, responsibilities become hard 0/1 nearest-mean assignments and EM turns into Lloyd's algorithm.

## Complexity
$O(nKd^2 + Kd^3)$ per iteration: the Cholesky factors and triangular solves dominate.

## Common bugs
- Computing densities directly and normalising: points 100 units away (test 5) underflow to `0/0`.
- Using the old mean in the covariance update.
- Using `np.linalg.inv` and `det` separately: `det` overflows or underflows in higher dimensions, `slogdet` or Cholesky doesn't.
- Returning the log-likelihood from the last E-step (before the final M-step) instead of under the final parameters.

## Interview follow-ups
- How do you choose $K$? (BIC/AIC, which penalise the $K(d + d(d+1)/2) + K - 1$ free parameters.)
- How would you make EM work on data that doesn't fit in memory? (Mini-batch / online EM with running sufficient statistics.)
"""
