import numpy as np

META = dict(title='Gradients of the multivariate Gaussian log-density', tags=['Probability', 'Calculus'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:11.25', 'mlmaths:12.10'],
            related=['gradient-of-matrix-valued-function', 'kl-divergence-discrete-and-continuous'])

STATEMENT = r"""
The log-density of $\mathcal N(\mu, \Sigma)$ in $d$ dimensions is

$$\log p(x) = -\tfrac12\Bigl(d\log 2\pi + \log\det\Sigma + (x-\mu)^\top\Sigma^{-1}(x-\mu)\Bigr)$$

Its gradients drive score-based models (with respect to $x$) and maximum-likelihood fitting (with respect to $\mu$ and $\Sigma$).

Write `gaussian_logpdf_grads(X, mu, Sigma)` that returns `(logp, dX, dmu, dSigma)`:

| Output | Shape | Meaning |
|---|---|---|
| `logp` | `(N,)` | $\log p(x_n)$ for every row of `X` |
| `dX` | `(N, d)` | row $n$: $\nabla_{x_n} \log p(x_n)$ (the score) |
| `dmu` | `(d,)` | $\nabla_\mu \sum_n \log p(x_n)$ |
| `dSigma` | `(d, d)` | $\nabla_\Sigma \sum_n \log p(x_n)$, treating the $d^2$ entries of $\Sigma$ as independent |

## Requirements
- `X`: `(N, d)`, `mu`: `(d,)`, `Sigma`: `(d, d)` symmetric positive definite; all floats.
- Don't form $\Sigma^{-1}(x - \mu)$ with `np.linalg.inv` in a loop; use one factorisation (Cholesky or `solve`) for the whole batch.
- No Python loops over samples.
"""

STARTER = '''import numpy as np

def gaussian_logpdf_grads(X, mu, Sigma):
    """Return (log N(x_n | mu, Sigma) per row, score per row, grad wrt mu, grad wrt Sigma) of the summed log-density."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gaussian_logpdf_grads(X, mu, Sigma):
    X, mu, Sigma = (np.asarray(a, dtype=float) for a in (X, mu, Sigma))
    N, d = X.shape
    Lc = np.linalg.cholesky(Sigma)                         # Sigma = Lc Lc^T
    D = X - mu                                             # (N, d)
    Sinv = np.linalg.solve(Lc.T, np.linalg.solve(Lc, np.eye(d)))
    S = D @ Sinv                                           # rows: Sigma^{-1} (x_n - mu); Sinv is symmetric
    logdet = 2 * np.sum(np.log(np.diag(Lc)))
    maha = np.sum(D * S, axis=1)
    logp = -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
    dSigma = 0.5 * (S.T @ S - N * Sinv)                    # sum_n of 0.5 (S_n S_n^T - Sigma^{-1})
    return logp, -S, S.sum(axis=0), dSigma
'''

HINTS = [
    'Let $r_n = \\Sigma^{-1}(x_n - \\mu)$. Then $\\nabla_{x_n}\\log p = -r_n$ and $\\nabla_\\mu \\sum_n \\log p = \\sum_n r_n$, since the quadratic form\'s gradient is $2\\Sigma^{-1}(x-\\mu)$.',
    'For $\\Sigma$ you need two matrix identities: $\\nabla_\\Sigma \\log\\det\\Sigma = \\Sigma^{-1}$ (for symmetric $\\Sigma$) and $\\nabla_\\Sigma\\, a^\\top\\Sigma^{-1}a = -\\Sigma^{-1}aa^\\top\\Sigma^{-1}$.',
    'Summed over samples: $\\nabla_\\Sigma = \\tfrac12\\bigl(R^\\top R - N\\Sigma^{-1}\\bigr)$ with $R$ the matrix of rows $r_n$. Get $\\log\\det\\Sigma$ as `2 * sum(log(diag(cholesky(Sigma))))`.',
]

_r = np.random.default_rng(73)
_M = _r.normal(size=(3, 3))
_S3 = _M @ _M.T + 0.5 * np.eye(3)
TESTS = [
    (np.array([[0.0, 0.0], [1.0, 0.0]]), np.zeros(2), np.eye(2)),
    (np.array([[1.0, 2.0]]), np.array([1.0, 0.0]), np.diag([1.0, 4.0])),
    (np.array([[0.5, -0.5], [2.0, 1.0], [-1.0, 0.0]]), np.array([0.5, 0.0]), np.array([[2.0, 0.8], [0.8, 1.0]])),
    (np.array([[3.0]]), np.array([1.0]), np.array([[0.25]])),
    (_r.normal(size=(10, 3)) @ np.linalg.cholesky(_S3).T + np.array([1.0, -1.0, 0.5]), np.array([1.0, -1.0, 0.5]), _S3),
    (_r.normal(size=(6, 4)) * 3, _r.normal(size=4), np.diag([0.1, 1.0, 10.0, 100.0])),
]

EXPLANATION = r"""
## The three gradients
With $r_n = \Sigma^{-1}(x_n - \mu)$:
- **Score**: $\nabla_x \log p = -\Sigma^{-1}(x - \mu)$. It points from $x$ back towards the mean, scaled by the precision; score-matching and diffusion models learn exactly this vector field.
- **Mean**: $\nabla_\mu \sum_n \log p = \sum_n r_n$. Setting it to zero gives the MLE $\hat\mu = \bar x$.
- **Covariance**: $\nabla_\Sigma \sum_n \log p = \tfrac12\bigl(\Sigma^{-1}\bigl(\sum_n (x_n-\mu)(x_n-\mu)^\top\bigr)\Sigma^{-1} - N\Sigma^{-1}\bigr)$. Setting it to zero gives $\hat\Sigma = \frac1N\sum_n(x_n-\mu)(x_n-\mu)^\top$, the MLE (with $1/N$, not $1/(N-1)$).

## About "independent entries"
$\Sigma$ is symmetric, so its entries aren't really independent. Treating them as independent gives the symmetric matrix above, which is the convention most references and autodiff tools use. If you parametrise only the upper triangle, the off-diagonal entries' gradients double.

## Why factorise
Cholesky costs $d^3/3$ once, gives $\log\det\Sigma$ for free from its diagonal, and is numerically stable for ill-conditioned covariances (test 6 spans three orders of magnitude). Computing $\det\Sigma$ directly overflows or underflows in high dimension, and its log is what you need anyway.

## Complexity
$O(d^3 + Nd^2)$.

## Common bugs
- Missing the $\tfrac12$ on the covariance gradient, or using $+N\Sigma^{-1}$.
- `np.log(np.linalg.det(Sigma))`: underflows to $-\infty$ for large $d$ with small variances.
- Mixing up $\Sigma^{-1}(x-\mu)$ as a row versus a column when batching: with rows as samples it's `D @ Sinv`.

## Interview follow-ups
- Gradient descent directly on $\Sigma$ can leave the positive definite cone. How does parametrising $\Sigma = LL^\top$ fix it, and what's $\nabla_L$?
- The reparametrisation trick writes $x = \mu + L\epsilon$. Why does that make $\nabla_\mu$ and $\nabla_L$ of $E[f(x)]$ easy to estimate?
"""
