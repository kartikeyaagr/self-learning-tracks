TITLE = 'Gaussian log-likelihood that survives high dimensions'
MODULES = ['10', '11', '12']

STATEMENT = r'''
Write `mvn_logpdf(X, mu, Sigma)` returning the log-density of $\mathcal{N}(\mu, \Sigma)$ at each row of `X` (shape `(N, d)`), as an array of shape `(N,)`.

Spec:
- Use a Cholesky factorisation. Raise `np.linalg.LinAlgError` if `Sigma` is not positive definite.
- Accurate for $d = 300$ with small variances, where $\det\Sigma$ underflows to 0.
- `N == 0` returns an empty array.

Allowed: `np.linalg.cholesky`, `scipy.linalg.solve_triangular`. Forbidden: `multivariate_normal`, `inv`, `det`, `slogdet`, `pinv`, `solve`, `lstsq`, `eigh`, `eig`. No Python loops.
'''

STARTER = '''def mvn_logpdf(X, mu, Sigma):
    # YOUR CODE
    raise NotImplementedError
'''

CHECK = r'''def _check():
    static_rules([mvn_logpdf], ["multivariate_normal", "inv", "det", "slogdet", "pinv", "solve", "lstsq", "eigh", "eig"], 0)
    banned = ("numpy.linalg.inv", "numpy.linalg.det", "numpy.linalg.slogdet", "numpy.linalg.solve", "numpy.linalg.eigh", "scipy.linalg.inv", "scipy.linalg.det")
    rng = np.random.default_rng(8)
    def spd(d, scale):
        A = rng.normal(size=(d, d)); return scale * (A @ A.T / d + 0.5 * np.eye(d))
    for name, d, scale, N in (("d=4", 4, 1.0, 50), ("d=1", 1, 2.0, 10), ("d=300 small variances", 300, 1e-3, 20)):
        S = spd(d, scale); mu = rng.normal(size=d); X = mu + rng.normal(size=(N, d)) @ np.linalg.cholesky(S).T
        ref = scipy.stats.multivariate_normal(mu, S).logpdf(X).reshape(N)
        with blocked(*banned): got = np.asarray(mvn_logpdf(X, mu, S))
        expect(got.shape == (N,), f"[{name}] shape {got.shape}, expected {(N,)}")
        expect(np.all(np.isfinite(got)), f"[{name}] non-finite values (det underflow?)")
        expect(np.allclose(got, ref, rtol=1e-8, atol=1e-6), f"[{name}] max abs error {np.abs(got - ref).max():.2e} vs scipy")
    with blocked(*banned): e = np.asarray(mvn_logpdf(np.zeros((0, 3)), np.zeros(3), np.eye(3)))
    expect(e.shape == (0,), f"N=0 should return shape (0,), got {e.shape}")
    try:
        with blocked(*banned): mvn_logpdf(np.zeros((2, 2)), np.zeros(2), np.array([[1.0, 2.0], [2.0, 1.0]]))
    except np.linalg.LinAlgError:
        pass
    else:
        raise CheckFail("indefinite Sigma did not raise LinAlgError")
'''

SOLUTION = '''def mvn_logpdf(X, mu, Sigma):
    X = np.asarray(X, dtype=float)
    d = len(mu)
    L = np.linalg.cholesky(Sigma)                     # Sigma = L L^T; raises LinAlgError unless positive definite
    W = scipy.linalg.solve_triangular(L, (X - mu).T, lower=True)   # (d, N): columns are L^{-1}(x - mu)
    maha = (W ** 2).sum(axis=0)                       # squared Mahalanobis distances
    logdet = 2.0 * np.log(np.diag(L)).sum()           # log det Sigma without ever forming det (no underflow)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
'''

MCQ = [
    {"q": r"For i.i.d. data $x_1, \dots, x_n \sim \mathcal{N}(\mu, \Sigma)$, the MLE is $\hat\mu = \bar x$ and $\hat\Sigma = \frac{1}{n}\sum_i (x_i - \bar x)(x_i - \bar x)^\top$. Is $\hat\Sigma$ unbiased, and if not, how far off is it for $n = 5$?",
     "options": [r"Unbiased: maximum-likelihood estimators are always unbiased.",
                 r"Biased: $\mathbb{E}[\hat\Sigma] = \frac{5}{4}\Sigma$, an overestimate by a factor 1.25.",
                 r"Biased: $\mathbb{E}[\hat\Sigma] = \frac{4}{5}\Sigma$, an underestimate by a factor 0.8, because $\bar x$ is fitted to the same data.",
                 r"Biased: $\mathbb{E}[\hat\Sigma] = \frac{1}{5}\Sigma$, because it divides by $n$."],
     "answer": 2,
     "why": r"$\mathbb{E}[\hat\Sigma] = \frac{n-1}{n}\Sigma$: deviations from the sample mean are systematically smaller than deviations from the true mean. Dividing by $n - 1$ removes the bias."},
    {"q": r"You fit a Gaussian to 64-dimensional features using $n = 40$ samples (sample mean and covariance), then call `mvn_logpdf` on new points. What happens?",
     "options": [r"It works: 40 samples are enough as long as no two are identical.",
                 r"The sample covariance has rank at most 39 < 64, so it is singular: Cholesky raises `LinAlgError` (or, with rounding, gives absurd log-densities). Fix with shrinkage such as $\hat\Sigma + \lambda I$ or Ledoit-Wolf, or a diagonal covariance.",
                 r"The covariance is full rank but ill-conditioned; standardising each feature to unit variance fixes it.",
                 r"The covariance has rank 40, so you only need 24 more samples; until then drop the 24 lowest-variance features."],
     "answer": 1,
     "why": r"Centred samples span at most $n - 1 = 39$ dimensions, so $\hat\Sigma$ has at least 25 zero eigenvalues. Standardising rescales features but cannot add rank; regularising the covariance makes it positive definite."},
]
