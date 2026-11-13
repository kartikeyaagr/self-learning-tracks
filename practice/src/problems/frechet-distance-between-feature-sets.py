import numpy as np

META = dict(title='Frechet distance between two feature sets (FID)', tags=['Generative Models', 'Metrics & Evaluation', 'Linear Algebra'],
            diff='Medium', runtime='numpy', lessons=['ieng:26.09'], related=['diagonal-gaussian-kl-divergence'])

STATEMENT = r"""
FID compares generated images with real ones by fitting a Gaussian to each set of network features and measuring the **Frechet (2-Wasserstein) distance** between the two Gaussians:

$$d^2 = \lVert \mu_1 - \mu_2\rVert^2 + \operatorname{Tr}\!\left(\Sigma_1 + \Sigma_2 - 2\,(\Sigma_1\Sigma_2)^{1/2}\right)$$

$\Sigma_1\Sigma_2$ is not symmetric, so computing its square root directly is awkward and numerically fragile. Use the identity

$$\operatorname{Tr}(\Sigma_1\Sigma_2)^{1/2} = \operatorname{Tr}\left(\Sigma_1^{1/2}\,\Sigma_2\,\Sigma_1^{1/2}\right)^{1/2} = \sum_i \sqrt{\lambda_i}$$

where $\lambda_i$ are the eigenvalues of the **symmetric** PSD matrix $M = \Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}$.

Write `frechet_distance(feats1, feats2)` and return $d^2$ as a Python `float`.

## Inputs
`feats1`: `(N1, D)`, `feats2`: `(N2, D)`, with $N_1, N_2 \ge 2$. Rows are samples.

## Conventions
- Means are column means. Covariances are **unbiased** (divide by $N - 1$), as `np.cov(feats, rowvar=False)` computes.
- Compute $\Sigma_1^{1/2}$ from a symmetric eigendecomposition, clipping negative eigenvalues (rounding noise) to $0$ first. Clip the eigenvalues of $M$ at $0$ too before the square root.
- For $D = 1$, the covariances are $1\times1$ matrices; your code should not special-case it.
"""

STARTER = '''import numpy as np

def frechet_distance(feats1, feats2):
    """Squared Frechet distance between Gaussians fitted to two (N, D) feature sets."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def psd_sqrt(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T      # V diag(sqrt w) V^T

def frechet_distance(feats1, feats2):
    f1, f2 = np.asarray(feats1, dtype=float), np.asarray(feats2, dtype=float)
    mu1, mu2 = f1.mean(axis=0), f2.mean(axis=0)
    S1 = np.atleast_2d(np.cov(f1, rowvar=False))
    S2 = np.atleast_2d(np.cov(f2, rowvar=False))
    R = psd_sqrt(S1)
    M = R @ S2 @ R
    M = (M + M.T) / 2                                      # symmetrise away rounding asymmetry
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh(M), 0.0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2.0 * tr_sqrt)
'''

HINTS = [
    'Split the formula into a mean term and a trace term. $\\operatorname{Tr}(\\Sigma_1 + \\Sigma_2)$ is easy; only $\\operatorname{Tr}(\\Sigma_1\\Sigma_2)^{1/2}$ needs work.',
    'A symmetric PSD square root is $V\\,\\mathrm{diag}(\\sqrt{w})\\,V^\\top$ from `np.linalg.eigh`. Use it for $\\Sigma_1^{1/2}$, form $M = \\Sigma_1^{1/2}\\Sigma_2\\Sigma_1^{1/2}$, and sum the square roots of `eigvalsh(M)`.',
    '`np.cov(f, rowvar=False)` returns a 0-d array when `D == 1`; wrap it in `np.atleast_2d`. Clip eigenvalues at 0 before every square root.',
]

_r = np.random.default_rng(73)
_A = _r.normal(size=(200, 4))
_B = _r.normal(size=(150, 4)) @ np.diag([1.0, 2.0, 0.5, 1.0]) + np.array([0.5, 0.0, -0.5, 1.0])
TESTS = [
    (np.array([[0.0], [2.0]]), np.array([[1.0], [3.0]])),
    (np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]), np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]])),
    (_A, _A.copy()),
    (_A, _B),
    (_A[:50], _A[50:120] * 1.5),
    (np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])),
    (_r.normal(size=(3, 5)), _r.normal(size=(4, 5)) + 2),
]

EXPLANATION = r"""
## Why the identity holds
$\Sigma_1\Sigma_2 = \Sigma_1^{1/2}\,(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})\,\Sigma_1^{-1/2}$ is similar to $M$, so the two have the same eigenvalues, and the trace of a matrix square root is the sum of the square roots of its eigenvalues. $M$ is symmetric PSD, so `eigvalsh` is stable and its eigenvalues are real and (up to rounding) non-negative. This avoids a general `sqrtm`, which can return complex values with tiny imaginary parts.

Test 3 compares a set with itself: $d^2 = 0$ up to rounding. Test 6 has a degenerate set (all rows identical, $\Sigma_1 = 0$), and test 7 has fewer samples than dimensions, so both covariances are rank-deficient: clipping matters.

## Complexity
$O((N_1 + N_2)D^2 + D^3)$: covariances plus two eigendecompositions. With Inception features $D = 2048$, the $D^3$ term is a few seconds.

## Common bugs
- Biased covariances (`ddof=0`): a different number from the standard FID.
- `np.sqrt(np.linalg.eigvals(S1 @ S2))`: complex or `nan` from rounding.
- Treating rows as variables (`np.cov(feats)` without `rowvar=False`).

## Interview follow-ups
- FID is biased for small sample sizes. Why, and how does Kernel Inception Distance avoid it?
- Why might a model with better FID still produce worse samples for a particular use?
"""
