import numpy as np

META = dict(title='Sample covariance matrix from scratch', tags=['Probability and Statistics', 'Linear Algebra'], diff='Easy',
            runtime='numpy', related=['pca-projection', 'pearson-correlation-matrix', 'mean-and-variance'])

STATEMENT = r"""
Rows of `X` (shape `(n, d)`) are observations and columns are features. Write `covariance(X, ddof=1)` that returns the `(d, d)` covariance matrix

$$\Sigma = \frac{1}{n - \text{ddof}} (X - \bar{x})^\top (X - \bar{x}),$$

where $\bar{x}$ is the vector of column means.

## Requirements
- `ddof=1` gives the unbiased sample covariance; `ddof=0` gives the maximum-likelihood (population) version.
- Do not call `np.cov`. Use centring and one matrix product.
- If `n - ddof <= 0`, return a `(d, d)` array of `nan`.
"""

STARTER = '''import numpy as np

def covariance(X, ddof=1):
    """X: (n, d). Return the (d, d) covariance matrix."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def covariance(X, ddof=1):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n - ddof <= 0:
        return np.full((d, d), np.nan)
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (n - ddof)
'''

HINTS = [
    'Subtract the column means first; every entry of the covariance is an average of products of centred values.',
    '`Xc.T @ Xc` sums the products for every pair of features at once, giving a `(d, d)` matrix.',
    'Divide by `n - ddof`, and handle the case where that is zero or negative before dividing.',
]

_r = np.random.default_rng(11)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]),),
    (np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), 0),
    (np.array([[2.0, 3.0, 4.0]]),),
    (_r.normal(size=(50, 4)) @ np.array([[1, 0.5, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0.1]]),),
    (_r.normal(size=(7, 3)) + 100.0, 0),
    (np.array([[1.0], [2.0], [4.0], [8.0]]),),
]

EXPLANATION = r"""
## Why centre first
Covariance measures how two features move together around their means: $\mathrm{cov}(a, b) = \mathbb{E}[(a - \bar a)(b - \bar b)]$. After centring, the whole matrix is a Gram matrix of the columns, $X_c^\top X_c$, divided by the number of degrees of freedom. The diagonal holds the variances.

## Why $n - 1$
Using the sample mean instead of the true mean makes centred values slightly too small on average; dividing by $n - 1$ (Bessel's correction) removes that bias. Maximum likelihood under a Gaussian model divides by $n$.

## Numerical note
The one-pass formula $\frac{1}{n}\sum x x^\top - \bar x \bar x^\top$ loses precision badly when the mean is large compared with the spread (try the test with an offset of 100). Centring first avoids that cancellation.

## Complexity
$O(nd^2)$ time for the product, $O(nd)$ memory for the centred copy.

## Interview follow-ups
- How do you get the correlation matrix from the covariance? (Divide by the outer product of standard deviations.)
- Why is the covariance matrix positive semi-definite? ($v^\top \Sigma v$ is the variance of $Xv$.)
"""
