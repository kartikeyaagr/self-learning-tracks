import numpy as np

META = dict(title='Pearson correlation matrix', tags=['Probability and Statistics', 'Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:11.21'], related=['correlation-significance-test', 'pca-projection', 'cosine-similarity-matrix'])

STATEMENT = r"""
The Pearson correlation between features $a$ and $b$ is their covariance divided by the product of their standard deviations:

$$r_{ab} = \frac{\sum_i (x_{ia} - \bar x_a)(x_{ib} - \bar x_b)}{\sqrt{\sum_i (x_{ia} - \bar x_a)^2}\,\sqrt{\sum_i (x_{ib} - \bar x_b)^2}}$$

Write `corr_matrix(X)` that returns the full $d \times d$ matrix $R$ for a data matrix whose **columns** are features.

## Requirements
- `X`: float array of shape `(n, d)` with $n \ge 2$. Return a float array of shape `(d, d)`.
- Diagonal entries are exactly `1.0`.
- A **constant** column has no defined correlation. Use `0.0` for every off-diagonal entry in its row and column (its diagonal entry is still `1.0`). `np.corrcoef` gives `nan` here, so it won't match.
- Use matrix products, not loops over pairs of columns.
"""

STARTER = '''import numpy as np

def corr_matrix(X):
    """X: (n, d), columns are features. Return the (d, d) correlation matrix."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def corr_matrix(X):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    norms = np.sqrt((Xc ** 2).sum(axis=0))
    safe = np.where(norms > 0, norms, 1.0)          # constant column: its centred values are all 0
    Z = Xc / safe                                   # unit-length centred columns (or a zero column)
    R = Z.T @ Z                                     # cosine similarity of centred columns
    np.fill_diagonal(R, 1.0)
    return R
'''

HINTS = [
    'Correlation is the cosine similarity of the **centred** columns. The $1/(n-1)$ factors cancel, so you never need them.',
    'Centre, scale each column to unit length, then one matrix product `Z.T @ Z` gives every pair at once.',
    'A constant column centres to all zeros. Divide it by 1 instead of 0: its products with everything are then 0. Finally overwrite the diagonal with 1.',
]

_r = np.random.default_rng(5)
_A = _r.normal(size=(40, 3))
TESTS = [
    (np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 1.0], [3.0, 6.0, 2.0]]),),
    (np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]),),
    (np.array([[0.0, 1.0], [1.0, 0.0]]),),
    (np.array([[1e8 + 1.0, -3.0], [1e8 + 2.0, -5.0], [1e8 + 4.0, -6.0]]),),
    (np.array([[2.0], [7.0], [1.0]]),),
    (np.column_stack([_A, _A[:, 0] * 2 + 0.1 * _r.normal(size=40)]),),
    (_r.normal(size=(15, 6)) * np.array([1.0, 10.0, 100.0, 0.1, 1.0, 3.0]),),
]

EXPLANATION = r"""
## Why it's a cosine
After centring, $r_{ab} = \frac{\tilde x_a \cdot \tilde x_b}{\lVert\tilde x_a\rVert\,\lVert\tilde x_b\rVert}$, the cosine of the angle between the two centred columns. That explains the range $[-1, 1]$ (Cauchy-Schwarz), why scaling or shifting a feature leaves $r$ unchanged, and why $|r| = 1$ means one column is an exact linear function of the other.

## Complexity
Centring is $O(nd)$ and the Gram matrix is $O(nd^2)$. Memory is $O(nd + d^2)$.

## Common bugs
- Forgetting to centre: you get cosine similarity, which depends on the mean.
- Normalising rows instead of columns, which gives correlations between samples.
- Leaving the diagonal as computed: it can come out as 0.9999999999999998, and a constant column gives 0 there.

## Interview follow-ups
- Pearson only sees linear relationships. What would you use for a monotone but curved one? (Spearman: Pearson on the ranks.)
- Two features have $r = 0.95$. Why might that hurt a linear regression, and what can you do? (Multicollinearity inflates coefficient variance. Drop one, combine them, or add ridge regularisation.)
"""
