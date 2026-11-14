import numpy as np

META = dict(title='Polynomial features up to a given degree', tags=['Feature Engineering', 'Regression'], diff='Easy', runtime='numpy',
            related=['interaction-features', 'linear-regression', 'ridge-regression'])

STATEMENT = r"""
Write `poly_features(X, degree, include_bias=True)` that expands each row of `X` (shape `(n, d)`) into all monomials of total degree at most `degree`.

## Column order
- If `include_bias`, the first column is all ones (degree 0).
- Then degree 1, degree 2, and so on. Within a degree, use the order of `itertools.combinations_with_replacement(range(d), deg)`: for $d = 2$, degree 2 gives $x_0^2, x_0x_1, x_1^2$.

Return a float array of shape `(n, number_of_monomials)`.
"""

STARTER = '''import numpy as np

def poly_features(X, degree, include_bias=True):
    """Return the polynomial feature matrix."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from itertools import combinations_with_replacement

def poly_features(X, degree, include_bias=True):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    cols = [np.ones(n)] if include_bias else []
    for deg in range(1, degree + 1):
        for idx in combinations_with_replacement(range(d), deg):
            cols.append(np.prod(X[:, list(idx)], axis=1))
    return np.column_stack(cols) if cols else np.empty((n, 0))
'''

HINTS = [
    '`combinations_with_replacement(range(d), deg)` yields index tuples like (0, 0) or (0, 1); each tuple is one monomial.',
    'A monomial is the product of the selected columns: `np.prod(X[:, list(idx)], axis=1)`.',
    'The number of features grows as C(d + degree, degree), so keep degree small for many inputs.',
]

_r = np.random.default_rng(4)
TESTS = [
    (np.array([[2.0, 3.0]]), 2),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), 2, False),
    (np.array([[2.0], [3.0]]), 3),
    (_r.normal(size=(4, 3)), 2),
    (_r.normal(size=(3, 2)), 3, False),
    (np.array([[1.0, -1.0, 2.0]]), 1),
]

EXPLANATION = r"""
## Why it works
A linear model on polynomial features fits a polynomial in the original inputs: the model is still linear in its weights, so least squares and its closed form still apply.

## Growth
With $d$ inputs and degree $p$ there are $\binom{d+p}{p}$ features: 3 inputs at degree 2 give 10, but 100 inputs at degree 3 give 176,851. Regularisation (ridge) and feature scaling become important, since high powers of large values dominate.

## Interview follow-ups
- How does the kernel trick avoid building these features explicitly?
- Why scale inputs before expanding them?
"""
