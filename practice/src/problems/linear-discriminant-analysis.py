import numpy as np

META = dict(title='Linear discriminant analysis scores', tags=['Classification', 'Probability and Statistics'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:11.04', 'mlmaths:11.05'], related=['gaussian-naive-bayes', 'pca-projection'])

STATEMENT = r"""
LDA models every class as a Gaussian with its own mean $\mu_c$ but one **shared** covariance $\Sigma$. Taking logs of $\pi_c\,\mathcal N(x; \mu_c, \Sigma)$ and dropping terms that are the same for every class leaves a score that is linear in $x$:

$$\delta_c(x) = x^\top \Sigma^{-1}\mu_c - \tfrac12\,\mu_c^\top \Sigma^{-1}\mu_c + \log \pi_c .$$

The predicted class is the one with the largest score.

Write `lda_scores(X_train, y_train, X_test)` that returns `(classes, scores)`.

## Fitting
- `classes`: distinct labels of `y_train`, sorted ascending. $C$ is their count.
- $\pi_c$: fraction of training rows in class $c$. $\mu_c$: class mean.
- Pooled covariance, unbiased:
$$\Sigma = \frac{1}{n - C}\sum_{c}\ \sum_{i:\,y_i = c} (x_i - \mu_c)(x_i - \mu_c)^\top .$$

## Input / output
- `X_train`: float `(n, d)`; `y_train`: int `(n,)`; `X_test`: float `(m, d)`. The tests guarantee $n > C$ and an invertible $\Sigma$.
- `classes`: int array `(C,)`; `scores`: float array `(m, C)` with `scores[i, k]` $= \delta_{\text{classes}[k]}(x_i)$, exactly as defined above (no extra constants).
- Don't form $\Sigma^{-1}$ explicitly: solve a linear system.
"""

STARTER = '''import numpy as np

def lda_scores(X_train, y_train, X_test):
    """Return (classes ascending, discriminant scores (m, C)) with a pooled covariance."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def lda_scores(X_train, y_train, X_test):
    X, y, Xt = np.asarray(X_train, float), np.asarray(y_train), np.asarray(X_test, float)
    classes, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    n, C = len(y), len(classes)
    mu = np.array([X[inv == k].mean(axis=0) for k in range(C)])     # (C, d)
    R = X - mu[inv]                                                 # each row minus its own class mean
    S = R.T @ R / (n - C)                                           # pooled, unbiased
    A = np.linalg.solve(S, mu.T)                                    # Sigma^{-1} mu_c as columns, (d, C)
    return classes, Xt @ A - 0.5 * np.sum(mu.T * A, axis=0) + np.log(counts / n)
'''

HINTS = [
    'Get the class means first, then subtract each row\'s **own** class mean before forming the covariance. `np.unique(..., return_inverse=True)` gives you a class index per row.',
    'With residuals $R$ (rows $x_i - \\mu_{y_i}$), the pooled covariance is $R^\\top R/(n - C)$.',
    'Solve once for all classes: `A = np.linalg.solve(S, mu.T)` has columns $\\Sigma^{-1}\\mu_c$. Then the scores are `Xt @ A - 0.5 * (mu.T * A).sum(0) + log(prior)`.',
]

_r = np.random.default_rng(37)
_L = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [-0.3, 0.2, 0.9]])
_X = np.vstack([_r.normal(size=(20, 3)) @ _L.T + m for m in ([0, 0, 0], [2, 1, 0], [0, 3, -1])])
_y = np.repeat([0, 1, 4], 20)
TESTS = [
    (np.array([[0.0], [2.0], [4.0], [6.0]]), np.array([0, 0, 1, 1]), np.array([[3.0], [0.0]])),
    (np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [4.0, 4.0], [6.0, 4.0], [4.0, 6.0]]), np.array([0, 0, 0, 1, 1, 1]), np.array([[3.0, 3.0], [1.0, 1.0]])),
    (np.array([[0.0], [2.0], [4.0], [6.0], [8.0]]), np.array([0, 0, 0, 1, 1]), np.array([[4.7]])),
    (np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [-1.0, 0.0], [0.0, -2.0], [-2.0, -1.0], [5.0, -5.0], [6.0, -4.0], [7.0, -6.0]]),
     np.array([3, 3, 3, 1, 1, 1, 2, 2, 2]), np.array([[0.0, 0.0], [6.0, -5.0], [100.0, 100.0]])),
    (_X, _y, _X[::7]),
    (_X * 50.0, _y, _r.normal(scale=50.0, size=(6, 3))),
]

EXPLANATION = r"""
## Where the linear score comes from
$\log\big(\pi_c\,\mathcal N(x;\mu_c,\Sigma)\big) = \log\pi_c - \tfrac12 (x-\mu_c)^\top\Sigma^{-1}(x-\mu_c) + \text{const}$. Expanding the quadratic gives $-\tfrac12 x^\top\Sigma^{-1}x + x^\top\Sigma^{-1}\mu_c - \tfrac12\mu_c^\top\Sigma^{-1}\mu_c$. The first term is the same for every class because $\Sigma$ is shared, so it cancels in the comparison. What's left is linear in $x$, which is why decision boundaries are hyperplanes. With per-class covariances (QDA) the quadratic term survives and boundaries become conic sections.

## Why n - C
Each class mean uses up one degree of freedom, so the pooled scatter has $n - C$ of them. Dividing by $n - C$ makes $\Sigma$ unbiased, matching the classical two-sample pooled variance when $C = 2$.

## Complexity
$O(nd^2)$ for the covariance, $O(d^3 + d^2 C)$ for the solve, $O(mdC)$ for the scores.

## Common bugs
- Subtracting the global mean instead of each row's class mean, which inflates $\Sigma$ with the between-class spread.
- Using `np.cov` on all of `X` (wrong centring and wrong denominator).
- Forgetting the prior term, which matters when classes are unbalanced (test 3: the query is a little closer to class 1's mean, but class 0 has more samples).

## Interview follow-ups
- How does LDA give a supervised dimensionality reduction to at most $C - 1$ dimensions? (Directions maximising between-class over within-class scatter: a generalised eigenproblem.)
- LDA and logistic regression both produce linear boundaries. When does each one win? (LDA is more data-efficient when the Gaussian assumption holds; logistic regression is more robust when it doesn't.)
"""
