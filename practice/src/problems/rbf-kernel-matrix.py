import numpy as np

META = dict(title='RBF kernel ridge regression', tags=['Classic ML', 'Regression', 'Linear Algebra'], diff='Hard', runtime='numpy',
            related=['euclidean-distance', 'cholesky-decomposition', 'least-squares-solution'])

STATEMENT = r"""
The Gaussian (RBF) kernel scores how similar two points are:

$$k(x, x') = \exp\!\big(-\gamma\,\lVert x - x'\rVert^2\big)$$

Kernel ridge regression fits $f(x) = \sum_i \alpha_i\, k(x, x_i)$ by solving one linear system with the training kernel matrix $K_{ij} = k(x_i, x_j)$:

$$(K + \lambda I)\,\alpha = y$$

Write `kernel_ridge(X, y, X_test, lam, gamma=None)` that builds the kernels, fits $\alpha$ and predicts.

## Bandwidth
If `gamma` is `None`, use the **median heuristic**: let $m$ be the median of the squared distances $\lVert x_i - x_j\rVert^2$ over all pairs $i < j$ of training points (`np.median`, so an even count averages the two middle values), and set $\gamma = 1/m$. You may assume $m > 0$.

## Inputs
- `X`: float `(n, d)` training inputs ($n \ge 2$), `y`: float `(n,)`, `X_test`: float `(t, d)` with $t \ge 1$.
- `lam`: float $> 0$. `gamma`: positive float or `None`.

## Output
A tuple `(K, alpha, y_pred, gamma)`:
- `K`: `(n, n)` training kernel; its diagonal is exactly 1.
- `alpha`: `(n,)` solution of $(K + \lambda I)\alpha = y$.
- `y_pred`: `(t,)` predictions $K_{\text{test}}\,\alpha$, where $(K_{\text{test}})_{ij} = k(x^{\text{test}}_i, x_j)$.
- `gamma`: the $\gamma$ used, as a Python float.

## Requirements
Compute squared distances without Python loops. Clamp them at 0, since the expansion $\lVert a\rVert^2 + \lVert b\rVert^2 - 2a\cdot b$ can round slightly negative. Don't use `np.linalg.inv`.
"""

STARTER = '''import numpy as np

def kernel_ridge(X, y, X_test, lam, gamma=None):
    """Return (K (n, n), alpha (n,), y_pred (t,), gamma) for RBF kernel ridge regression."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sq_dists(A, B):
    # ||a||^2 + ||b||^2 - 2 a.b for every pair, in O(|A||B|) memory; clamp round-off below zero.
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)

def kernel_ridge(X, y, X_test, lam, gamma=None):
    X, y, X_test = (np.asarray(a, dtype=float) for a in (X, y, X_test))
    D = _sq_dists(X, X)
    np.fill_diagonal(D, 0.0)                              # exact zeros, so the diagonal of K is exactly 1
    if gamma is None:
        iu = np.triu_indices(len(X), k=1)                 # each unordered pair once, i < j
        gamma = 1.0 / np.median(D[iu])
    K = np.exp(-gamma * D)
    # K is symmetric PSD and lam > 0, so K + lam I is SPD: a Cholesky-based solve would also work.
    alpha = np.linalg.solve(K + lam * np.eye(len(X)), y)
    y_pred = np.exp(-gamma * _sq_dists(X_test, X)) @ alpha
    return K, alpha, y_pred, float(gamma)
'''

HINTS = [
    'Write one helper that returns all squared distances between rows of two matrices; you need it for train-train and test-train.',
    'Squared distances: `(A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T`, then `np.maximum(..., 0)`. For the median heuristic use `np.triu_indices(n, k=1)` so each pair counts once and the zero diagonal is excluded.',
    'Set the train-train diagonal to exactly 0 before exponentiating. Then `alpha = np.linalg.solve(K + lam * np.eye(n), y)` and `y_pred = exp(-gamma * D_test) @ alpha`.',
]

_r = np.random.default_rng(101)
_X = _r.uniform(-3, 3, size=(25, 1))
_y = np.sin(_X[:, 0]) + 0.1 * _r.normal(size=25)
_X2 = _r.normal(size=(30, 3))
_y2 = np.cos(_X2[:, 0]) * _X2[:, 1] + 0.05 * _r.normal(size=30)
TESTS = [
    (np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 0.0]), np.array([[1.0], [0.5]]), 0.1, 1.0),
    (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]), np.array([1.0, 2.0, 3.0]), np.array([[0.0, 0.0]]), 1.0),
    (np.array([[0.0], [3.0]]), np.array([1.0, -1.0]), np.array([[0.0], [3.0], [100.0]]), 1e-3, 0.5),
    (np.array([[1.0], [2.0], [4.0], [8.0]]), np.array([1.0, 1.0, 1.0, 1.0]), np.array([[3.0]]), 0.5),
    (_X, _y, np.linspace(-3, 3, 7)[:, None], 0.01, 2.0),
    (_X, _y, np.linspace(-3, 3, 7)[:, None], 0.1),
    (_X2, _y2, _r.normal(size=(5, 3)), 0.05),
    (_X2 * 1e3, _y2, _r.normal(size=(4, 3)) * 1e3, 0.05),
]

EXPLANATION = r"""
## Why it works
Kernel ridge is ridge regression in the feature space $\phi$ with $k(x, x') = \phi(x)^\top\phi(x')$. The representer theorem says the minimiser of $\sum_i (f(x_i) - y_i)^2 + \lambda\lVert f\rVert^2$ is a combination of kernels centred on the training points, and substituting $f = \sum_i\alpha_i k(\cdot, x_i)$ turns the problem into $(K + \lambda I)\alpha = y$. The RBF kernel's $\phi$ is infinite-dimensional, but you never compute it: only the $n\times n$ matrix $K$.

## The median heuristic
$\gamma$ sets the length scale. Too large and $K \approx I$: each point only "sees" itself and predictions away from the data fall to 0 (test 3's far-away query). Too small and $K$ is nearly all ones, which underfits. $\gamma = 1/\text{median}(\lVert x_i - x_j\rVert^2)$ makes a typical pair have kernel value $e^{-1}$, and it rescales automatically: test 8 uses test 7's training inputs multiplied by $10^3$, and gets exactly the same `K` and `alpha`.

## Complexity
$O(n^2 d)$ to build $K$, $O(n^3)$ for the solve, $O(tnd)$ to predict. The cubic solve is why kernel methods stop scaling around $n \approx 10^4$–$10^5$; Nyström or random Fourier features approximate $K$ with low rank.

## Common bugs
- Taking the median over the full matrix, including the zero diagonal and each pair twice.
- Forgetting to clamp: tiny negative squared distances are harmless in `exp`, but they break `sqrt` in other kernels, and the diagonal may come out as `0.99999999`.
- Using $\exp(-\lVert x - x'\rVert^2 / \gamma)$ when the statement multiplies by $\gamma$.

## Interview follow-ups
- Why is $K + \lambda I$ always invertible here? (The RBF kernel matrix is positive semi-definite, so adding $\lambda > 0$ makes every eigenvalue at least $\lambda$.)
- How does an RBF SVM differ from kernel ridge? (Hinge loss instead of squared loss: the solution is sparse in $\alpha$, with only support vectors non-zero, but needs a QP solver.)
"""
