import numpy as np

META = dict(title='PCA and ZCA whitening', tags=['Feature Engineering', 'Linear Algebra'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:11.04', 'mlmaths:05.02'], related=['eigendecomposition', 'mahalanobis-distance', 'pca'])

STATEMENT = r"""
Whitening transforms data so its features are uncorrelated with unit variance: the sample covariance becomes $I$. With the eigendecomposition of the sample covariance $\Sigma = V\Lambda V^\top$ (`ddof=1`), two standard choices are

$$W_{\text{PCA}} = (\Lambda + \epsilon I)^{-1/2} V^\top, \qquad W_{\text{ZCA}} = V(\Lambda + \epsilon I)^{-1/2} V^\top$$

and the whitened data is $Z = (X - \bar x)W^\top$. PCA whitening rotates onto the principal axes and rescales. ZCA rotates back afterwards, so the result stays as close as possible to the original features (useful for images).

Write `whiten(X, method, eps=0.0)`.

## Inputs
- `X`: float array `(n, d)`, $n > d \ge 1$, with full-rank sample covariance and **distinct** eigenvalues.
- `method`: `'pca'` or `'zca'`.
- `eps`: float $\ge 0$, added to every eigenvalue before the inverse square root (regularisation).

## Conventions for `'pca'`
- Rows of $W$ are ordered by eigenvalue, **largest first**.
- Each row of $W$ is sign-fixed so its entry with the largest absolute value is positive.

($W_{\text{ZCA}}$ is symmetric and sign-free.)

## Output
A tuple `(W, Z)`: `W` is `(d, d)` and `Z` is `(n, d)`. With `eps=0`, `np.cov(Z, rowvar=False)` is the identity.
"""

STARTER = '''import numpy as np

def whiten(X, method, eps=0.0):
    """X: (n, d). method: 'pca' or 'zca'. Return (W (d, d), Z = (X - mean) @ W.T)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def whiten(X, method, eps=0.0):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (len(X) - 1)
    vals, V = np.linalg.eigh(cov)                       # ascending
    vals, V = vals[::-1], V[:, ::-1]                    # descending; columns are eigenvectors
    scale = 1.0 / np.sqrt(vals + eps)
    W = scale[:, None] * V.T                            # PCA whitening: rows = scaled eigenvectors
    if method == 'pca':
        idx = np.abs(W).argmax(axis=1)
        W = W * np.sign(W[np.arange(len(W)), idx])[:, None]
    else:
        W = V @ W                                       # ZCA: rotate back, V diag(scale) V^T (signs cancel)
    return W, Xc @ W.T
'''

HINTS = [
    'Both methods start the same way: centre, compute the `ddof=1` covariance, and eigendecompose it with `np.linalg.eigh`. Reverse to get descending order.',
    '$W_{\\text{PCA}}$ is $V^\\top$ with row $i$ scaled by $1/\\sqrt{\\lambda_i + \\epsilon}$. $W_{\\text{ZCA}}$ is $V$ times that. Signs of eigenvectors cancel in ZCA ($v_i v_i^\\top$), but not in PCA, so fix them there only.',
    'For PCA signs: `idx = np.abs(W).argmax(axis=1)` then `W *= np.sign(W[np.arange(d), idx])[:, None]`. Return `(W, Xc @ W.T)`.',
]

_r = np.random.default_rng(97)
_A = np.array([[2.0, 0.0, 0.0], [1.2, 0.8, 0.0], [-0.6, 0.3, 0.5]])
_X = _r.normal(size=(60, 3)) @ _A.T + np.array([10.0, -5.0, 0.0])
_P = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 3.0], [1.0, 2.0], [3.0, 0.0], [5.0, 4.0]])
TESTS = [
    (np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 3.0], [0.0, -3.0], [0.0, 0.0]]), 'pca'),
    (np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 3.0], [0.0, -3.0], [0.0, 0.0]]), 'zca'),
    (_P, 'pca'),
    (_P, 'zca'),
    (_P, 'zca', 0.5),
    (_X, 'pca'),
    (_X, 'zca'),
    (_X, 'pca', 1e-2),
    (np.array([[1.0], [2.0], [4.0], [7.0]]), 'pca'),
]

EXPLANATION = r"""
## Why the covariance becomes $I$
$\operatorname{cov}(Z) = W\Sigma W^\top$. For PCA, $W\Sigma W^\top = \Lambda^{-1/2}V^\top(V\Lambda V^\top)V\Lambda^{-1/2} = I$. ZCA just left-multiplies by the orthogonal $V$, which keeps the covariance at $V I V^\top = I$. In fact *any* $RW_{\text{PCA}}$ with orthogonal $R$ whitens; ZCA is the choice that minimises $\lVert Z - X_c\rVert_F$, so its features still look like the originals.

With $\epsilon > 0$ the covariance becomes $\operatorname{diag}(\lambda_i/(\lambda_i + \epsilon))$: low-variance directions, often noise, are amplified less.

## Sign handling
$W_{\text{ZCA}} = \sum_i \lambda_i^{-1/2} v_i v_i^\top$ is unchanged if any $v_i$ flips. $W_{\text{PCA}}$'s rows are the $v_i$ themselves, so the sign convention is needed. Test 1's covariance is diagonal, so its eigenvectors are the axes, swapped into descending variance order.

## Complexity
$O(nd^2)$ for the covariance, $O(d^3)$ for the eigendecomposition, $O(nd^2)$ to transform.

## Common bugs
- `ddof=0`: the whitened covariance comes out as $\frac{n-1}{n}I$.
- Scaling columns instead of rows of $V^\top$: `V.T * scale` multiplies column $j$ by `scale[j]`, which mixes up the eigenvalues.
- Applying $W$ instead of $W^\top$ on the right ($Z = X_c W^\top$, because rows of $X_c$ are samples).
- Forgetting to centre with the mean.

## Interview follow-ups
- How is whitening related to Mahalanobis distance? (Euclidean distance between whitened points is the Mahalanobis distance between the originals.)
- Why does batch norm not whiten? (Full whitening needs a $d\times d$ decomposition per batch and is unstable to backpropagate through; BN only standardises each feature, dropping the correlations.)
"""
