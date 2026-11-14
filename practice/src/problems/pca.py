import numpy as np

META = dict(title='PCA by SVD with a variance target', tags=['Linear Algebra', 'Feature Engineering'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:08.06', 'mlmaths:07.07'], related=['pca-projection', 'low-rank-approximation', 'whitening-transform'])

STATEMENT = r"""
PCA keeps the few directions that explain most of the variance. Instead of fixing the number of components, practitioners often ask for "enough components to explain 95% of the variance". Implement that with an SVD of the centred data, and return the compressed-then-restored data.

Write `pca_reconstruct(X, target)`.

## Steps
1. Centre: $X_c = X - \bar x$ (column means).
2. SVD: $X_c = U S V^\top$. The variance explained by component $i$ is proportional to $s_i^2$.
3. Choose the **smallest** $k \ge 1$ whose cumulative ratio $\sum_{i\le k} s_i^2 \big/ \sum_i s_i^2$ is $\ge$ `target`.
4. Reconstruct from the top $k$ components: $\hat X = X_c V_k V_k^\top + \bar x$, where $V_k$ holds the first $k$ right singular vectors as columns.

## Inputs
- `X`: float array `(n, d)` with $n \ge 2$ and not all rows identical.
- `target`: float in $(0, 1]$. The tests never place a cumulative ratio within $10^{-6}$ of `target`.

## Output
A tuple `(k, X_hat, ratio)`:
- `k`: int.
- `X_hat`: float array `(n, d)`.
- `ratio`: float, the cumulative explained-variance ratio of the first `k` components.

All three are sign-independent, so no sign convention is needed.
"""

STARTER = '''import numpy as np

def pca_reconstruct(X, target):
    """X: (n, d). Return (k, X_hat (n, d), cumulative ratio) for the smallest k explaining >= target variance."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pca_reconstruct(X, target):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    k = int(np.searchsorted(cum, target) + 1)         # first index with cum >= target, as a count
    k = min(k, len(s))                                # guard target=1.0 against round-off in cum[-1]
    Vk = Vt[:k].T                                     # (d, k) top right singular vectors as columns
    X_hat = Xc @ Vk @ Vk.T + mean                     # project onto the k-dim subspace, then un-centre
    return k, X_hat, float(cum[k - 1])
'''

HINTS = [
    'You never need the covariance matrix: the right singular vectors of the centred data are the principal directions, and $s_i^2/(n-1)$ are the variances.',
    '`cum = np.cumsum(s**2) / np.sum(s**2)` is non-decreasing, so `np.searchsorted(cum, target)` finds the first position where it reaches `target`. Add 1 to turn the position into a count.',
    'With `Vk = Vt[:k].T`, `X_hat = Xc @ Vk @ Vk.T + mean`. The sign of each column of `Vk` cancels in $V_k V_k^\\top$. Clamp `k` to `len(s)` so `target=1.0` is safe.',
]

_r = np.random.default_rng(83)
_Z = _r.normal(size=(50, 2)) * np.array([5.0, 2.0])
_Xlow = _Z @ _r.normal(size=(2, 6)) + 0.05 * _r.normal(size=(50, 6)) + 3.0
TESTS = [
    (np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]), 0.9),
    (np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), 0.5),
    (np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), 0.9),
    (np.array([[0.0], [1.0], [5.0]]), 0.3),
    (_Xlow, 0.95),
    (_Xlow, 0.5),
    (_r.normal(size=(20, 5)) * np.array([4.0, 3.0, 2.0, 1.0, 0.5]) + 100.0, 0.97),
]

EXPLANATION = r"""
## Why the SVD gives PCA
The sample covariance is $\Sigma = X_c^\top X_c/(n-1) = V\frac{S^2}{n-1}V^\top$. So the right singular vectors are the covariance eigenvectors, and the variance along component $i$ is $s_i^2/(n-1)$; the $n-1$ cancels in the ratio. The SVD route never forms $X_c^\top X_c$, which would square the condition number.

## Why the reconstruction is sign-free
$V_k V_k^\top$ is the orthogonal projection onto the top-$k$ subspace. Flipping any column of $V_k$ leaves it unchanged, so $\hat X$, $k$ and the ratio are well-defined even though individual components are not. That's why this problem needs no sign rule, unlike one that returns the scores $X_c V_k$.

Test 1 lies exactly on a line, so one component explains 100%. Tests 2 and 3 have variances 8/3 and 2/3 along the axes: one component gives 80%, enough for 0.5 but not 0.9. Test 5 is a noisy rank-2 cloud: two components restore it almost perfectly.

## Complexity
$O(nd\min(n,d))$ for the SVD, $O(ndk)$ for the reconstruction.

## Common bugs
- Not centring, so the first component points at the mean.
- Forgetting to add the mean back to $\hat X$.
- `np.searchsorted` without `+ 1`: it returns a 0-based position, not a count.
- Ratios from $s_i$ instead of $s_i^2$.

## Interview follow-ups
- What does the reconstruction error measure, and how is it used for anomaly detection? (The distance from each point to the principal subspace; large errors flag points that don't fit the low-rank structure.)
- When would you use kernel PCA or an autoencoder instead? (When the data lies near a curved manifold that no linear subspace captures.)
"""
