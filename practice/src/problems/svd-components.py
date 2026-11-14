import numpy as np

META = dict(title='SVD as a sum of rank-1 layers', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:07.07'], related=['low-rank-approximation', 'outer-product', 'matrix-rank'])

STATEMENT = r"""
The singular value decomposition writes any matrix as a weighted sum of rank-1 matrices:

$$A = U\Sigma V^\top = \sum_{i=1}^{r} \sigma_i\, u_i v_i^\top, \qquad \sigma_1 > \sigma_2 > \dots > \sigma_r > 0$$

The vectors $u_i$, $v_i$ are only defined up to a joint sign flip ($-u_i$, $-v_i$ gives the same term), but each **layer** $\sigma_i u_i v_i^\top$ is unique when the singular values are distinct. So the layers are a sign-free way to report an SVD.

Write `svd_layers(A)`.

## Input
- `A`: float array `(m, n)`, $m, n \ge 1$, not all zeros. You may assume its non-zero singular values are distinct.

## Output
A tuple `(s, layers)`:
- `s`: float array `(r,)` of the singular values above the cutoff, largest first.
- `layers`: float array `(r, m, n)` where `layers[i]` $= \sigma_i u_i v_i^\top$.

Keep $\sigma_i$ only if $\sigma_i > \max(m, n)\cdot\varepsilon\cdot\sigma_1$ with $\varepsilon$ = `np.finfo(float).eps`.
"""

STARTER = '''import numpy as np

def svd_layers(A):
    """A: (m, n). Return (s (r,), layers (r, m, n)) with layers[i] = s[i] * outer(u_i, v_i)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def svd_layers(A):
    A = np.asarray(A, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = int((s > max(A.shape) * np.finfo(float).eps * s[0]).sum())
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    # layers[i, j, k] = s[i] * U[j, i] * Vt[i, k]: a batch of scaled outer products.
    layers = np.einsum('i,ji,ik->ijk', s, U, Vt)
    return s, layers
'''

HINTS = [
    '`np.linalg.svd(A, full_matrices=False)` returns `U`, `s`, `Vt` with the singular values already sorted largest first. Note that $v_i$ is a **row** of `Vt`.',
    'Truncate to the $r$ singular values above the cutoff. Each layer is `s[i] * np.outer(U[:, i], Vt[i])`; the sign ambiguity cancels because $u_i$ and $v_i$ flip together.',
    'Without a loop: `np.einsum("i,ji,ik->ijk", s, U, Vt)`, or `s[:, None, None] * U.T[:, :, None] * Vt[:, None, :]`.',
]

_r = np.random.default_rng(47)
TESTS = [
    (np.array([[3.0, 0.0], [0.0, -2.0]]),),
    (np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),),
    (np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),),
    (np.array([[4.0]]),),
    (np.array([[1.0, -2.0, 2.0]]),),
    (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]),),
    (_r.normal(size=(4, 3)),),
    (_r.normal(size=(5, 2)) @ _r.normal(size=(2, 4)),),
]

EXPLANATION = r"""
## Why layers are sign-free
If $(u_i, v_i)$ is a pair of singular vectors, so is $(-u_i, -v_i)$, and different LAPACK builds may return either. The product $u_i v_i^\top$ is unchanged by the flip, so $\sigma_i u_i v_i^\top$ is the same everywhere. When two singular values are equal, the individual vectors are no longer unique even up to sign (any rotation within their shared subspace works), which is why the statement rules that out.

## Reading the layers
The layers are mutually orthogonal under the Frobenius inner product, so $\lVert A\rVert_F^2 = \sum_i \sigma_i^2$. Summing the first $k$ layers gives the best rank-$k$ approximation of $A$ (Eckart–Young). Test 2 is a rank-1 matrix: one layer reproduces it exactly. Test 8 is rank 2 by construction: two layers.

## Complexity
$O(mn\min(m,n))$ for the SVD, plus $O(rmn)$ to build the layers.

## Common bugs
- Taking columns of `Vt` instead of rows: `Vt[i]` is $v_i^\top$.
- Using `full_matrices=True`, then multiplying `U` of shape `(m, m)` against `Vt` of shape `(n, n)` and getting shape errors.
- Keeping the round-off singular values (~$10^{-16}$) of a rank-deficient matrix.

## Interview follow-ups
- How do the singular values relate to the eigenvalues of $A^\top A$? ($\sigma_i^2 = \lambda_i(A^\top A)$, and the $v_i$ are its eigenvectors.)
- What is $\lVert A\rVert_2$, the spectral norm? ($\sigma_1$: the most any unit vector gets stretched.)
"""
