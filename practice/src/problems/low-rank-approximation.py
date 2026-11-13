import numpy as np

META = dict(title='Best rank-k approximation', tags=['Linear Algebra'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:07.02'], related=['svd-components', 'matrix-rank', 'pca'])

STATEMENT = r"""
Among all matrices of rank at most $k$, the one closest to $A$ in Frobenius norm is the truncated SVD (Eckart–Young):

$$A_k = \sum_{i=1}^{k} \sigma_i\, u_i v_i^\top, \qquad \lVert A - A_k\rVert_F = \sqrt{\textstyle\sum_{i>k} \sigma_i^2}$$

This is the maths behind image compression, LoRA-style weight compression and latent factor models.

Write `low_rank(A, k)` that returns the approximation and how much it loses.

## Inputs
- `A`: float array `(m, n)`, $m, n \ge 1$.
- `k`: int, $k \ge 0$. It may exceed $\min(m, n)$, in which case $A_k = A$.

## Output
A tuple `(A_k, rel_err, n_params)`:
- `A_k`: float array `(m, n)`.
- `rel_err`: float, $\lVert A - A_k\rVert_F / \lVert A\rVert_F$, computed from the singular values. If $A$ is the zero matrix, `rel_err` is `0.0`.
- `n_params`: int, the number of floats needed to store the factored form with $k' = \min(k, m, n)$ layers: $k'(m + n + 1)$ (each layer stores $u_i$, $v_i$ and $\sigma_i$).

You may assume $\sigma_k > \sigma_{k+1}$ whenever $1 \le k < \min(m, n)$, so $A_k$ is unique.
"""

STARTER = '''import numpy as np

def low_rank(A, k):
    """A: (m, n), k >= 0. Return (A_k, relative Frobenius error, number of stored floats)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def low_rank(A, k):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(s))
    A_k = (U[:, :k] * s[:k]) @ Vt[:k]           # scale columns of U by sigma, then recombine
    total = np.sum(s ** 2)                      # ||A||_F^2
    # The tail energy is the discarded singular values; this avoids subtracting two nearly equal matrices.
    rel_err = float(np.sqrt(np.sum(s[k:] ** 2) / total)) if total > 0 else 0.0
    return A_k, rel_err, k * (m + n + 1)
'''

HINTS = [
    'Use `np.linalg.svd(A, full_matrices=False)` and keep the first `k` singular triplets. Clamp `k` to the number of singular values first.',
    '`(U[:, :k] * s[:k]) @ Vt[:k]` rebuilds $A_k$ without forming a diagonal matrix. For the error, you never need $A - A_k$: the discarded energy is $\\sum_{i>k}\\sigma_i^2$.',
    '`rel_err = sqrt(sum(s[k:]**2) / sum(s**2))`, guarding the zero matrix. `n_params = k * (m + n + 1)` with the clamped `k`.',
]

_r = np.random.default_rng(53)
TESTS = [
    (np.array([[3.0, 0.0], [0.0, 1.0]]), 1),
    (np.array([[2.0, 4.0], [1.0, 2.0], [0.0, 0.0]]), 1),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), 0),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), 5),
    (np.zeros((3, 2)), 1),
    (np.array([[1.0, 2.0, 3.0, 4.0]]), 1),
    (_r.normal(size=(6, 5)), 2),
    (_r.normal(size=(8, 3)) @ _r.normal(size=(3, 7)) + 0.01 * _r.normal(size=(8, 7)), 3),
]

EXPLANATION = r"""
## Why truncation is optimal
Write $A = \sum_i \sigma_i u_i v_i^\top$. The layers are orthogonal in the Frobenius inner product, so dropping layers $k+1, \dots$ costs exactly $\sum_{i>k}\sigma_i^2$. Eckart–Young says no rank-$k$ matrix does better: any rank-$k$ matrix must miss some direction in the span of $v_1, \dots, v_{k+1}$, on which $A$ stretches by at least $\sigma_{k+1}$.

Test 8 is a rank-3 matrix plus small noise: three layers capture almost all of it, so `rel_err` is tiny. Test 7 is pure noise: the singular values decay slowly and rank 2 leaves most of the energy behind.

## Storage
A full $m\times n$ matrix needs $mn$ floats; rank $k$ needs $k(m+n+1)$. For a $4096\times 4096$ weight at $k = 64$ that's $\approx 3\%$ of the original size.

## Complexity
$O(mn\min(m,n))$ for the SVD, $O(mnk)$ for the reconstruction. For $k \ll \min(m, n)$, randomised SVD or Lanczos get the top $k$ triplets much faster.

## Common bugs
- `np.diag(s) @ ...` with `full_matrices=True`: shapes don't match for non-square $A$.
- Computing the error as `norm(A - A_k)`: correct but loses precision when the error is tiny, and it costs another pass.
- Not clamping `k`, so `n_params` counts layers that don't exist.

## Interview follow-ups
- What is the best rank-$k$ approximation in spectral norm, and its error? (The same $A_k$; error $\sigma_{k+1}$.)
- How would you pick $k$ automatically? (Smallest $k$ whose retained energy $\sum_{i\le k}\sigma_i^2 / \sum_i\sigma_i^2$ passes a threshold, or look for the elbow in the spectrum.)
"""
