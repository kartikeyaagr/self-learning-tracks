TITLE = 'PCA from scratch'
MODULES = ['05', '06', '07', '08']

STATEMENT = r'''
Write `pca(X, k)` for a data matrix `X` of shape `(n, d)` (rows are samples). Return `(components, explained_variance, Z)`:
- `components`: `(k, d)`, orthonormal rows, ordered by decreasing variance.
- `explained_variance`: `(k,)`, the variance along each component, using the sample covariance (divide by `n - 1`).
- `Z`: `(n, k)`, the scores (projections of the centred data).
- Sign convention: in each component, the entry with the largest absolute value is positive.
- Raise `ValueError` unless `1 <= k <= min(n, d)`.

Forbidden: `svd`, `PCA`, `sklearn`, `TruncatedSVD`. You may use `np.linalg.eigh`. No Python loops.
'''

STARTER = '''def pca(X, k):
    # YOUR CODE
    raise NotImplementedError
'''

CHECK = r'''def _ref_pca(X, k):
    X = np.asarray(X, float); Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k]; s = np.sign(V[np.arange(k), np.abs(V).argmax(1)]); V = V * s[:, None]
    return V, S[:k] ** 2 / (len(X) - 1), Xc @ V.T

def _check():
    static_rules([pca], ["svd", "PCA", "sklearn", "TruncatedSVD"], 0)
    banned = ("numpy.linalg.svd", "scipy.linalg.svd")
    rng = np.random.default_rng(4)
    def make(n, d, offset):
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        return rng.normal(size=(n, d)) * np.linspace(6, 0.5, d) @ Q.T + offset
    cases = [("n=500 d=10 k=3, mean offset 1e3", make(500, 10, 1e3), 3),
             ("d > n: n=20 d=50 k=5", make(20, 50, 0.0), 5),
             ("k = d", make(300, 6, -7.0), 6),
             ("float32 input", make(400, 8, 2.0).astype(np.float32), 4)]
    for name, X, k in cases:
        Vr, er, Zr = _ref_pca(X.astype(float), k)
        X0 = X.copy()
        with blocked(*banned): V, ev, Z = pca(X, k)
        expect(np.array_equal(X, X0), f"[{name}] X was modified in place")
        V, ev, Z = map(np.asarray, (V, ev, Z))
        expect(V.shape == (k, X.shape[1]) and ev.shape == (k,) and Z.shape == (X.shape[0], k), f"[{name}] wrong output shapes {V.shape}, {ev.shape}, {Z.shape}")
        expect(np.allclose(V @ V.T, np.eye(k), atol=1e-6), f"[{name}] components are not orthonormal")
        tol = 1e-3 if X.dtype == np.float32 else 1e-6
        expect(np.allclose(ev, er, rtol=tol), f"[{name}] explained variance {ev[:3]} vs reference {er[:3]} (n-1? centring?)")
        expect(np.allclose(V, Vr, atol=tol * 10), f"[{name}] components differ from reference (order, sign convention, centring?)")
        expect(np.allclose(Z, Zr, atol=tol * 100), f"[{name}] scores Z differ from reference")
    for bad_k in (0, 11, -1):
        try:
            pca(make(30, 10, 0.0), bad_k)
        except ValueError:
            pass
        else:
            raise CheckFail(f"k={bad_k} with d=10 should raise ValueError")
'''

SOLUTION = '''def pca(X, k):
    X = np.asarray(X, dtype=float)                    # float64 working copy (float32 input is promoted)
    n, d = X.shape
    if not 1 <= k <= min(n, d):
        raise ValueError(f"k must be between 1 and min(n, d) = {min(n, d)}, got {k}")
    Xc = X - X.mean(axis=0)                           # centre first: a large mean offset would swamp the covariance
    evals, evecs = np.linalg.eigh(Xc.T @ Xc / (n - 1))
    V = evecs[:, ::-1][:, :k].T                       # eigh sorts ascending; keep the k largest as rows
    V = V * np.sign(V[np.arange(k), np.abs(V).argmax(axis=1)])[:, None]
    return V, evals[::-1][:k], Xc @ V.T
'''

MCQ = [
    {"q": r"Libraries compute PCA from the SVD of the centred $X$ rather than from `eigh` of $X^\top X$. Why?",
     "options": [r"`eigh` is $O(d^3)$ while the SVD is $O(nd)$, so it is purely a speed choice.",
                 r"Forming $X^\top X$ squares the condition number, so a component with $\sigma_i/\sigma_1 \approx 10^{-9}$ has $\lambda_i/\lambda_1 \approx 10^{-18}$, below float64 resolution, and comes out as noise.",
                 r"`eigh` does not guarantee orthonormal eigenvectors, while the SVD does.",
                 r"$X^\top X$ is not exactly symmetric in floating point, so `eigh` can return complex eigenvalues."],
     "answer": 1,
     "why": r"$\lambda_i = \sigma_i^2$, so relative gaps square. Eigenvalues smaller than about $\varepsilon_{\text{mach}}\lambda_1$ drown in rounding error, while the SVD of $X$ resolves them down to about $\varepsilon_{\text{mach}}\sigma_1$."},
    {"q": r"PCA on centred data with sample-covariance eigenvalues $\lambda_1 \ge \dots \ge \lambda_d$. You keep $k$ components and reconstruct each $x_i$ as $\hat x_i$. What is $\frac{1}{n-1}\sum_i \lVert x_i - \hat x_i\rVert^2$?",
     "options": [r"$\lambda_1 + \dots + \lambda_k$, the variance that was kept.",
                 r"$\lambda_{k+1}$, the largest discarded eigenvalue.",
                 r"$\lambda_{k+1} + \dots + \lambda_d$, the sum of the discarded eigenvalues.",
                 r"$\sqrt{\lambda_{k+1}} + \dots + \sqrt{\lambda_d}$, the discarded singular values."],
     "answer": 2,
     "why": r"The residual $x_i - \hat x_i$ is the projection onto the discarded eigenvectors, and the variance along eigenvector $j$ is $\lambda_j$. Dividing by $n$ instead of $n-1$ only rescales by $(n-1)/n$."},
]
