TITLE = 'Rank and null space by row reduction'
MODULES = ['04']

STATEMENT = r'''
Write `rank_nullspace(A, tol=None)` returning `(r, N)`:
- `r`: the rank of `A`.
- `N`: an `(n, n - r)` array whose columns are a basis of the null space of `A` (they need not be orthonormal). If `r == n`, return shape `(n, 0)`.
- Treat a pivot as zero if its absolute value is at most `tol`. Default `tol = 1e-9 * max|A|`.

Use row reduction to reduced row echelon form. Forbidden: `matrix_rank`, `svd`, `null_space`, `orth`, `qr`, `lstsq`, `pinv`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `solve`. At most one level of Python loop.
'''

STARTER = '''def rank_nullspace(A, tol=None):
    # YOUR CODE
    raise NotImplementedError
'''

CHECK = r'''def _check():
    static_rules([rank_nullspace], ["matrix_rank", "svd", "null_space", "orth", "qr", "lstsq", "pinv", "eig", "eigh", "eigvals", "eigvalsh", "solve"], 1)
    banned = ("numpy.linalg.matrix_rank", "numpy.linalg.svd", "scipy.linalg.null_space", "scipy.linalg.svd", "numpy.linalg.qr")
    rng = np.random.default_rng(3)
    cases = [
        ("rank-3 6x8", rng.normal(size=(6, 3)) @ rng.normal(size=(3, 8)), 3),
        ("zero 3x4", np.zeros((3, 4)), 0),
        ("full column rank 5x3", rng.normal(size=(5, 3)), 3),
        ("wide full row rank 3x7", rng.normal(size=(3, 7)), 3),
        ("rank-2 plus 1e-13 noise", rng.normal(size=(7, 2)) @ rng.normal(size=(2, 6)) + 1e-13 * rng.normal(size=(7, 6)), 2),
        ("duplicate columns", np.repeat(rng.normal(size=(4, 2)), 3, axis=1), 2),
        ("integer matrix", np.array([[1, 2, 3], [2, 4, 6], [1, 0, 1]]), 2),
    ]
    for name, A, r_true in cases:
        A0 = np.array(A, copy=True)
        with blocked(*banned): r, N = rank_nullspace(A)
        n = A0.shape[1]
        expect(r == r_true, f"[{name}] rank {r}, expected {r_true}")
        N = np.asarray(N, float)
        expect(N.shape == (n, n - r_true), f"[{name}] N has shape {N.shape}, expected {(n, n - r_true)}")
        if N.shape[1]:
            scale = max(np.abs(A0).max(), 1.0) * max(np.abs(N).max(), 1.0)
            expect(np.abs(A0 @ N).max() <= 1e-8 * scale, f"[{name}] A @ N is not close to zero (max {np.abs(A0 @ N).max():.1e})")
            expect(np.linalg.matrix_rank(N) == n - r_true, f"[{name}] columns of N are not independent")
'''

SOLUTION = '''def rank_nullspace(A, tol=None):
    R = np.array(A, dtype=float)                      # a copy we can reduce in place
    m, n = R.shape
    if tol is None:
        tol = 1e-9 * np.abs(R).max() if R.size else 0.0
    pivots = []
    for j in range(n):                                # one pass over columns builds the RREF
        r = len(pivots)
        if r == m:
            break
        p = r + np.abs(R[r:, j]).argmax()
        if abs(R[p, j]) <= tol:                       # no usable pivot: column j is free
            continue
        R[[r, p]] = R[[p, r]]
        R[r] /= R[r, j]
        others = np.arange(m) != r
        R[others] -= np.outer(R[others, j], R[r])     # clear column j above and below the pivot
        pivots.append(j)
    r = len(pivots)
    free = np.setdiff1d(np.arange(n), pivots)
    N = np.zeros((n, len(free)))
    N[free, np.arange(len(free))] = 1.0               # each free variable set to 1 in turn ...
    N[pivots] = -R[:r][:, free]                       # ... pivot variables read off the RREF
    return r, N
'''

MCQ = [
    {"q": r"An embedding table $E$ is $50{,}000 \times 768$. Its singular values fall from about 10 to below $10^{-6}$ after the 120th. Viewing $E$ as a map $\mathbb{R}^{768} \to \mathbb{R}^{50000}$, what are its numerical rank and null-space dimension?",
     "options": [r"Rank 768 and nullity 0, because $E$ has far more rows than columns.",
                 r"Rank 120 and nullity 49,880.",
                 r"Rank 648 and nullity 120.",
                 r"Rank 120 and nullity 648: the embeddings effectively live in a 120-dimensional subspace."],
     "answer": 3,
     "why": r"Rank-nullity counts against the input dimension: $768 = 120 + 648$. Only about 120 directions of the 768-dimensional embedding space carry any signal."},
    {"q": r"A linear layer maps $\mathbb{R}^{4096} \to \mathbb{R}^{1024}$. What is the smallest possible dimension of its null space, and what follows for the next layer?",
     "options": [r"3072: at least 3072 input directions map to zero, so no later layer can recover those components of the input.",
                 r"0, if the weights are full rank: a full-rank layer loses no information.",
                 r"1024, the output dimension.",
                 r"It cannot be bounded without knowing the weights."],
     "answer": 0,
     "why": r"The rank is at most 1024, so nullity $= 4096 - \text{rank} \ge 3072$. Inputs that differ by a null-space vector give identical outputs, so the layer cannot be inverted."},
]
