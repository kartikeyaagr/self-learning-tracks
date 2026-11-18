TITLE = 'A linear solver you can trust'
MODULES = ['03']

STATEMENT = r'''
Write `gauss_solve(A, B)` that solves $AX = B$ by Gaussian elimination with partial pivoting.

Spec:
- `A` is `(n, n)`. `B` is `(n,)` or `(n, k)`. Return a float64 array with the same shape as `B`.
- Accept integer inputs. Do not modify `A` or `B` in place.
- Raise `np.linalg.LinAlgError` if `A` is singular to working precision (use a pivot threshold of `n * eps * max|A|`).
- Must be backward stable on ill-conditioned but non-singular inputs.

Forbidden: `solve`, `inv`, `lstsq`, `pinv`, `qr`, `lu`, `lu_factor`, `lu_solve`, `cho_solve`, `svd`, `solve_triangular`. At most one level of Python loop (a loop over columns is fine; a loop over rows inside it is not).
'''

STARTER = '''def gauss_solve(A, B):
    # YOUR CODE
    raise NotImplementedError
'''

CHECK = r'''def _check():
    static_rules([gauss_solve], ["solve", "inv", "lstsq", "pinv", "qr", "lu", "lu_factor", "lu_solve", "cho_solve", "svd", "solve_triangular"], 1)
    banned = ("numpy.linalg.solve", "numpy.linalg.inv", "numpy.linalg.lstsq", "scipy.linalg.solve", "scipy.linalg.lu_factor", "scipy.linalg.inv")
    rng = np.random.default_rng(2)
    A = rng.normal(size=(60, 60)); B = rng.normal(size=(60, 4))
    A0, B0 = A.copy(), B.copy()
    with blocked(*banned): X = gauss_solve(A, B)
    expect(np.array_equal(A, A0) and np.array_equal(B, B0), "your function modified A or B in place")
    expect(X.shape == (60, 4), f"shape {X.shape}, expected (60, 4)")
    expect(np.allclose(X, np.linalg.solve(A, B), rtol=1e-8, atol=1e-10), "random 60x60 system: answer differs from reference")
    b = rng.normal(size=60)
    with blocked(*banned): x = gauss_solve(A, b)
    expect(x.shape == (60,), f"1-D right-hand side should give shape (60,), got {x.shape}")
    try:
        with blocked(*banned): x = gauss_solve(np.array([[1e-20, 1.0], [1.0, 1.0]]), np.array([1.0, 2.0]))
    except np.linalg.LinAlgError:
        raise CheckFail("[[1e-20, 1], [1, 1]] is well conditioned but you raised LinAlgError (pivoting?)")
    expect(np.allclose(x, [1.0, 1.0], atol=1e-10), f"tiny-pivot system gives {x}, expected about [1, 1] (pivoting?)")
    with blocked(*banned): x = gauss_solve(np.array([[2, 1], [1, 3]]), np.array([3, 5]))
    expect(np.allclose(x, [0.8, 1.4]) and x.dtype == np.float64, "integer input: wrong answer or dtype")
    with blocked(*banned): x = gauss_solve(np.array([[4.0]]), np.array([2.0]))
    expect(np.allclose(x, [0.5]), "1x1 system wrong")
    for S in (np.array([[1.0, 2.0], [2.0, 4.0]]), np.zeros((3, 3)), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)):
        try:
            with blocked(*banned): gauss_solve(S, np.ones(len(S)))
        except np.linalg.LinAlgError:
            pass
        else:
            raise CheckFail(f"singular matrix {S.tolist()} did not raise LinAlgError")
    H = scipy.linalg.hilbert(9); xt = np.ones(9); bh = H @ xt
    with blocked(*banned): xh = gauss_solve(H, bh)
    rel_res = np.linalg.norm(H @ xh - bh) / (np.linalg.norm(H, 2) * np.linalg.norm(xh))
    expect(rel_res < 1e-14, f"Hilbert(9): relative residual {rel_res:.1e} too large; not backward stable")
'''

SOLUTION = '''def gauss_solve(A, B):
    A = np.array(A, dtype=float)                      # copies: inputs stay untouched
    B = np.array(B, dtype=float)
    n = len(A)
    M = np.hstack([A, B.reshape(n, -1)])              # augmented [A | B]
    tol = n * np.finfo(float).eps * np.abs(A).max()
    for j in range(n):                                # forward elimination, one column at a time
        p = j + np.abs(M[j:, j]).argmax()             # partial pivoting: largest entry on or below the diagonal
        if abs(M[p, j]) <= tol:
            raise np.linalg.LinAlgError("matrix is singular to working precision")
        M[[j, p]] = M[[p, j]]
        M[j + 1:] -= np.outer(M[j + 1:, j] / M[j, j], M[j])
    X = M[:, n:]
    for j in range(n - 1, -1, -1):                    # back substitution (a separate loop, not nested)
        X[j] = (X[j] - M[j, j + 1:n] @ X[j + 1:]) / M[j, j]
    return X.reshape(B.shape)
'''

MCQ = [
    {"q": r"You solve the Hilbert(9) system $Hx = H\mathbf{1}$ in float64 with a backward-stable solver. $\kappa(H) \approx 5 \times 10^{11}$. The residual is tiny. Roughly how many correct significant digits should you expect in $x$?",
     "options": [r"About 16: a tiny residual means $x$ is accurate to machine precision.",
                 r"About 4 to 5: the relative error can reach about $\kappa \cdot \varepsilon_{\text{mach}} \approx 5 \times 10^{11} \times 2 \times 10^{-16} \approx 10^{-4}$.",
                 r"About 11, the exponent of the condition number.",
                 r"None: a matrix with $\kappa > 10^{8}$ is singular in float64, so the solve is meaningless."],
     "answer": 1,
     "why": r"Backward stability makes the residual small, but the forward error is amplified by up to $\kappa$: you lose about $\log_{10}\kappa \approx 11.7$ of float64's roughly 16 digits, leaving about 4 to 5."},
    {"q": r"A teammate fits linear regression with `np.linalg.solve(X.T @ X, X.T @ y)`. $\kappa(X) = 10^{8}$ and the coefficients look random. What is going on, and what is the fix?",
     "options": [r"$\kappa(X^\top X) = 10^{4}$, so the system is fine; the data must be noisy, so collect more.",
                 r"$\kappa(X^\top X) = 10^{8}$; `solve` does not pivot, so switch to `np.linalg.inv(X.T @ X) @ X.T @ y`.",
                 r"$\kappa(X^\top X) = 10^{16}$, which is the limit of float64; fix it by forming `inv(X.T @ X)` explicitly, which is more accurate than `solve`.",
                 r"$\kappa(X^\top X) = 10^{16} \approx 1/\varepsilon_{\text{mach}}$, so the normal equations keep no correct digits; solve the least-squares problem on $X$ directly with QR or `np.linalg.lstsq`."],
     "answer": 3,
     "why": r"Forming $X^\top X$ squares the condition number. QR or SVD-based `lstsq` works with $X$ itself, where $\kappa = 10^{8}$ still leaves about 8 digits."},
]
