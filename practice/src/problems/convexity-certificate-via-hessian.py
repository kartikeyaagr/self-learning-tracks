import numpy as np

META = dict(title='Certify convexity from the Hessian', tags=['Optimization', 'Linear Algebra'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:07.01', 'mlmaths:05.02'],
            related=['classify-critical-point-from-hessian', 'minimum-of-axis-aligned-paraboloid'])

STATEMENT = r"""
A quadratic $f(x) = x^\top A x + b^\top x + c$ is convex exactly when its Hessian is positive semidefinite. Careful: $A$ is **not** required to be symmetric. Only its symmetric part shapes $f$, and the Hessian is

$$H = \nabla^2 f = A + A^\top.$$

Write `convexity_certificate(A, tol=1e-9)` that returns a tuple `(label, lam_min, direction)`:

- `lam_min`: the smallest eigenvalue of $H$, as a Python `float`.
- `label`: a string.
  - `'strictly convex'` if `lam_min > tol`
  - `'convex'` if `-tol <= lam_min <= tol`
  - `'not convex'` if `lam_min < -tol`
- `direction`: when the label is `'not convex'`, a unit-length float array `(d,)` along which $f$ curves downward (an eigenvector of $H$ for `lam_min`), so that $v^\top H v = \lambda_{\min} < 0$. Otherwise `None`.

## Conventions
- `A`: float array `(d, d)`, `d >= 1`. `b` and `c` don't affect convexity, so they aren't inputs.
- Sign: flip `direction` so its entry with the largest absolute value is positive (ties: the first such entry).
- Tests guarantee that whenever the label is `'not convex'`, the smallest eigenvalue of $H$ is simple, so the direction is unique up to that sign.
"""

STARTER = '''import numpy as np

def convexity_certificate(A, tol=1e-9):
    """Return (label str, lam_min float, direction (d,) array or None)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def convexity_certificate(A, tol=1e-9):
    A = np.asarray(A, dtype=float)
    H = A + A.T                              # Hessian of x^T A x; the antisymmetric part of A cancels
    vals, vecs = np.linalg.eigh(H)           # ascending eigenvalues, orthonormal eigenvectors
    lam = float(vals[0])
    if lam > tol:
        return 'strictly convex', lam, None
    if lam >= -tol:
        return 'convex', lam, None
    v = vecs[:, 0]
    v = v * np.sign(v[np.argmax(np.abs(v))])     # canonical sign: largest-magnitude entry positive
    return 'not convex', lam, v
'''

HINTS = [
    'Write $x^\\top A x$ with $A$ split into symmetric and antisymmetric parts. What happens to the antisymmetric part?',
    'Symmetrise first, $H = A + A^\\top$, then use `np.linalg.eigh`. It returns eigenvalues in ascending order, so `vals[0]` is the smallest.',
    'The certificate of non-convexity is the eigenvector in column 0 of the `eigh` output. Fix its sign with `np.argmax(np.abs(v))`.',
]

TESTS = [
    (np.array([[1.0, 0.0], [0.0, 2.0]]),),
    (np.array([[1.0, 0.0], [0.0, -3.0]]),),
    (np.array([[0.0, 5.0], [-5.0, 0.0]]),),
    (np.array([[1.0, 4.0], [0.0, 2.0]]),),
    (np.array([[1.0, 1.0], [1.0, 1.0]]),),
    (np.array([[-2.0]]),),
    (np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]),),
    (np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0], [1.5, 0.0, 0.2]]),),
]

EXPLANATION = r"""
## Why the symmetric part
Any $A$ splits as $S + K$ with $S = (A + A^\top)/2$ symmetric and $K = (A - A^\top)/2$ antisymmetric. Since $x^\top K x$ is a scalar equal to its own transpose $x^\top K^\top x = -x^\top K x$, it is 0. So $f$ depends only on $S$, and the Hessian is $2S = A + A^\top$. Test 3 is pure rotation-like $A$: $f \equiv 0$, convex but not strictly.

## Why eigenvalues certify convexity
A twice-differentiable function is convex iff its Hessian is PSD everywhere. For a quadratic the Hessian is constant, and $H$ symmetric means $v^\top H v \ge \lambda_{\min}\lVert v\rVert^2$ with equality at the bottom eigenvector. So $\lambda_{\min} \ge 0$ is the whole test, and when it fails the eigenvector is a concrete witness: along $x = t v$, $f$ behaves like $\tfrac12\lambda_{\min} t^2 \to -\infty$.

## Complexity
`eigh` on a $d \times d$ matrix is $O(d^3)$. A cheaper yes/no test for strict convexity is a Cholesky factorisation (`np.linalg.cholesky` succeeds iff $H$ is positive definite), but it gives neither $\lambda_{\min}$ nor a witness direction.

## Common bugs
- Calling `eigh(A)` on the unsymmetrised matrix. `eigh` silently reads only one triangle, so test 4 gets the wrong answer.
- Using `np.linalg.eig`, which can return complex values and unsorted pairs.
- Checking `lam_min >= 0` exactly. Round-off makes a singular PSD matrix report $-10^{-17}$; that is why a tolerance exists.

## Interview follow-ups
- For a non-quadratic $f$, the Hessian varies with $x$. How would you test convexity on a region? (Bound $\lambda_{\min}(\nabla^2 f(x))$ over the region; sampling points can only disprove convexity, never prove it.)
- Logistic loss has Hessian $X^\top D X$ with $D$ diagonal positive. Why is it always convex? ($v^\top X^\top D X v = \lVert D^{1/2} X v\rVert^2 \ge 0$.)
"""
