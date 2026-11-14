---
id: "01.07"
course: "ml-toolkit"
module: "01"
slug: "linear-algebra-and-random-numbers-in-numpy"
title: "Linear algebra and random numbers in NumPy"
keyIdeas: ["`@` (`np.matmul`) broadcasts over leading batch axes and calls BLAS; `*` is elementwise. Prefer `@` to `np.dot`, whose behaviour on 3-D and higher inputs is different and rarely what you want.", "To compute $A^{-1}b$, call `np.linalg.solve(A, b)`: one LU factorisation, about a third of the work of `inv`, and more accurate. Expect to lose roughly $\\log_{10}\\kappa(A)$ digits either way.", "`lstsq` solves least squares through the SVD and copes with rank deficiency. Forming the normal equations $X^\\top X$ squares the condition number.", "Use `eigh` for symmetric matrices (real eigenvalues, ascending, orthonormal vectors) and `svd(full_matrices=False)` for data matrices. Eigenvectors are only defined up to sign.", "Create one `rng = np.random.default_rng(seed)` and pass it around; derive independent streams with `rng.spawn` or `SeedSequence`, never by reusing the global `np.random.seed` state."]
question: "A ridge regression implementation computes `w = np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T @ y` with X of shape (1,000,000, 50). Comment on its cost and accuracy, and propose a better version. When would you prefer lstsq instead?"
answer: "The expensive part is `X.T @ X`, about $n d^2 = 2.5 \\times 10^9$ multiply-adds, which is fine. The inverse is only 50 x 50, so its cost is negligible, but it is still the wrong tool: `np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)` is cheaper and more accurate, and because the matrix is symmetric positive definite, a Cholesky solve (`scipy.linalg.cho_factor` and `cho_solve`) is better still. Also note the evaluation order: `inv(...) @ X.T @ y` multiplies a 50 x 50 matrix by the 50 x 1,000,000 `X.T` first; computing `X.T @ y` first avoids that large intermediate. Accuracy depends on $\\kappa(X)^2$ shrunk by the ridge term. When lam is tiny or zero and features are badly scaled or collinear, use `lstsq` on the augmented system `[X; sqrt(lam) I] w = [y; 0]`, which works with $\\kappa(X)$ rather than its square."
sources: [{"title": "NumPy reference: Linear algebra (numpy.linalg)", "url": "https://numpy.org/doc/stable/reference/routines.linalg.html", "note": "matmul broadcasting rules and the full list of factorisations and solvers."}, {"title": "numpy.linalg.lstsq", "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html", "note": "Return values, rcond, and the minimum-norm solution for rank-deficient X."}, {"title": "NumPy reference: Random Generator", "url": "https://numpy.org/doc/stable/reference/random/generator.html", "note": "default_rng, the Generator methods, and spawn."}, {"title": "NEP 19: Random number generator policy", "url": "https://numpy.org/neps/nep-0019-rng-policy.html", "note": "Why the global state was deprecated and what stream compatibility NumPy promises."}]
practice: ["solve-linear-system", "least-squares-solution", "eigendecomposition", "svd-components", "random-array-generation"]
tags: ["linear algebra", "conditioning", "reproducibility"]
tools: ["NumPy", "SciPy"]
---

## matmul and the @ operator

`A @ B` multiplies matrices: `(n, k) @ (k, m)` gives `(n, m)`. With more than two dimensions, the last two axes are the matrices and the leading axes broadcast like any elementwise operation, so `(32, 8, 64, 16) @ (16, 64)` gives `(32, 8, 64, 64)`. A 1-D operand is treated as a row or column vector and that axis is dropped from the result.

Two practical points. First, `np.dot` predates `@` and uses a different rule for 3-D inputs (it contracts the last axis of the first array with the second-to-last of the second, producing a 4-D outer structure), so stick to `@` or `einsum`. Second, the work goes to an optimised BLAS library; float32 runs roughly twice as fast as float64 there and uses half the memory bandwidth.

## solve, not inv

The maths says $x = A^{-1}b$. The code should not. `np.linalg.solve(A, b)` computes an LU factorisation with partial pivoting (about $\tfrac{2}{3}n^3$ flops) and then two cheap triangular solves. `np.linalg.inv(A) @ b` costs about $2n^3$ for the inverse, about three times as much, and is less accurate because the rounding errors of the explicit inverse are multiplied into the result. If you need to solve against many right-hand sides, pass them as columns of one `B`, or factorise once with SciPy (`lu_factor`, `cho_factor`) and reuse it.

Accuracy is governed by the **condition number** $\kappa(A) = \sigma_{\max} / \sigma_{\min}$. As a rule of thumb, the relative error in $x$ is about $\kappa(A)$ times machine epsilon: with $\kappa = 10^{10}$ in float64 ($\varepsilon \approx 10^{-16}$) you can trust about 6 digits, and in float32 none. `np.linalg.solve` raises `LinAlgError` only for an *exactly* singular matrix; a nearly singular one quietly returns large, noisy numbers. Check `np.linalg.cond(A)` when results look suspicious. The ML Maths course's linear systems module covers the theory behind invertibility and row reduction.

~~~dg-flow Pick the routine from the structure of the problem.
Square A, solve Ax = b -(general)-> np.linalg.solve (LU)
Symmetric positive definite A -(faster, stabler)-> *Cholesky (scipy cho_solve)
Tall X, min of the residual norm -(rank-safe)-> np.linalg.lstsq (SVD)
~~~

## Least squares with lstsq

For an overdetermined system $Xw \approx y$ with $X$ of shape $(n, d)$, $n > d$, `np.linalg.lstsq(X, y, rcond=None)` returns the minimiser of $\lVert Xw - y \rVert_2$, plus the residuals, the numerical rank and the singular values. It works through the SVD, so collinear features do not crash it: singular values below `rcond` times the largest are treated as zero, and you get the minimum-norm solution.

The tempting shortcut `np.linalg.solve(X.T @ X, X.T @ y)` (the normal equations) is faster but squares the condition number: $\kappa(X^\top X) = \kappa(X)^2$. A feature matrix with $\kappa = 10^5$, common with unscaled features, becomes $10^{10}$, and in float32 the answer is noise. Normal equations are acceptable when $X$ is well conditioned, or with a ridge term, $(X^\top X + \lambda I)w = X^\top y$, which caps the condition number.

## eigh and svd

- `np.linalg.eigh(S)` for symmetric (or Hermitian) matrices such as covariance matrices returns real eigenvalues in **ascending** order and orthonormal eigenvectors as **columns**. For PCA you usually want the last columns.
- `np.linalg.eig` handles general matrices but may return complex values in no particular order. Do not use it on a covariance matrix: rounding can introduce tiny imaginary parts.
- `np.linalg.svd(X, full_matrices=False)` returns `U` (n, r), `s` (r,) in **descending** order, and `Vt` (r, d), with $X = U\,\mathrm{diag}(s)\,V^\top$. For PCA, centre `X` and take the first rows of `Vt`; explained variance is `np.square(s) / (n - 1)`. The full form allocates an n x n `U`, which is easy to blow up memory with.
- Eigenvectors and singular vectors are defined only up to sign (and up to rotation within repeated eigenvalues). Tests and comparisons must allow for this, for example by fixing the sign so the largest-magnitude entry of each vector is positive.

## Random numbers you can reproduce

The legacy API (`np.random.seed`, `np.random.rand`) uses one hidden global state. Any library call that draws from it changes your sequence, and processes forked from the same parent (for example data-loader workers) inherit identical state and produce identical "random" augmentations. The modern API makes the state explicit:

```python
rng = np.random.default_rng(42)        # PCG64 generator
x = rng.normal(size=(1000, 3))
idx = rng.permutation(len(x))
workers = rng.spawn(8)                 # 8 independent child generators
```

Pass `rng` into functions instead of seeding inside them. Child streams from `spawn` (or `np.random.SeedSequence(42).spawn(8)`) are statistically independent; `default_rng(seed + i)` usually works but carries no such guarantee. One more caveat from NumPy's policy (NEP 19): the `Generator` stream for a given seed may change between NumPy versions when algorithms improve, whereas the legacy `RandomState` is frozen. Exact bit-for-bit reproducibility therefore also requires pinning the NumPy version.

<div class="ex">
<div class="lab">Worked example</div>

**how many digits the normal equations cost**

Fit $y = w_0 + w_1 t$ where `t` holds 1,000 Unix timestamps spread over one year, around $1.7 \times 10^9$ seconds. The ones column and the `t` column are nearly parallel: measured $\kappa(X) \approx 3 \times 10^{11}$.

The normal equations work with $\kappa(X^\top X) \approx 10^{23}$, beyond what float64 can resolve. In one run, `lstsq` matched the reference solution to a relative error of $10^{-14}$, while `solve(X.T @ X, X.T @ y)` was off by $2 \times 10^{-9}$: five digits gone. In float32 neither result is usable.

The best fix is upstream: centre and scale `t` first, after which $\kappa$ is close to 1 and every method agrees.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

use `@`, `solve`, `lstsq`, `eigh` and `svd(full_matrices=False)` instead of their tempting alternatives, check the condition number when answers look odd, and thread one explicit `Generator` through your code.

</div>
