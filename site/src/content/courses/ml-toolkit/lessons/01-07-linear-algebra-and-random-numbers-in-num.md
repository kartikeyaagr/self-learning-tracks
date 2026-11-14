---
id: "01.07"
course: "ml-toolkit"
module: "01"
slug: "linear-algebra-and-random-numbers-in-numpy"
title: "Linear algebra and random numbers in NumPy"
keyIdeas: ["`@` (`np.matmul`) is the matrix product: `(n, k) @ (k, m)` gives `(n, m)`, leading batch axes broadcast, and the work goes to BLAS. `*` is elementwise, and `np.dot` behaves differently on 3-D inputs, so prefer `@` or `einsum`.", "To compute $A^{-1}b$, call `np.linalg.solve(A, b)`: one LU factorisation, about a third of the work of `inv`, and more accurate. `solve` only fails on exactly singular matrices; nearly singular ones quietly return huge numbers.", "The **condition number** $\\kappa = \\sigma_{\\max}/\\sigma_{\\min}$ measures sensitivity: expect to lose about $\\log_{10}\\kappa$ of your 16 (float64) or 7 (float32) digits.", "`lstsq` solves least squares through the SVD and copes with redundant columns by returning the minimum-norm solution. The normal equations $X^\\top X w = X^\\top y$ square the condition number.", "Use `eigh` for symmetric matrices (real eigenvalues, ascending, eigenvectors as columns) and `svd(full_matrices=False)` for data matrices (descending singular values). Vectors are only defined up to sign.", "Create one `rng = np.random.default_rng(seed)` and pass it into functions; give parallel workers independent streams with `rng.spawn`, never by copying the global `np.random.seed` state."]
question: "A ridge regression implementation computes `w = np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T @ y` with X of shape (1,000,000, 50). Comment on its cost and accuracy, and propose a better version. When would you prefer lstsq instead?"
answer: "Cost: the expensive part is `X.T @ X`, about $n d^2 = 2.5 \\times 10^9$ multiply-adds, which BLAS handles in a second or so. The inverse is only 50 x 50, so its cost is negligible, but it is still the wrong tool: `np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)` is cheaper and more accurate, and because the matrix is symmetric positive definite, a Cholesky solve (`scipy.linalg.cho_factor` and `cho_solve`) is better still. Also watch the evaluation order: `@` is left-associative, so `inv(...) @ X.T @ y` first multiplies a 50 x 50 matrix by the 50 x 1,000,000 `X.T`, creating a 400 MB intermediate and doing $n d^2$ extra work; computing `X.T @ y` first gives a 50-vector. Accuracy: the system's condition number is $(\\sigma_{\\max}^2 + \\lambda)/(\\sigma_{\\min}^2 + \\lambda)$, roughly $\\kappa(X)^2$ shrunk by the ridge term, so with a reasonable lam and standardised features it is fine. When lam is tiny or zero and the features are badly scaled or nearly collinear, $\\kappa(X)^2$ can exceed $10^{16}$ and the normal equations lose every digit. Then use `lstsq` on the augmented system `[X; sqrt(lam) I] w = [y; 0]`, which minimises the same ridge objective but works with the condition number of the augmented X rather than its square. In all cases, standardising the columns of X first is the cheapest improvement."
sources: [{"title": "NumPy reference: Linear algebra (numpy.linalg)", "url": "https://numpy.org/doc/stable/reference/routines.linalg.html", "note": "matmul broadcasting rules and the full list of factorisations and solvers."}, {"title": "numpy.linalg.lstsq", "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html", "note": "Return values, rcond, and the minimum-norm solution for rank-deficient X."}, {"title": "NumPy reference: Random Generator", "url": "https://numpy.org/doc/stable/reference/random/generator.html", "note": "default_rng, the Generator methods, and spawn."}, {"title": "NEP 19: Random number generator policy", "url": "https://numpy.org/neps/nep-0019-rng-policy.html", "note": "Why the global state was deprecated and what stream compatibility NumPy promises."}]
practice: ["solve-linear-system", "least-squares-solution", "eigendecomposition", "svd-components", "random-array-generation"]
tags: ["linear algebra", "conditioning", "reproducibility"]
tools: ["NumPy", "SciPy"]
---

## What you will learn

- How `@` (matrix multiplication) lines up shapes, including stacks of matrices, and why it beats `np.dot` and `*`.
- Why you solve $Ax = b$ with `np.linalg.solve` instead of computing an inverse, with measured speed and accuracy.
- What the **condition number** is, and how to predict how many correct digits an answer can have.
- How `lstsq`, `eigh` and `svd` work in practice: what they return, in which order, and the sign trap.
- How to make random numbers reproducible with one explicit `Generator`, and how to give parallel workers independent streams.

*Before you start:* you should be comfortable with array shapes and broadcasting (01.03), reductions along an axis (01.05) and `einsum` (01.06). It helps to know what a matrix product and a system of linear equations are; the ML Maths course covers both if they are new.

## From reshaping to linear algebra

Lesson 01.06 was about moving axes around and writing contractions with `einsum`. This last lesson puts those shapes to work. Almost every model calls four kinds of routines: a matrix product, a linear solve, a factorisation (PCA, whitening) and a random number generator. NumPy has a good version of each, sitting next to a tempting wrong one.

## The idea in one picture

Solving a linear system means finding the numbers that satisfy several equations at once. Here are two equations in two unknowns:

$$2x + y = 3, \qquad x + 3y = 5.$$

In matrix form this is $Ax = b$ with

| | column x | column y | right-hand side b
| --- | --- | --- | --- |
| equation 1 | 2 | 1 | 3
| equation 2 | 1 | 3 | 5

Each equation is a line in the plane, and the solution is the point where the lines cross.

```python
import numpy as np
A = np.array([[2., 1.], [1., 3.]])
b = np.array([3., 5.])
x = np.linalg.solve(A, b)
print(x)                       # [0.8 1.4]
print(np.allclose(A @ x, b))   # True
print(np.linalg.cond(A))       # 2.618...
```

~~~dg-fig 01.07-lines.svg
Left: two lines crossing at a clear angle pin down the solution (0.8, 1.4). Right: two nearly parallel lines. Changing the right-hand side by only 0.001 moves the crossing from (1, 1) to (0, 2), which is what a large condition number means.
~~~

On the left the lines meet at a wide angle, so small changes in the numbers barely move the crossing: the system is **well conditioned**. On the right the lines are almost parallel, and a change of 0.001 in $b$ moves the answer by a whole unit: the system is **ill conditioned**. Rounding errors in floating point are exactly such tiny changes.

The **condition number** $\kappa(A)$ puts a number on this sensitivity: about 2.6 on the left, about 4000 on the right. You will meet it again in every section below.

## Step 1: multiply with @

`A @ B` is the matrix product. The rule for shapes: `(n, k) @ (k, m)` gives `(n, m)`. The inner sizes (the two `k`s) must match, and each entry of the result is one row of `A` dotted with one column of `B`.

~~~dg-fig 01.07-matmul.svg
A (2, 3) matrix times a (3, 2) matrix. The highlighted row of A and column of B multiply element by element and add up to the single highlighted entry of the (2, 2) result.
~~~

In symbols, $C_{ij} = \sum_{k} A_{ik} B_{kj}$: $i$ picks the row of $A$, $j$ the column of $B$, and $k$ runs along the shared inner axis. Check the highlighted cell: row 1 of `A` is `[3, 4, 5]`, column 0 of `B` is `[0, 2, 4]`, and $3 \cdot 0 + 4 \cdot 2 + 5 \cdot 4 = 28$.

```python
A = np.arange(6).reshape(2, 3)       # [[0 1 2] [3 4 5]]
B = np.arange(6).reshape(3, 2)       # [[0 1] [2 3] [4 5]]
print(A @ B)                         # [[10 13] [28 40]]
print(A * A)                         # elementwise, same shape: [[0 1 4] [9 16 25]]

v = np.array([1., 2., 3.])
print((A @ v).shape)                 # (2,)  a 1-D operand acts as a vector; its axis is dropped

batch = np.ones((32, 8, 64, 16))
W = np.ones((16, 64))
print((batch @ W).shape)             # (32, 8, 64, 64)
```

Three rules:

- `*` is **elementwise** multiplication with broadcasting (01.03). It is not the matrix product. Mixing the two is the most common linear algebra bug in NumPy code.
- With more than two dimensions, the **last two axes are the matrices** and all leading axes broadcast like any elementwise operation. A batch of 32 x 8 matrices times one weight matrix just works.
- A 1-D operand is treated as a column (on the right) or a row (on the left), and that axis disappears from the result.

Why not `np.dot`? On 3-D inputs it builds an outer structure instead of a batch: `np.dot` of shapes `(2,3,4)` and `(5,4,6)` gives `(2, 3, 5, 6)`, while `@` on `(2,3,4)` and `(2,4,6)` gives `(2, 3, 6)`. Use `@`, and `einsum` (01.06) for anything unusual.

Under the hood `@` calls an optimised **BLAS** library (Basic Linear Algebra Subprograms) that uses all your cores. A Python loop over rows is hundreds of times slower, and float32 runs roughly twice as fast as float64: here a 1000 x 1000 product took 8.6 ms in float64 and 4.1 ms in float32 (machine-dependent).

## Step 2: solve, do not invert

The maths textbook writes the solution of $Ax = b$ as $x = A^{-1} b$. Your code should not. `np.linalg.solve(A, b)` does three things:

1. It factorises $A = PLU$ with **partial pivoting**: $L$ is lower-triangular, $U$ upper-triangular, and $P$ reorders rows so that no tiny number is used as a divisor. This costs about $\tfrac{2}{3} n^3$ floating-point operations (flops) for an $n \times n$ matrix.
2. It solves $Ly = Pb$ by forward substitution, about $n^2$ flops.
3. It solves $Ux = y$ by back substitution, another $n^2$ flops.

`np.linalg.inv(A) @ b` instead builds the whole inverse, about $2n^3$ flops, roughly three times the work, and then multiplies. It is also less accurate, because the rounding errors made while forming every entry of the inverse are carried into the result.

~~~dg-fig 01.07-timing.svg
Measured single-threaded time to solve one system of size n: building the inverse and multiplying is three to four times slower than solve at the larger sizes. Exact timings depend on the machine.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**solving a 2 x 2 system by hand, the way solve does**

System: $2x + y = 3$ and $x + 3y = 5$.

Elimination (the L part): subtract $\tfrac{1}{2}$ of equation 1 from equation 2. The multiplier $\tfrac{1}{2}$ is the single entry of $L$ below the diagonal. Equation 2 becomes $(1 - 1)x + (3 - 0.5)y = 5 - 1.5$, that is $2.5y = 3.5$. Now the system is upper-triangular (the U part).

Back substitution: $y = 3.5 / 2.5 = 1.4$. Then $2x = 3 - 1.4 = 1.6$, so $x = 0.8$.

Check: $2(0.8) + 1.4 = 3$ and $0.8 + 3(1.4) = 0.8 + 4.2 = 5$. Both equations hold.

Inverse route: $\det A = 2 \cdot 3 - 1 \cdot 1 = 5$, so $A^{-1} = \begin{pmatrix} 0.6 & -0.2 \\ -0.2 & 0.4 \end{pmatrix}$ and $A^{-1}b = (1.8 - 1.0,\; -0.6 + 2.0) = (0.8, 1.4)$. Same answer, plus $n^2$ extra numbers we never needed, each with its own rounding error.

Condition number: $A$ is symmetric with eigenvalues $(5 \pm \sqrt 5)/2 = 3.618$ and $1.382$, so $\kappa = 3.618 / 1.382 = 2.618$, matching `np.linalg.cond(A)`.

</div>

```python
B = np.array([[3., 1.], [5., 0.]])     # two right-hand sides as columns
print(np.linalg.solve(A, B))           # [[0.8 0.6] [1.4 -0.2]], one factorisation for both

try:
    np.linalg.solve(np.array([[1., 2.], [2., 4.]]), b)
except np.linalg.LinAlgError as e:
    print(e)                           # Singular matrix

S = np.array([[1., 2.], [2., 4.0000000001]])     # almost singular
print(np.linalg.solve(S, b))           # [ 2.0e+10 -1.0e+10]  no error, huge numbers
print(np.linalg.cond(S))               # 2.5e+11
```

The last two lines are the dangerous case. `solve` raises `LinAlgError` only when a pivot is *exactly* zero. A matrix that is merely nearly singular returns enormous, noise-dominated numbers without complaint. For repeated solves with one matrix, factorise once with SciPy (`lu_factor`/`lu_solve`, or `cho_factor`/`cho_solve`) and reuse it.

~~~dg-flow Pick the routine from the structure of the problem.
Square A, solve Ax = b -(general)-> np.linalg.solve (LU)
Symmetric positive definite A -(faster, stabler)-> *Cholesky (scipy cho_solve)
Tall X, min of the residual norm -(rank-safe)-> np.linalg.lstsq (SVD)
~~~

## Step 3: count your digits with the condition number

The condition number is the ratio of the largest to the smallest **singular value** (the stretch factors of the matrix, defined properly in step 5):

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}.$$

Here $\sigma_{\max}$ is how much $A$ can stretch a vector at most and $\sigma_{\min}$ how much at least. If some direction is squashed almost to zero, $\kappa$ is huge and the matrix is nearly singular.

The rule of thumb: the relative error in the computed $x$ is about $\kappa(A)$ times **machine epsilon** $\varepsilon$, the relative rounding size of the floating-point type ($\varepsilon \approx 2.2 \times 10^{-16}$ for float64, $1.2 \times 10^{-7}$ for float32). In digits:

$$\text{digits lost} \approx \log_{10} \kappa(A).$$

Number check: float64 carries about 16 significant digits. With $\kappa = 10^{10}$ you lose about 10 and can trust about 6; float32 carries about 7, so nothing is left. The nearly parallel lines ($\kappa \approx 4000$) cost about 3.6 digits. The rule is a worst case; real errors are often smaller.

## Step 4: least squares with lstsq

A **tall** matrix $X$ of shape $(n, d)$ with $n > d$ (more examples than features) gives an **overdetermined** system $Xw \approx y$: no $w$ satisfies all $n$ equations, so we ask for the one that minimises the length of the residual, $\lVert Xw - y \rVert_2$. This is linear regression.

```python
size = np.array([30, 42, 48, 55, 61, 68, 75, 83, 90, 104.])
rent = np.array([11, 13.5, 13, 16, 15.5, 18.5, 19, 20, 23.5, 25.])
X = np.c_[np.ones_like(size), size]                 # intercept column + feature, shape (10, 2)

w, res, rank, sv = np.linalg.lstsq(X, rent, rcond=None)
print(w)       # [4.690 0.1953]   intercept and slope
print(res)     # [5.970]          sum of squared residuals
print(rank)    # 2                numerical rank of X
print(sv)      # [218.58 0.995]   singular values, so cond(X) is about 220
```

`lstsq` works through the **singular value decomposition** (SVD), so it copes with redundant columns. Add the size in square feet (`size * 10.76`) as a third column: the rank stays 2 and the third singular value is about $4 \times 10^{-14}$. Singular values below `rcond` times the largest count as zero, and you get the **minimum-norm solution**, the shortest of all equally good weight vectors. Here the two size weights were 0.00167 and 0.0180, and $0.00167 + 10.76 \times 0.0180 = 0.1953$, the original slope. Predictions are unchanged; the individual weights mean nothing. (`res` comes back empty when the rank is deficient.)

The tempting shortcut is the **normal equations**, `np.linalg.solve(X.T @ X, X.T @ y)`. It is faster for very tall $X$, but it squares the condition number: $\kappa(X^\top X) = \kappa(X)^2$. For the rent data that turns 220 into 48,000, harmless. For badly scaled features with $\kappa(X) = 10^5$ it becomes $10^{10}$, and in float32 the answer is noise.

~~~dg-fig 01.07-digits.svg
Measured on matrices built with a chosen condition number: the error of lstsq grows like ε·κ, the error of the normal equations like ε·κ², reaching 100% (pure noise) once κ passes about 10⁸ in float64.
~~~

This was measured: for each $\kappa$ we built a $200 \times 5$ matrix with that condition number and a target with known weights. The normal equations reach 100% error at $\kappa \approx 10^8$, where $\kappa^2 \approx 1/\varepsilon$.

<div class="ex">
<div class="lab">Worked example</div>

**how many digits the normal equations cost on timestamps**

Fit $y = w_0 + w_1 t$ where `t` holds 1,000 Unix timestamps spread over one year, around $1.7 \times 10^9$ seconds. The ones column and the `t` column are nearly parallel, because the timestamps differ by only about 2% of their size. Measured: $\kappa(X) \approx 3.3 \times 10^{11}$.

The normal equations work with $\kappa(X^\top X) = \kappa(X)^2 \approx 10^{23}$, far beyond the $10^{16}$ that float64 can resolve.

In our run, against a reference computed on centred and scaled time, `lstsq` had a relative error of about $8 \times 10^{-13}$ while `solve(X.T @ X, X.T @ y)` was off by about $2 \times 10^{-9}$: more than three extra digits gone. Here the loss was smaller than the worst-case rule predicts, but with less benign data it can be total. In float32 neither result is usable.

The real fix is upstream: subtract the mean of `t` and divide by its standard deviation before fitting. Then $\kappa$ is close to 1 and every method agrees.

</div>

Normal equations are fine when $X$ is well conditioned (standardised features, no near-duplicates), or with a ridge term, $(X^\top X + \lambda I)w = X^\top y$, which lifts the smallest eigenvalue to at least $\lambda$ and caps the condition number.

## Step 5: eigh and svd

An **eigenvector** of a square matrix $S$ is a direction that $S$ only stretches, never turns: $Sv = \lambda v$, where the number $\lambda$ is its **eigenvalue**. For a covariance matrix the eigenvectors are perpendicular axes of the data cloud and each eigenvalue is the variance along its axis. That is principal component analysis (PCA).

```python
C = np.array([[3., 1.6], [1.6, 1.2]])     # a 2 x 2 covariance matrix
vals, vecs = np.linalg.eigh(C)
print(vals)            # [0.264 3.936]   ascending!
print(vecs[:, -1])     # [-0.863 -0.505] main axis = LAST column
print(np.linalg.eig(C).eigenvalues)   # [3.936+0.j 0.264+0.j]  complex dtype, own order
```

- `np.linalg.eigh(S)` is for symmetric (or Hermitian) matrices. It returns real eigenvalues in **ascending** order and orthonormal eigenvectors as **columns**. For PCA you usually want the last columns.
- `np.linalg.eig` handles general matrices; on the same input it returned a complex dtype in another order. Avoid it for covariance matrices.

The **singular value decomposition** factors any matrix, square or not, as $X = U\,\mathrm{diag}(s)\,V^\top$. Here $V^\top$ (`Vt`) holds orthonormal directions in feature space as rows, `s` holds the non-negative **singular values** (the stretch along each direction) in **descending** order, and `U` holds the matching directions in example space as columns. For PCA you centre $X$ and take the SVD directly, without forming the covariance matrix:

```python
rng = np.random.default_rng(0)
D = rng.normal(size=(500, 3)) @ np.array([[3., 0, 0], [1, 1, 0], [0, 0, .2]])
Dc = D - D.mean(axis=0)                                   # centre first
U, s, Vt = np.linalg.svd(Dc, full_matrices=False)
print(U.shape, s.shape, Vt.shape)                         # (500, 3) (3,) (3, 3)
print(s**2 / (len(D) - 1))                                # [9.809 0.808 0.042]  variances, descending
print(np.linalg.eigh(np.cov(Dc, rowvar=False))[0])        # [0.042 0.808 9.809]  same, ascending
print(np.allclose(Dc, U * s @ Vt))                        # True
```

The principal directions are the first rows of `Vt`, and the explained variance is `s**2 / (n - 1)`. Always pass `full_matrices=False`: the default full form allocates an $n \times n$ matrix `U`, which for 100,000 rows is 80 GB of float64.

~~~dg-fig 01.07-pca.svg
A correlated two-feature cloud and its principal axes. eigh gives the main axis as its last column; svd gives the same axis as the first row of Vt, here with the opposite sign. Both are correct.
~~~

Eigenvectors and singular vectors are defined only **up to sign**: if $v$ is an eigenvector, so is $-v$. In the figure, `eigh` and `svd` found the same axis pointing in opposite directions. When a test compares vectors, or a pipeline must give identical projections across runs, fix the sign, for example so that the largest-magnitude entry of each vector is positive. With repeated eigenvalues the vectors are only defined up to a rotation within that subspace, so compare the subspace, not the vectors.

## Step 6: random numbers you can reproduce

NumPy random numbers are **pseudo-random**: a deterministic algorithm (the *bit generator*) produces a stream that looks random, and the **seed** picks the starting point. Same seed, same stream.

The legacy API (`np.random.seed`, `np.random.rand`) keeps one hidden global state. Any library that draws from it shifts your sequence, and processes forked from one parent, such as data-loader workers, inherit identical state and produce identical "random" augmentations. The modern API makes the state an explicit object:

```python
rng = np.random.default_rng(42)          # a Generator using the PCG64 bit generator
print(rng.normal(size=3))                # [ 0.305 -1.040  0.750]
print(rng.integers(0, 10, size=5))       # [0 6 2 0 5]   upper bound excluded
print(rng.permutation(5))                # [2 4 0 3 1]
print(rng.choice(['a', 'b', 'c'], size=4))   # ['b' 'a' 'c' 'b']

workers = rng.spawn(8)                   # 8 independent child generators
```

Pass `rng` into functions as an argument instead of seeding inside them.

~~~dg-fig 01.07-rng.svg
Left: two workers seeded with the same number draw identical values, so their lines coincide. Right: two children from rng.spawn draw different, statistically independent values.
~~~

For parallel work, give each worker its own child stream from `rng.spawn(n)` or `np.random.SeedSequence(42).spawn(n)`. These children are designed to be statistically independent. Seeding workers with `default_rng(seed + i)` usually works in practice but carries no such guarantee.

One caveat from NumPy's random policy (NEP 19): the stream that a `Generator` produces for a given seed may change between NumPy versions when algorithms are improved, while the legacy `RandomState` stream is frozen. Exact bit-for-bit reproducibility therefore also requires pinning the NumPy version.

## Checking your results

- **Residual check.** `np.linalg.norm(A @ x - b) / np.linalg.norm(b)` near $10^{-16}$ means the solver did its job, not that `x` is accurate when $\kappa$ is large.
- **Condition check.** Print `np.linalg.cond(A)`; subtract $\log_{10}\kappa$ from 16 (float64) or 7 (float32) for the digits you can trust.
- **Rank check.** For least squares, compare the `rank` from `lstsq` with the number of columns. A smaller rank means redundant features.
- **Reconstruction check.** For factorisations, `np.allclose(Dc, U * s @ Vt)` should be `True`.
- **Reproducibility check.** Same seed twice gives identical outputs; a new seed changes them.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Shape is right but values are wrong | used `*` where `@` was meant (or the reverse) | check which product you need; `@` contracts, `*` is elementwise
| 4-D result from two 3-D arrays | `np.dot` on stacked matrices | use `@` or `einsum`
| Huge weights with opposite signs | near-collinear features, solving the normal equations | `lstsq`, drop duplicates, standardise, add ridge
| `solve` returns 1e10-sized numbers, no error | nearly singular matrix | check `np.linalg.cond`; rescale or regularise
| Out-of-memory in `svd` | `full_matrices=True` on a tall matrix | `full_matrices=False`
| PCA component flips sign between runs | sign ambiguity of eigenvectors | fix the sign convention before comparing
| Wrong principal component picked | forgot `eigh` sorts ascending | take `vecs[:, ::-1]` or use `svd`
| All data-loader workers augment identically | shared global or copied seed | `rng.spawn` one child per worker

## Going deeper

**Backward stability.** LU with partial pivoting, the QR and SVD algorithms used by `lstsq`, and symmetric eigensolvers are *backward stable*: the computed answer is the exact answer to a problem whose inputs differ from yours by a few units of $\varepsilon$. The condition number then converts that input-sized error into an output error, which is where "relative error $\approx \kappa\varepsilon$" comes from. The explicit inverse is not backward stable for solving, which is the precise reason to avoid it.

**Why the normal equations square $\kappa$.** If $X = U\,\mathrm{diag}(s)\,V^\top$, then $X^\top X = V\,\mathrm{diag}(s^2)\,V^\top$. Its eigenvalues are $s_i^2$, so its condition number is $\kappa(X)^2$. The ridge term adds $\lambda$ to every eigenvalue, giving $\kappa = (\sigma_{\max}^2 + \lambda)/(\sigma_{\min}^2 + \lambda)$.

**Minimum-norm solution.** Writing $X^+ = V\,\mathrm{diag}(1/s_i)\,U^\top$ with the reciprocals of tiny singular values replaced by zero gives the **pseudo-inverse**; `lstsq` returns $X^+ y$ and `np.linalg.pinv` builds $X^+$ explicitly. Ridge is the smooth version of dropping small $s_i$: it scales each component by $s_i^2/(s_i^2 + \lambda)$.

**Random streams.** `SeedSequence` hashes your seed (and a spawn key) into a well-mixed initial state, so seeds 0, 1, 2 do not produce correlated PCG64 streams, and children of children stay distinct. Neither generator is cryptographically secure; use the `secrets` module for keys and tokens.

## Key terms

| Term | Meaning
| --- | --- |
| Matrix product (`@`) | Rows of the left matrix dotted with columns of the right; leading axes broadcast
| BLAS / LAPACK | Optimised libraries that NumPy calls for products, solves and factorisations
| LU factorisation | Writing $A$ as lower times upper triangular (with row swaps) to solve $Ax = b$
| Condition number $\kappa$ | Largest over smallest singular value; about $\log_{10}\kappa$ digits are lost
| Machine epsilon | Relative rounding size: about $2.2 \times 10^{-16}$ (float64), $1.2 \times 10^{-7}$ (float32)
| Normal equations | $X^\top X w = X^\top y$; correct but square the condition number
| Minimum-norm solution | The shortest weight vector among equally good least-squares fits
| Eigen / singular vector | A direction a matrix only stretches; defined only up to sign
| Generator | An explicit random number object from `default_rng(seed)`, passed around in code
| Spawn | Deriving independent child generators for parallel workers

<div class="take">
<div class="lab">Key takeaway</div>

use `@` for products, `solve` instead of `inv`, `lstsq` instead of the normal equations, `eigh` and `svd(full_matrices=False)` for factorisations, check the condition number when answers look odd, and thread one explicit `Generator` (spawning children for workers) through your code.

</div>
