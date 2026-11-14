---
id: "14.05"
course: "ml-maths-course"
module: "14"
slug: "second-order-and-quasi-newton-methods"
title: "Second-order and quasi-Newton methods"
keyIdeas: ["Newton's method fits a parabola (the second-order Taylor model) to the loss and jumps to its bottom: $p = -H^{-1}\\nabla f$. It is **affine-invariant**, so ill-conditioning does not slow it, and near a minimum it converges **quadratically** (the number of correct digits doubles each step).", "Its costs are $O(d^2)$ memory and $O(d^3)$ time per step, and if the Hessian has negative eigenvalues it heads for saddles or maxima, so it needs damping ($H + \\lambda I$) or a trust region on non-convex problems.", "A **line search** picks the step length: backtracking halves $t$ from 1 until the Armijo sufficient-decrease condition holds; quasi-Newton methods also need the Wolfe curvature condition.", "**BFGS** builds an inverse-Hessian approximation from gradient differences using the secant condition $H_{k+1}s_k = y_k$; **L-BFGS** keeps only the last $m$ pairs, so it needs $O(md)$ memory instead of $O(d^2)$.", "L-BFGS is the default for smooth, deterministic, moderate-size problems (scikit-learn's logistic regression, SciPy fitting); mini-batch noise corrupts its curvature pairs, so deep learning mostly uses first-order methods or structured curvature such as K-FAC."]
question: "You need to fit an L2-regularised logistic regression with 50,000 sparse features on 2 million rows that fit in memory. Compare plain gradient descent, Newton's method and L-BFGS for this job: estimate the memory and per-iteration cost of each, say roughly how many iterations each would need, and say which you would use and how you would check that it converged."
answer: "Gradient descent is cheap per iteration, one pass over the data for the gradient and $O(d)$ memory, but logistic-regression Hessians with many correlated sparse features are badly conditioned, so it needs thousands of passes and a carefully tuned step (14.01: iterations grow with $\\kappa$). Newton's method would converge in perhaps 10 to 20 iterations, accepting $t = 1$ near the end with the gradient norm's exponent doubling, but each iteration needs the 50,000 by 50,000 Hessian: 2.5 billion entries (20 GB in fp64) plus a Cholesky factorisation of about $d^3/3 \\approx 4\\times10^{13}$ flops, which is impractical; the workable form is truncated Newton (Newton-CG), which solves $Hp = -\\nabla f$ approximately with conjugate gradients using Hessian-vector products, each costing about one pass over the sparse data. L-BFGS uses only full gradients, which are exact because the data fit in memory, keeps about $m = 10$ curvature pairs (2md = 1 million numbers, 8 MB), uses a Wolfe line search so no step size has to be tuned, and typically converges in tens to a few hundred passes. I would use L-BFGS, which is why scikit-learn makes it the default; the L2 penalty makes the problem strongly convex, so there is one minimum and the Hessian is never singular. To check convergence I would watch the gradient norm fall to the solver tolerance, confirm that late iterations accept the full step (nfev close to nit), check that the solver did not stop on its iteration limit (scikit-learn warns with ConvergenceWarning; raise max_iter or standardise features if it does), and compare validation loss with a run at a tighter tolerance. If an L1 penalty were required, the objective would be non-smooth and a proximal or coordinate method (14.06) would replace L-BFGS."
sources: [{"title": "Boyd and Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Chapter 9.5: Newton's method, affine invariance and its convergence analysis; 9.2 covers backtracking."}, {"title": "Tibshirani: Convex Optimization lecture notes, quasi-Newton methods", "url": "https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/quasi-newton.pdf", "note": "Secant condition, BFGS update and L-BFGS two-loop recursion."}, {"title": "SciPy: minimize(method='L-BFGS-B')", "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html", "note": "Parameters such as maxcor (m) and the stopping tolerances in practice."}, {"title": "Martens: New insights and perspectives on the natural gradient method", "url": "https://arxiv.org/abs/1412.1193", "note": "How Gauss-Newton and natural gradient relate, the basis for K-FAC-style deep-learning optimisers."}]
practice: ["newton-s-method", "l-bfgs", "hessian-vector-product", "classify-critical-point-from-hessian"]
tags: ["Newton's method", "L-BFGS", "line search"]
tools: ["NumPy", "SciPy"]
---

## What you will learn

- Why using the curvature (the Hessian) lets an optimiser pick both a good direction and a good step length, so ill-conditioning stops mattering.
- How Newton's method works: fit a parabola to the loss at the current point and jump to its bottom.
- What "quadratic convergence" means in practice (the number of correct digits doubles each step), and why Newton is still rarely used at scale.
- How a backtracking line search with the Armijo and Wolfe conditions decides how far to move along a direction.
- How BFGS and L-BFGS learn the curvature from gradients alone, and when L-BFGS is the right tool in ML.

*Before you start:* you need gradients and the Hessian matrix (ML Maths module 09, especially the second-order Taylor expansion in 09.04), gradient descent and the condition number $\kappa$ (14.01), and the diagonal scaling idea behind Adam (14.03).

## Using curvature directly

Lessons 14.01 to 14.04 used the slope alone, and paid for curvature with iterations: about $\kappa$ steps for gradient descent, $\sqrt\kappa$ with momentum. Adam (14.03) uses a crude diagonal curvature estimate; the schedules of 14.04 tune one global step size. **Second-order methods** use the Hessian, or an approximation of it, to choose both the direction and the length of every step, and on problems where they fit in memory they finish in a handful of iterations what gradient descent does in thousands.

## The idea in one picture

Take a one-variable function with a known answer:

$$f(x) = e^x - 2x, \qquad f'(x) = e^x - 2, \qquad f''(x) = e^x.$$

The minimum is where $e^x = 2$, at $x^\star = \ln 2 = 0.693147\ldots$ Start at $x_0 = 0$. There the value is 1, the slope is $-1$ and the curvature is 1. Those three numbers define a parabola that touches $f$ at $x_0$ and bends the same way. Its bottom is at $x_1 = 1$, so we jump there, build a new parabola, and repeat.

| step $k$ | $x_k$ | distance to $\ln 2$
| --- | --- | --- |
| 0 | 0 | 0.69
| 1 | 1 | 0.31
| 2 | 0.735759 | 0.043
| 3 | 0.694042 | 0.00090
| 4 | 0.6931476 | 0.0000004
| 5 | 0.69314718056 | $8 \times 10^{-14}$

~~~dg-fig 14.05-quadmodel.svg
Left: each Newton step replaces the curve by its touching parabola (dashed) and moves to that parabola's lowest point. Right: Newton's error collapses to machine precision in five steps; gradient descent with step 0.2 shrinks it by the same factor (about 0.6) every step.
~~~

Read the last column: 0.31, 0.043, 0.0009, 0.0000004. Each error is roughly the square of the one before (times about one half). The number of correct digits goes 1, 3, 6, 13: it roughly doubles once the method is close. This is **quadratic convergence**, and it is the reason second-order methods exist. Gradient descent on the same function removes a fixed fraction of the error per step (**linear convergence**), which shows up as a straight line on the log plot.

## Step 1: the local quadratic model

In many dimensions the touching parabola becomes the second-order Taylor model at the current point $x_k$:

$$f(x_k + p) \approx m(p) = f(x_k) + \nabla f(x_k)^\top p + \tfrac12\, p^\top H p.$$

Here $p$ is the step (a vector the size of the parameters), $\nabla f(x_k)$ the gradient, and $H = \nabla^2 f(x_k)$ the **Hessian**: a symmetric $d \times d$ matrix whose entry $(i, j)$ says how the slope in direction $i$ changes as you move in direction $j$. The first two terms are the tangent plane gradient descent uses; the third adds the bend.

If $H$ is **positive definite** (every eigenvalue is positive, so the model is a bowl), the model has a single minimum. Setting the gradient of $m$ to zero gives $\nabla f + Hp = 0$.

## Step 2: Newton's step

Solving that equation gives the **Newton step**

$$p = -H^{-1}\nabla f(x_k), \qquad x_{k+1} = x_k + p.$$

In one dimension this is $x_{k+1} = x_k - f'(x_k)/f''(x_k)$: slope divided by curvature. Strongly curved directions get short steps and flat ones long steps, which one learning rate cannot do.

Number check on the stretched bowl from 14.03, $f(w) = \tfrac12(20 w_1^2 + 0.2 w_2^2)$, starting at $(1, 1)$. The gradient is $(20, 0.2)$ and the Hessian is diagonal with entries 20 and 0.2. The Newton step is $(-20/20, -0.2/0.2) = (-1, -1)$, which lands exactly on the minimum $(0, 0)$. Gradient descent with its largest safe rate, 0.09, moves to $(1 - 1.8,\ 1 - 0.018) = (-0.8, 0.982)$: it overshoots the steep axis and barely moves along the flat one.

In code never form $H^{-1}$: solve $Hp = -\nabla f$ (`np.linalg.solve`, or a Cholesky factorisation), which is cheaper and more accurate.

<div class="ex">
<div class="lab">Worked example</div>

**Newton's method by hand on $f(x) = e^x - 2x$**

The update is $x \leftarrow x - \dfrac{e^x - 2}{e^x} = x - 1 + 2e^{-x}$.

Step 1, from $x_0 = 0$: $e^{0} = 1$, so $x_1 = 0 - 1 + 2 \times 1 = 1$. Error $1 - 0.693147 = 0.306853$.

Step 2: $e^{-1} = 0.367879$, so $x_2 = 1 - 1 + 0.735759 = 0.735759$. Error 0.042612.

Step 3: $e^{-0.735759} = 0.479142$, so $x_3 = 0.735759 - 1 + 0.958283 = 0.694042$. Error 0.000895.

Step 4: $e^{-0.694042} = 0.499553$, so $x_4 = 0.694042 - 1 + 0.999106 = 0.693148$ (more digits: 0.6931476). Error $4.0 \times 10^{-7}$.

Check the quadratic pattern: $0.3069^2 / 2 = 0.047$ (actual 0.043), $0.0426^2/2 = 0.00091$ (actual 0.00090), $0.000895^2/2 = 4.0 \times 10^{-7}$ (actual $4.0 \times 10^{-7}$). For this function the constant is $f'''/(2f'') = 1/2$ at the minimum, which is why "half the square" fits so well.

</div>

## Step 3: why Newton ignores ill-conditioning

Newton's method is **affine-invariant**: stretch or rotate the variables ($x = Ay$, $A$ invertible), run Newton on $y$, and you get the same points mapped through $A$. Gradient descent depends on the units of every feature, which is why 14.01 told you to standardise.

So the condition number that ruled 14.01 drops out. On any exact quadratic Newton lands on the minimum in one step, whether $\kappa$ is 1 or $10^6$. On a non-quadratic function the model is only approximate, and two regimes appear:

- **Far from the minimum**, the parabola can be a poor fit, so the full step can overshoot or even increase the loss. A line search (Step 4) shortens it.
- **Near a minimum** with positive-definite Hessian, the full step $t = 1$ is accepted and the error obeys $\lVert e_{k+1}\rVert \le C\lVert e_k\rVert^2$.

~~~dg-fig 14.05-rosenbrock.svg
The Rosenbrock function has a curved, narrow valley. Gradient descent (with a backtracking line search) zig-zags along the floor and is still at a loss of about 1e-6 after 5000 steps; Newton reaches 1e-8 at step 20 and BFGS at step 31, using the same start.
~~~

## Step 4: line search, choosing how far to go

A method like Newton or BFGS gives a **descent direction** $p$, one with $\nabla f^\top p < 0$ (the loss decreases if you move a little along it). A **line search** picks the step length $t > 0$ in $x_{k+1} = x_k + t p$; rather than minimise exactly along the line, it accepts any $t$ passing two tests.

1. **Armijo (sufficient decrease):** $f(x + tp) \le f(x) + c_1 t\, \nabla f(x)^\top p$, with $c_1$ small, typically $10^{-4}$. The right side is a line starting at the current loss with a slope that is a small fraction of the true initial slope. The step must land below it, so it must give at least a little of the decrease the slope promised.
2. **Curvature (Wolfe):** $\nabla f(x + tp)^\top p \ge c_2\, \nabla f(x)^\top p$, with $c_2 = 0.9$ for quasi-Newton methods. The slope at the new point must be noticeably less negative than at the start. This rules out steps so short that nothing about the curvature has been learned.

**Backtracking** is the simplest way to satisfy Armijo: try $t = 1$ (the natural Newton step), and halve it until the test passes.

~~~dg-fig 14.05-armijo.svg
Loss along a unit gradient direction on Rosenbrock, starting from (-1.2, 1). Trials t = 1 and t = 0.5 land above the Armijo line and are rejected; t = 0.25 lands below it and is accepted. The shaded band is every step the test allows.
~~~

The numbers behind the figure: the loss at the start is 24.2 and the directional slope is $-232.9$. Trying $t = 1$ gives 171.3, far above the line, and $t = 0.5$ gives 44.7, still above it. At $t = 0.25$ the loss is 6.3, below the Armijo bound $24.2 - 0.1 \times 0.25 \times 232.9 = 18.4$ (with $c_1 = 0.1$ for the picture), so it is accepted.

```python
def backtracking(f, x, fx, g, p, t=1.0, c1=1e-4, shrink=0.5):
    slope = g @ p                       # must be negative
    while f(x + t * p) > fx + c1 * t * slope:
        t *= shrink
    return t
```

## Step 5: quasi-Newton, learning curvature from gradients

Newton needs the Hessian, which is expensive. **Quasi-Newton** methods never compute it. They watch how the gradient changes between steps and build an approximation from that.

After a step $s_k = x_{k+1} - x_k$, the gradient changes by $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$. For a quadratic, $y_k = H s_k$ exactly; for a smooth function, approximately. Asking the new approximation $H_{k+1}$ to reproduce this is the **secant condition**

$$H_{k+1}\, s_k = y_k.$$

In one dimension it fixes the curvature estimate completely: $H_{k+1} = y_k / s_k$, the slope of the derivative between two points. That is the classic secant method.

<div class="ex">
<div class="lab">Worked example</div>

**one quasi-Newton step on $f(x) = e^x - 2x$, curvature from two gradients**

Suppose we have visited $x_0 = 0$ and $x_1 = 1$. The derivatives are $f'(0) = 1 - 2 = -1$ and $f'(1) = e - 2 = 0.718282$.

Step and gradient change: $s = 1 - 0 = 1$, $y = 0.718282 - (-1) = 1.718282$. Curvature estimate $B = y/s = 1.718282$ (the true values are 1 at $x_0$ and 2.718 at $x_1$; the estimate sits between).

Quasi-Newton step: $x_2 = 1 - 0.718282 / 1.718282 = 1 - 0.418023 = 0.581977$. Error 0.111.

Continuing the same rule gives errors 0.016, 0.00093, $7.7 \times 10^{-6}$, $3.6 \times 10^{-9}$: faster than linear but slower than Newton's doubling. This is **superlinear** convergence, bought without a single second derivative.

</div>

In $d$ dimensions one equation cannot pin down a whole matrix, so **BFGS** (Broyden, Fletcher, Goldfarb, Shanno) chooses the update closest to the previous approximation that satisfies the secant condition. It works directly with an inverse-Hessian approximation $B \approx H^{-1}$, so the direction is just $p = -B\nabla f$ and no linear system is solved. The update is a rank-two correction:

$$B_{k+1} = (I - \rho_k s_k y_k^\top)\, B_k\, (I - \rho_k y_k s_k^\top) + \rho_k s_k s_k^\top, \qquad \rho_k = \frac{1}{y_k^\top s_k}.$$

It keeps $B$ symmetric and positive definite as long as $y_k^\top s_k > 0$, which says the slope increased along the step, and the Wolfe curvature condition guarantees exactly that. This is why BFGS is paired with a Wolfe line search. BFGS converges superlinearly, costs $O(d^2)$ per step, and still stores a dense $d \times d$ matrix.

## Step 6: L-BFGS, the version that scales

**L-BFGS** (limited-memory BFGS) does not store $B$ at all. It keeps only the last $m$ pairs $(s_i, y_i)$, typically $m = 5$ to 20, and computes $B\nabla f$ from them with the **two-loop recursion** in about $4md$ operations. Older pairs are dropped, so curvature information from far away (and probably stale) never accumulates.

~~~dg-flow One L-BFGS iteration: curvature comes only from recent gradient differences, never from a stored Hessian.
Gradient g_k -(two-loop recursion with m pairs)-> Direction p = -B g -(Wolfe line search)-> *Step x_k+1 -(store s, y; drop oldest)-> Updated memory
~~~

~~~dg-fig 14.05-memory.svg
Bytes needed in fp64 (at d = 10^6 and m = 10: 20 million numbers versus 10^12). A dense Hessian (Newton, BFGS) grows as d² and passes 16 GB at about 45 000 parameters; L-BFGS with m = 10 stores 2md numbers, only 20 times the weights themselves.
~~~

| Method | Memory | Cost per step | Local rate | Needs
| --- | --- | --- | --- | --- |
| Gradient descent | $O(d)$ | one gradient | linear, depends on $\kappa$ | step size
| Newton | $O(d^2)$ | gradient + Hessian + $O(d^3)$ solve | quadratic | convexity or safeguards
| BFGS | $O(d^2)$ | gradient + $O(d^2)$ | superlinear | Wolfe line search
| L-BFGS | $O(md)$ | gradient + $O(md)$ | linear, much better constant | deterministic gradients

## Second-order methods in code

Here is Newton with backtracking on L2-regularised logistic regression (2000 synthetic rows, 20 features), next to SciPy's L-BFGS and scikit-learn, which uses L-BFGS by default.

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=2000, n_features=20, random_state=0)
y = 2 * y - 1                          # labels in {-1, +1}
lam, (n, d) = 1e-2, X.shape            # L2 strength, data shape

def loss(w):
    return np.mean(np.logaddexp(0, -y * (X @ w))) + 0.5 * lam * w @ w
def grad(w):
    s = -y / (1 + np.exp(y * (X @ w)))
    return X.T @ s / n + lam * w
def hess(w):
    p = 1 / (1 + np.exp(-(X @ w)))
    return (X.T * (p * (1 - p))) @ X / n + lam * np.eye(d)

w = np.zeros(d)                        # Newton + backtracking
for it in range(20):
    g = grad(w)
    if np.linalg.norm(g) < 1e-10:
        break
    p = -np.linalg.solve(hess(w), g)   # solve, never invert
    t = 1.0
    while loss(w + t * p) > loss(w) + 1e-4 * t * (g @ p):
        t *= 0.5
    w = w + t * p
    print(it, f"{loss(w):.10f}", f"{np.linalg.norm(grad(w)):.1e}", t)
# 0 0.2201700499 1.5e-01 1.0
# 1 0.1488446956 4.5e-02 1.0
# 2 0.1357947075 9.1e-03 1.0
# 3 0.1350235840 6.1e-04 1.0
# 4 0.1350198819 3.2e-06 1.0
# 5 0.1350198818 8.7e-11 1.0

res = minimize(loss, np.zeros(d), jac=grad, method="L-BFGS-B",
               options=dict(maxcor=10, gtol=1e-8))   # maxcor is m
print(res.nit, res.nfev, f"{res.fun:.10f}")          # 9 10 0.1350198820

clf = LogisticRegression(C=1 / (n * lam), fit_intercept=False).fit(X, (y + 1) // 2)
print(clf.solver, clf.n_iter_, np.abs(clf.coef_.ravel() - w).max())
# lbfgs [8] 8.6e-05   (agrees with Newton to its default tolerance)
```

Look at the gradient norms: $1.5 \times 10^{-1}$, $4.5 \times 10^{-2}$, $9 \times 10^{-3}$, $6 \times 10^{-4}$, $3 \times 10^{-6}$, $9 \times 10^{-11}$. The exponent roughly doubles at the end, and every step accepted $t = 1$: Newton was in its quadratic regime from the start. In scikit-learn, `C` is the inverse of the total penalty, hence $C = 1/(n\lambda)$ to match our averaged loss.

## Reading the result

How do you tell whether a second-order run is healthy?

- **Gradient norm.** It should fall steadily; for Newton its exponent should roughly double at the end. A stall around $10^{-6}$ usually means an inaccurate gradient: check it with finite differences.
- **Accepted step length.** Near the solution $t = 1$ should be accepted almost every time; late backtracking means the curvature model is wrong (a Hessian bug or a non-smooth loss).
- **Function evaluations versus iterations.** SciPy reports both (`nit`, `nfev`); a ratio near 1 means the line search rarely backtracks.
- **Iterations versus cost.** Count work, not just iterations. On the harder problem below, Newton needed 4 iterations to get within $10^{-8}$ of the best loss and L-BFGS 176, but each Newton iteration builds and factorises a Hessian, while each L-BFGS iteration costs about one gradient.

~~~dg-fig 14.05-logreg.svg
A deliberately ill-conditioned logistic regression (feature scales from 0.2 to 5, Hessian condition number about 4 × 10^5 at the optimum). Gradient descent with step 1/L is still 2 × 10^-3 above the best loss after 2000 iterations; L-BFGS reaches 10^-8 at iteration 176 and Newton at iteration 4.
~~~

## What goes wrong

The most common failure is non-convexity. Newton's step goes to the stationary point of the local model, and if $H$ has a negative eigenvalue that point is a maximum or saddle of the model. Newton is attracted to *any* point where the gradient is zero.

~~~dg-fig 14.05-saddle.svg
On the double well f(x) = x⁴/4 − x²/2, plain Newton from x = 0.4 jumps to −0.25, then 0.036, and settles on the maximum at 0, where the curvature is negative. Adding 1.5 to the curvature whenever it is not safely positive (damping) keeps every step downhill and reaches the minimum at 1.
~~~

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss goes up after a full Newton step | far from the minimum, model overshoots | backtracking line search
| Converges to a point with a non-positive-definite Hessian | negative curvature, attracted to saddle or maximum | damping $H + \lambda I$, trust region, check eigenvalues
| `LinAlgError: singular matrix` | collinear features, no regularisation | add an L2 penalty, or damp the Hessian
| L-BFGS wanders on mini-batches | noisy gradient differences corrupt the $(s, y)$ pairs | full-batch gradients, or switch to Adam or SGD
| Out of memory | dense $d \times d$ Hessian | L-BFGS, or Hessian-vector products with conjugate gradients

Standard repairs: **damping** ($H + \lambda I$, as in Levenberg-Marquardt), **trust regions** (trust the model only inside a radius that grows or shrinks with how well it predicted the decrease), and **Gauss-Newton**, which for least-squares losses uses $J^\top J$ ($J$ the Jacobian of the residuals), positive semi-definite by construction.

## Where these methods are used in ML

L-BFGS is the default solver of scikit-learn's `LogisticRegression` and a standard choice in SciPy (`minimize(method="L-BFGS-B")`, which also handles box constraints) for full-batch problems with up to millions of parameters: GLMs, CRFs, Gaussian-process hyperparameters, scientific curve fitting.

It breaks down on mini-batches: the gradient difference $y_k$ mixes real curvature with the noise between two batches, and the line search assumes $f$ can be evaluated exactly. Deep learning instead uses structured curvature: Hessian-vector products, Gauss-Newton and natural-gradient methods such as K-FAC, and Shampoo-style preconditioners, which sit between Adam's diagonal and full Newton.

## Going deeper

**Hessian-vector products.** You rarely need $H$ itself, only $Hv$ for some vector $v$. Automatic differentiation computes $Hv$ as the gradient of $\nabla f(x)^\top v$, for a small constant multiple of one gradient's cost (forward-over-reverse mode in JAX: `jax.jvp(jax.grad(f), (x,), (v,))`). **Truncated Newton** (Newton-CG) solves $Hp = -\nabla f$ approximately with a few conjugate-gradient iterations, each needing one $Hv$, so it gets Newton-like directions in $O(d)$ memory.

**Why quadratic?** Write $e_k = x_k - x^\star$. In one dimension, Taylor expansion of $f'$ around $x^\star$ gives $e_{k+1} = \frac{f'''(\xi)}{2 f''(x_k)} e_k^2$ for some $\xi$ between $x_k$ and $x^\star$. The constant depends on how fast the curvature changes (the third derivative, or a Lipschitz constant of $H$ in many dimensions), which is why the guarantee is only local: far away, $e_k^2$ is not small.

**Convergence classes.** Linear: $\lVert e_{k+1}\rVert \le c\lVert e_k\rVert$, $c < 1$. Superlinear: $\lVert e_{k+1}\rVert / \lVert e_k\rVert \to 0$ (BFGS; the 1-D secant method has order 1.618). Quadratic: Newton. L-BFGS is formally linear, with a far better constant than gradient descent.

**Cost of exact Newton.** $d^2$ numbers and about $d^3/3$ flops per solve: at $d = 10^4$, 800 MB in fp64 and $3 \times 10^{11}$ flops per step; at $d = 10^9$, impossible.

## Key terms

| Term | Meaning
| --- | --- |
| Hessian | Matrix of second derivatives; describes how the slope changes in every direction
| Newton step | $p = -H^{-1}\nabla f$, the minimiser of the local quadratic model
| Affine invariance | Rescaling or rotating the variables does not change the method's path
| Quadratic convergence | The error is squared each step, so correct digits roughly double
| Line search | Choosing the step length along a given direction
| Armijo / Wolfe conditions | Accept a step only if it decreases the loss enough and the slope has flattened enough
| Secant condition | $H_{k+1} s_k = y_k$: the curvature estimate must reproduce the observed gradient change
| BFGS / L-BFGS | Quasi-Newton methods that build curvature from gradient differences; L-BFGS keeps only the last $m$ pairs
| Damping / trust region | Safeguards that keep Newton-type steps downhill on non-convex problems

<div class="take">
<div class="lab">Key takeaway</div>

Newton's method removes ill-conditioning by stepping with the inverse Hessian and converges quadratically, but costs $O(d^3)$ and needs safeguards on non-convex problems; L-BFGS with a Wolfe line search gets most of the benefit in $O(md)$ memory and is the tool of choice for smooth, full-batch problems.

</div>
