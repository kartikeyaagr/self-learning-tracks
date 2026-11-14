---
id: "14.05"
course: "ml-maths-course"
module: "14"
slug: "second-order-and-quasi-newton-methods"
title: "Second-order and quasi-Newton methods"
keyIdeas: ["Newton's method steps $-H^{-1}\\nabla f$, the exact minimiser of the local quadratic model; it is affine-invariant and converges quadratically near a minimum (the number of correct digits doubles each step).", "Its costs are $O(d^2)$ memory and $O(d^3)$ time per step, and on non-convex functions an indefinite Hessian can send it towards saddles or maxima, so it needs damping or a trust region.", "Line searches pick the step length: backtracking until the Armijo sufficient-decrease condition holds, plus the Wolfe curvature condition for quasi-Newton methods.", "BFGS builds an inverse-Hessian approximation from gradient differences via the secant condition $H_{k+1}s_k = y_k$; L-BFGS keeps only the last $m$ pairs ($O(md)$ memory).", "L-BFGS is the default for smooth, deterministic, moderate-size problems (logistic regression, scientific fitting); mini-batch noise breaks its curvature pairs, so deep learning mostly uses first-order methods."]
question: "You need to fit a logistic regression with 50,000 sparse features on 2 million rows that fit in memory. Compare plain gradient descent, Newton's method and L-BFGS for this job and say which you would use."
answer: "Gradient descent is cheap per iteration, one pass over the data, but logistic-regression Hessians with many correlated sparse features are badly conditioned, so it needs thousands of passes and a carefully tuned step. Newton's method would converge in perhaps 10 to 20 iterations, but each needs the 50,000 by 50,000 Hessian: 2.5 billion entries (20 GB in fp64) and a factorisation of about $4\\times10^{13}$ flops, which is impractical (an iterative solver with Hessian-vector products, truncated Newton, would be the workable form). L-BFGS uses only full gradients, which are exact because the data fit in memory, keeps about $m = 10$ curvature pairs (1 million numbers), uses a Wolfe line search, and typically converges in tens to a few hundred passes without step-size tuning. I would use L-BFGS, which is why scikit-learn makes it the default, with an L2 penalty to make the problem strongly convex; if an L1 penalty were required, a proximal or coordinate method (14.06) would replace it."
sources: [{"title": "Boyd and Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Chapter 9.5: Newton's method, affine invariance and its convergence analysis; 9.2 covers backtracking."}, {"title": "Tibshirani: Convex Optimization lecture notes, quasi-Newton methods", "url": "https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/quasi-newton.pdf", "note": "Secant condition, BFGS update and L-BFGS two-loop recursion."}, {"title": "SciPy: minimize(method='L-BFGS-B')", "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html", "note": "Parameters such as maxcor (m) and the stopping tolerances in practice."}, {"title": "Martens: New insights and perspectives on the natural gradient method", "url": "https://arxiv.org/abs/1412.1193", "note": "How Gauss-Newton and natural gradient relate, the basis for K-FAC-style deep-learning optimisers."}]
practice: ["newton-s-method", "l-bfgs", "hessian-vector-product", "classify-critical-point-from-hessian"]
tags: ["Newton's method", "L-BFGS", "line search"]
tools: ["NumPy", "SciPy"]
---

## Using curvature directly

First-order methods (14.01 to 14.03) see only the slope and pay for curvature with iterations: order $\kappa$ steps for gradient descent, $\sqrt\kappa$ with momentum. Adam uses a crude diagonal curvature estimate. Second-order methods use the Hessian, or an approximation of it, to choose both the direction and the length of each step.

## Newton's method

Minimise the second-order Taylor model at $x_k$ (lesson 09.04):

$$f(x_k + p) \approx f(x_k) + \nabla f^\top p + \tfrac12 p^\top H p \quad\Rightarrow\quad p = -H^{-1}\nabla f(x_k).$$

On an exact quadratic this lands on the minimum in one step, whatever $\kappa$ is. More generally Newton's method is **affine-invariant**: rescaling or rotating the variables changes nothing, which is exactly the ill-conditioning that cripples gradient descent. Near a minimum with positive-definite Hessian, convergence is **quadratic**: $\lVert e_{k+1}\rVert \le C\lVert e_k\rVert^2$.

<div class="ex">
<div class="lab">Worked example</div>

**Newton on $f(x) = e^x - 2x$, minimum at $\ln 2 = 0.693147$**

Here $f'(x) = e^x - 2$ and $f''(x) = e^x$, so the update is $x \leftarrow x - (e^x - 2)/e^x = x - 1 + 2e^{-x}$.

From $x_0 = 0$: $x_1 = 1$ (error 0.31), $x_2 = 0.7358$ (error 0.043), $x_3 = 0.69404$ (error 0.0009), $x_4 = 0.6931476$ (error $4\times10^{-7}$).

Each error is roughly half the square of the previous one: the digits double, the signature of quadratic convergence.

</div>

The problems:

- **Cost.** Storing $H$ takes $d^2$ numbers and solving with it about $d^3/3$ flops. At $d = 10^4$ that is 800 MB in fp64 and around $3\times10^{11}$ flops per step; at $d = 10^9$ it is impossible.
- **Non-convexity.** If $H$ has negative eigenvalues, $-H^{-1}\nabla f$ may point uphill or towards a saddle point; Newton's method is attracted to any stationary point. Fixes: damping ($H + \lambda I$, Levenberg-Marquardt), trust regions (minimise the model only within a radius), or the Gauss-Newton approximation, which is positive semi-definite by construction for least-squares losses.
- **Far from the optimum**, the full step can overshoot, which is why it is combined with a line search.

## Line search

Given a descent direction $p$ (with $\nabla f^\top p < 0$), a line search chooses a step length $t$. Exact minimisation along the line is wasteful; instead accept any $t$ satisfying:

1. **Armijo (sufficient decrease)**: $f(x + tp) \le f(x) + c_1 t\,\nabla f(x)^\top p$, with $c_1$ small (typically $10^{-4}$).
2. **Curvature (Wolfe)**: $\nabla f(x + tp)^\top p \ge c_2\,\nabla f(x)^\top p$, with $c_2 = 0.9$ for quasi-Newton methods; it rules out steps so short that the slope has barely changed.

Backtracking starts at $t = 1$ (the natural Newton step) and multiplies by 0.5 until Armijo holds.

```
def backtracking(f, x, fx, g, p, t=1.0, c1=1e-4, shrink=0.5):
    slope = g @ p                       # must be negative
    while f(x + t * p) > fx + c1 * t * slope:
        t *= shrink
    return t
```

## Quasi-Newton: BFGS and L-BFGS

Quasi-Newton methods never form $H$. They learn about it from the gradients they already compute. After a step $s_k = x_{k+1} - x_k$ with gradient change $y_k = \nabla f_{k+1} - \nabla f_k$, the true Hessian approximately satisfies $H s_k \approx y_k$. BFGS updates an inverse-Hessian approximation $B$ so that this **secant condition** holds exactly, with a rank-two correction that keeps $B$ symmetric positive-definite as long as $y_k^\top s_k > 0$, which the Wolfe curvature condition guarantees. BFGS converges superlinearly but still stores a dense $d\times d$ matrix.

**L-BFGS** stores only the last $m$ pairs $(s_i, y_i)$, typically $m = 5$ to 20, and applies $B$ to the gradient with the two-loop recursion in $O(md)$ time. For $d = 10^6$ and $m = 10$ that is 20 million numbers instead of $10^{12}$.

~~~dg-flow One L-BFGS iteration: curvature comes only from recent gradient differences, never from a stored Hessian.
Gradient g_k -(two-loop recursion with m pairs)-> Direction p = -B g -(Wolfe line search)-> *Step x_k+1 -(store s, y; drop oldest)-> Updated memory
~~~

| Method | Memory | Cost per step | Local rate | Needs
| --- | --- | --- | --- | --- |
| Gradient descent | $O(d)$ | one gradient | linear, depends on kappa | step size
| Newton | $O(d^2)$ | gradient + Hessian + $O(d^3)$ solve | quadratic | convexity or safeguards
| BFGS | $O(d^2)$ | gradient + $O(d^2)$ | superlinear | line search
| L-BFGS | $O(md)$ | gradient + $O(md)$ | linear, much better constant | deterministic gradients

## Where these methods are used in ML

L-BFGS is the default solver for scikit-learn's `LogisticRegression` and a standard choice in SciPy (`minimize(method="L-BFGS-B")`, which also handles box constraints) for full-batch problems with up to millions of parameters: GLMs, CRFs, Gaussian-process hyperparameters, physics fits. It fails on mini-batch training because noisy gradient differences $y_k$ corrupt the curvature pairs, and the line search assumes $f$ can be evaluated exactly. Deep learning instead uses structured approximations of curvature: Hessian-vector products $Hv$ (computable in about two gradient evaluations with autodiff), Gauss-Newton and natural-gradient methods such as K-FAC, and Shampoo-style preconditioners, which sit between Adam's diagonal and full Newton.

<div class="take">
<div class="lab">Key takeaway</div>

Newton's method removes ill-conditioning by stepping with the inverse Hessian and converges quadratically, but costs $O(d^3)$ and needs safeguards on non-convex problems; L-BFGS with a Wolfe line search gets most of the benefit in $O(md)$ memory and is the tool of choice for smooth, full-batch problems.

</div>
