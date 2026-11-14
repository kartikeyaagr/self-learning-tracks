---
id: "02.02"
course: "machine-learning"
module: "02"
slug: "ridge-lasso-and-elastic-net"
title: "Ridge, lasso and elastic net"
keyIdeas: ["Ridge solves $(X^\\top X + \\lambda I)w = X^\\top y$; in the SVD basis it shrinks each direction by $\\sigma_j^2/(\\sigma_j^2 + \\lambda)$, so low-variance (often collinear) directions are shrunk most.", "The lasso has no closed form, but each coordinate update is a **soft-threshold** $S(\\rho, \\lambda) = \\mathrm{sign}(\\rho)\\max(|\\rho| - \\lambda, 0)$, which is where exact zeros come from.", "Elastic net mixes L1 and L2: it still selects features but keeps correlated groups together and can select more than $n$ features when $d > n$.", "Fit the whole **regularisation path** on a log grid from $\\lambda_{\\max}$ down, with warm starts, and pick $\\lambda$ by cross-validation; the one-standard-error rule picks a sparser model at no real cost.", "Penalty strength depends on feature scale and on the library's objective scaling; standardise first, and check whether the loss is divided by $n$."]
question: "You run lasso on unstandardised data. Income is in dollars (standard deviation 30,000) and age is in years (standard deviation 12); both matter about equally. Which coefficient does the lasso tend to zero out, why, and what would change if you re-expressed income in thousands of dollars?"
answer: "The lasso penalises $|w_j|$ regardless of units. To have the same effect on predictions, the age coefficient must be about 2,500 times larger than the income coefficient (30,000 / 12), so it costs 2,500 times more penalty for the same contribution. In coordinate-descent terms, age's update is thresholded far more heavily relative to its signal, so age is driven to zero first while income survives almost unshrunk. Re-expressing income in thousands multiplies its coefficient by 1,000 and its penalty with it, so the selection shifts toward age: the chosen model now depends on an arbitrary choice of units. Standardising both features (with training statistics) removes that dependence, puts every coefficient on a per-standard-deviation scale, and makes a single $\\lambda$ meaningful across features."
sources: [{"title": "Regularization Paths for Generalized Linear Models via Coordinate Descent", "url": "https://www.jstatsoft.org/article/view/v033i01", "note": "The glmnet paper (Friedman, Hastie, Tibshirani): soft-threshold updates, warm-started paths and the lambda grid."}, {"title": "Statistical Learning with Sparsity", "url": "https://hastie.su.domains/StatLearnSparsity/", "note": "Chapters 2-4 on the lasso, elastic net and their solvers. Free PDF."}, {"title": "scikit-learn: Linear models (Ridge, Lasso, Elastic-Net)", "url": "https://scikit-learn.org/stable/modules/linear_model.html", "note": "The exact objectives behind alpha and l1_ratio, and the CV variants LassoCV and ElasticNetCV."}, {"title": "The Elements of Statistical Learning, ch. 3.4", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Ridge via the SVD, effective degrees of freedom and the lasso path."}]
practice: ["ridge-regression", "lasso-regression", "lasso-regression-using-ista", "subgradients-for-non-smooth-functions", "hessian-matrix-and-convexity-of-linear-regressio"]
tags: ["shrinkage", "sparsity"]
tools: ["NumPy", "scikit-learn"]
---

## Picking up from module 01

Lesson 01.04 introduced L2 and L1 penalties, their constraint and prior views, and why lasso drops one of two correlated features. This lesson is the mechanics for linear regression: what ridge does to each direction of the data, how the lasso is actually solved, what elastic net adds, and how to choose $\lambda$ without fooling yourself. Assume columns of $X$ are standardised (mean 0, variance 1) and $y$ is centred, so the intercept is unpenalised and drops out.

## Ridge through the SVD

The ridge objective $\|y - Xw\|^2 + \lambda\|w\|^2$ has the closed form

$$\hat w_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y.$$

For any $\lambda > 0$ the matrix is positive definite, so the solution exists even with collinear columns or $d > n$. The SVD $X = U\Sigma V^\top$ shows what it does to predictions:

$$\hat y_{\text{ridge}} = \sum_j u_j \,\frac{\sigma_j^2}{\sigma_j^2 + \lambda}\, u_j^\top y.$$

OLS keeps each component $u_j^\top y$ whole. Ridge multiplies it by a factor between 0 and 1 that is close to 1 for large singular values and close to 0 for small ones. Small singular values are directions where the features barely vary, typically differences between near-duplicate columns, and they are exactly where OLS coefficients blow up. The **effective degrees of freedom** $\mathrm{df}(\lambda) = \sum_j \sigma_j^2/(\sigma_j^2 + \lambda)$ runs from $d$ at $\lambda = 0$ to $0$ as $\lambda \to \infty$, which is a better complexity measure than counting features.

## Lasso by coordinate descent

The lasso objective, in the scaling scikit-learn and glmnet use, is

$$\tfrac{1}{2n}\|y - Xw\|^2 + \lambda \|w\|_1.$$

The kink at zero rules out a closed form, but one coordinate at a time is easy. Hold all other weights fixed, let $r_{(j)} = y - \sum_{k \ne j} x_k w_k$ be the partial residual, and $\rho_j = \tfrac1n x_j^\top r_{(j)}$. With standardised columns the exact minimiser over $w_j$ is the soft-threshold $w_j = S(\rho_j, \lambda)$: shrink $\rho_j$ toward zero by $\lambda$, and if it crosses zero, stop at zero. Cycle over coordinates until nothing moves.

```python
import numpy as np
def soft(r, lam): return np.sign(r) * np.maximum(np.abs(r) - lam, 0.0)

def lasso_cd(X, y, lam, iters=200):        # X standardised, y centred
    n, d = X.shape; w = np.zeros(d); r = y.copy()
    for _ in range(iters):
        for j in range(d):
            r += X[:, j] * w[j]             # remove j's contribution
            w[j] = soft(X[:, j] @ r / n, lam)
            r -= X[:, j] * w[j]
    return w
```

Two facts follow directly. At $\lambda_{\max} = \max_j |x_j^\top y|/n$ every coordinate thresholds to zero, which is where the path starts. And any feature whose correlation with the current residual stays below $\lambda$ never enters the model. An alternative solver, ISTA, takes a full gradient step on the squared loss and then soft-thresholds every coordinate; it is slower per unit of progress but generalises to other smooth losses.

The lasso has known limits: with $d > n$ it selects at most $n$ features, and among a group of correlated features it picks one somewhat arbitrarily.

## Elastic net

Elastic net adds both penalties:

$$\tfrac{1}{2n}\|y - Xw\|^2 + \lambda\Big(\alpha\|w\|_1 + \tfrac{1-\alpha}{2}\|w\|_2^2\Big).$$

The coordinate update becomes $w_j = S(\rho_j, \lambda\alpha) / (1 + \lambda(1-\alpha))$: threshold, then shrink. The L2 part makes the objective strictly convex, so correlated features get similar weights (the grouping effect) and more than $n$ features can be selected. Beware the naming: scikit-learn's `alpha` is this $\lambda$, and its `l1_ratio` is this $\alpha$.

| Property | Ridge | Lasso | Elastic net
| --- | --- | --- | --- |
| Closed form | yes | no | no
| Exact zeros | no | yes | yes
| Correlated group | shares weight | keeps one | keeps the group, shrunk
| Features selected when d > n | all, none zero | at most n | can exceed n
| Usual solver | Cholesky or SVD | coordinate descent | coordinate descent

## The path, and choosing lambda

Never fit one $\lambda$. Fit a grid of about 100 values spaced evenly in log scale from $\lambda_{\max}$ down to $\lambda_{\max}/10^{3}$ or so, starting each fit from the previous solution (a warm start). Neighbouring solutions are close, so the whole path costs little more than a few cold fits.

~~~dg-flow The lasso path: start where every weight is zero and relax the penalty, letting features enter one by one.
lambda max: all weights 0 -(lower lambda)-> few strong features -(lower)-> *CV minimum -(lower)-> near OLS, overfit
~~~

Choose $\lambda$ by K-fold cross-validation on the training data (module 06 covers the variants). The curve of CV error against $\log\lambda$ is usually flat near its minimum, so many $\lambda$ values are statistically indistinguishable. The **one-standard-error rule** picks the largest $\lambda$ whose CV error is within one standard error of the minimum: same accuracy within noise, fewer features, more stable. Three practical traps:

- **Scale.** Standardise using training-fold statistics only, inside the CV loop.
- **Library conventions.** scikit-learn's `Ridge` minimises $\|y - Xw\|^2 + \alpha\|w\|^2$ with no $1/n$; `Lasso` divides the loss by $2n$. The same number means different strengths, and ridge's effective strength changes with dataset size.
- **Shrinkage bias.** Lasso shrinks the weights it keeps. If you want unbiased coefficients for the selected set, refit OLS on them (the relaxed lasso), and validate that refit too.

<div class="ex">
<div class="lab">Worked example</div>

**shrinkage factors and a threshold**

Ridge on a design with singular values $(10, 1, 0.1)$ and $\lambda = 1$: factors $100/101 = 0.99$, $1/2 = 0.5$ and $0.01/1.01 \approx 0.01$. The strong direction is untouched, the weak one is almost removed, and $\mathrm{df} \approx 1.5$ out of 3.

Lasso coordinate update with $\lambda = 0.3$: if $\rho_j = 0.8$ the new weight is $0.5$; if $\rho_j = 0.2$ it is exactly $0$; if $\rho_j = -0.5$ it is $-0.2$.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

ridge rescales directions of the data, the lasso soft-thresholds coordinates, and elastic net does both. Standardise, fit the whole path with warm starts, and choose lambda by cross-validation, preferring the sparser end of the flat region.

</div>
