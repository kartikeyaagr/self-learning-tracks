---
id: "02.02"
course: "machine-learning"
module: "02"
slug: "ridge-lasso-and-elastic-net"
title: "Ridge, lasso and elastic net"
keyIdeas: ["Regularisation adds $\\lambda$ times a size penalty to the squared error. It accepts slightly worse training error in exchange for smaller, steadier weights, which usually predict better on new data.", "**Ridge** (L2) has the closed form $(X^\\top X + \\lambda I)w = X^\\top y$. In the SVD basis it scales each direction by $\\sigma_j^2/(\\sigma_j^2 + \\lambda)$, muting the poorly measured (often collinear) directions most.", "**Lasso** (L1) has no closed form, but each coordinate update is a **soft-threshold** $S(\\rho, \\lambda) = \\mathrm{sign}(\\rho)\\max(|\\rho| - \\lambda, 0)$, which is where exact zeros and feature selection come from.", "**Elastic net** mixes L1 and L2: it still selects features, but keeps correlated groups together and can select more than $n$ features when $d > n$.", "Fit the whole path on a log grid from $\\lambda_{\\max}$ down with warm starts, choose $\\lambda$ by cross-validation, and prefer the one-standard-error choice. Standardise inside a pipeline, and check how your library scales the loss before reusing an `alpha` value."]
question: "You run lasso on unstandardised data. Income is in dollars (standard deviation 30,000) and age is in years (standard deviation 12); both matter about equally. Which coefficient does the lasso tend to zero out, why, and what would change if you re-expressed income in thousands of dollars?"
answer: "The lasso penalises $|w_j|$ regardless of units. To have the same effect on predictions, the age coefficient must be about 2,500 times larger than the income coefficient (30,000 / 12), so it pays 2,500 times more penalty for the same contribution to the fit. Put differently, income can carry its share of the fit with a weight so tiny that the penalty barely notices it, while age needs a large weight that the penalty charges heavily. So age is driven to zero first while income survives almost unshrunk. Re-expressing income in thousands multiplies its coefficient by 1,000 and its penalty with it, so the selection shifts toward age: the chosen model now depends on an arbitrary choice of units, which is a red flag. Standardising both features (with training-set statistics only, inside a pipeline so each CV fold uses its own) removes that dependence, puts every coefficient on a per-standard-deviation scale, and makes a single $\\lambda$ meaningful across features. The same problem affects ridge and elastic net, only less visibly because ridge shrinks rather than zeroes."
sources: [{"title": "Regularization Paths for Generalized Linear Models via Coordinate Descent", "url": "https://www.jstatsoft.org/article/view/v033i01", "note": "The glmnet paper (Friedman, Hastie, Tibshirani): soft-threshold updates, warm-started paths and the lambda grid."}, {"title": "Statistical Learning with Sparsity", "url": "https://hastie.su.domains/StatLearnSparsity/", "note": "Chapters 2-4 on the lasso, elastic net and their solvers. Free PDF."}, {"title": "scikit-learn: Linear models (Ridge, Lasso, Elastic-Net)", "url": "https://scikit-learn.org/stable/modules/linear_model.html", "note": "The exact objectives behind alpha and l1_ratio, and the CV variants LassoCV and ElasticNetCV."}, {"title": "The Elements of Statistical Learning, ch. 3.4", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Ridge via the SVD, effective degrees of freedom and the lasso path."}]
practice: ["ridge-regression", "lasso-regression", "lasso-regression-using-ista", "subgradients-for-non-smooth-functions", "hessian-matrix-and-convexity-of-linear-regressio"]
tags: ["shrinkage", "sparsity"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- Why a price on large weights makes a linear model steadier and better on new data.
- How **ridge** (L2 penalty) shrinks every weight smoothly, direction by direction.
- How the **lasso** (L1 penalty) sets weights to exactly zero, and the update that solves it.
- What **elastic net** adds, and how to choose between the three.
- How to fit a regularisation path and pick the penalty strength by cross-validation.

*Before you start:* lesson 02.01 (least squares, normal equations, collinearity) and lesson 01.04 (L1 and L2 penalties, bias and variance). Cross-validation is covered fully in module 06. Step 3 uses the SVD; skim it on a first read.

Lesson 02.01 ended with two warnings: extra columns let least squares fit noise, and near-duplicate columns make weights explode. Both have the same cure: keep the squared error, and add a second term that grows with the size of the weights.

## The idea in one picture

Take twelve noisy points from a smooth wave and fit a degree-9 polynomial: nine standardised features $x, x^2, \dots, x^9$ plus an intercept. Then fit the same features again with a penalty of strength $\lambda$ on the squared weights.

~~~dg-fig 02.02-wiggly.svg
Same features, same data, three penalty strengths. Without a penalty the curve shoots up between and beyond the points; a small ridge penalty gives a smooth curve close to the truth; a large one flattens it.
~~~

| Model | Sum of absolute weights | Training error (MSE) | Error against the true curve
| --- | --- | --- | --- |
| No penalty (OLS) | 45,236 | 0.053 | 4.13
| Ridge, $\lambda = 0.01$ | 8.5 | 0.093 | 0.037
| Ridge, $\lambda = 10$ | 0.6 | 0.233 | 0.225

The unpenalised fit has the lowest training error, but its weights are in the thousands with alternating signs: huge terms that almost cancel at the training points and swing wildly between them. Its error against the true curve is about 100 times worse than ridge's. A tiny penalty ($\lambda = 0.01$) gives up a little training error, cuts the weights 5,000-fold and recovers the wave. Too strong a penalty ($\lambda = 10$) flattens everything and underfits.

Words used from here on:

- **Regularisation**: adding a term to the objective that prefers simpler models, here a **penalty** on weight size.
- **Shrinkage**: the effect, weights closer to zero than least squares would put them.
- **Penalty strength** $\lambda$ (scikit-learn: `alpha`): the cost of a unit of weight relative to a unit of error. $\lambda = 0$ is plain least squares; huge $\lambda$ forces every weight to zero.

## Step 1: put a price on the weights

Error term plus $\lambda$ times a size term; the methods differ only in how they measure size.

$$\text{ridge:}\quad \|y - Xw\|^2 + \lambda \|w\|_2^2, \qquad \|w\|_2^2 = \sum_j w_j^2$$

$$\text{lasso:}\quad \tfrac{1}{2n}\|y - Xw\|^2 + \lambda \|w\|_1, \qquad \|w\|_1 = \sum_j |w_j|$$

$X$ is the $n \times d$ feature matrix ($n$ examples, $d$ features), $y$ the targets, $w$ the weights. The squared **L2 norm** sums squared weights; the **L1 norm** sums absolute values. The lasso's error is divided by $2n$, as in scikit-learn and glmnet; scikit-learn's ridge has no such factor, so the same `alpha` means a different strength in `Ridge` and `Lasso`.

Three conventions:

- **Standardise the columns** of $X$ (mean 0, standard deviation 1): the penalty treats every weight alike, so features need comparable scales.
- **Centre** $y$. The intercept is then $\bar y$ and is **not penalised**; shrinking it would pull every prediction toward zero.
- Choose $\lambda$ on held-out data; training error always prefers $\lambda = 0$.

## Step 2: ridge has an exact answer

Set the gradient of the ridge objective to zero, exactly as in 02.01. The gradient is $-2X^\top(y - Xw) + 2\lambda w$, so the minimum satisfies

$$(X^\top X + \lambda I)\, w = X^\top y \quad\Longrightarrow\quad \hat w_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y.$$

$I$ is the identity matrix. The only change from the normal equations is "$+\lambda$" on the diagonal, and it matters: for any $\lambda > 0$ the matrix is positive definite, hence invertible, even with collinear columns or $d > n$. With one centred feature it becomes $w = \sum x_i y_i \,/\, (\sum x_i^2 + \lambda)$: the OLS slope with a bigger denominator.

<div class="ex">
<div class="lab">Worked example</div>

**ridge and lasso by hand on four points**

Reuse 02.01's data: $x = (1, 2, 3, 4)$, $y = (2, 3, 5, 6)$. Centre both: $\tilde x = (-1.5, -0.5, 0.5, 1.5)$, $\tilde y = (-2, -1, 1, 2)$. From 02.01, $\sum \tilde x \tilde y = 7$ and $\sum \tilde x^2 = 5$, so OLS gives $w = 7/5 = 1.4$.

Ridge with $\lambda = 5$ (scikit-learn `Ridge` scaling): $w = 7/(5 + 5) = 0.7$. The intercept is unpenalised: $b = \bar y - w\bar x = 4 - 0.7 \times 2.5 = 2.25$. With $\lambda = 1$: $w = 7/6 \approx 1.167$.

Why 0.7 wins at $\lambda = 5$: its residuals are $\tilde y - 0.7\tilde x = (-0.95, -0.65, 0.65, 0.95)$, squared error $0.9025 + 0.4225 + 0.4225 + 0.9025 = 2.65$, penalty $5 \times 0.7^2 = 2.45$, total $5.10$. OLS has squared error only $0.2$ but penalty $5 \times 1.4^2 = 9.8$, total $10.0$. Ridge trades a worse fit for much smaller weights.

Lasso (scikit-learn scaling, loss divided by $2n = 8$): $\rho = \tfrac1n\sum \tilde x \tilde y = 7/4 = 1.75$ and $\tfrac1n \sum \tilde x^2 = 5/4 = 1.25$. The answer is $w = S(\rho, \lambda)/1.25$, where $S$ subtracts $\lambda$ from $|\rho|$ and stops at zero (step 4).

With $\lambda = 0.5$: $S(1.75, 0.5) = 1.25$, so $w = 1.25/1.25 = 1.0$ and $b = 4 - 1.0 \times 2.5 = 1.5$. With $\lambda \ge 1.75$: $S = 0$, so $w = 0$ exactly and the model predicts the mean, 4, for everyone. scikit-learn returns these exact numbers.

</div>

## Step 3: what ridge does to each direction of the data

Why does ridge tame the exploding weights? Write $X = U\Sigma V^\top$, where the columns $v_j$ of $V$ are directions in feature space, $\sigma_j$ (the **singular values**) measure how much the data spreads along each direction, and $u_j$ are the matching directions in example space. Then the ridge predictions are

$$\hat y_{\text{ridge}} = \sum_j u_j \,\frac{\sigma_j^2}{\sigma_j^2 + \lambda}\, u_j^\top y.$$

OLS keeps every component $u_j^\top y$ whole (factor 1). Ridge multiplies it by a factor between 0 and 1: close to 1 when $\sigma_j$ is large, close to 0 when $\sigma_j$ is small. Small $\sigma_j$ means the features barely vary along that direction, typically the difference between near-duplicate columns ($x^8$ minus $x^9$ in the polynomial). The data says almost nothing there, so OLS weights along it are mostly noise and can be enormous. Ridge mutes exactly those directions.

The **effective degrees of freedom** $\mathrm{df}(\lambda) = \sum_j \sigma_j^2/(\sigma_j^2 + \lambda)$ falls from $d$ at $\lambda = 0$ toward 0: a better complexity measure than counting features.

Number check: singular values $(10, 1, 0.1)$ with $\lambda = 1$ give factors $100/101 \approx 0.99$, $1/2 = 0.5$ and $0.01/1.01 \approx 0.01$. The strong direction is untouched, the weak one is essentially deleted, and $\mathrm{df} \approx 1.5$ out of 3.

## Step 4: why the lasso produces exact zeros

Ridge makes weights small but never exactly zero; the lasso zeroes some out, so it selects features. Two ways to see why:

**Geometry.** A penalised fit equals minimising the error under a budget: $\|w\|_2^2 \le t$ (a disc in 2-D) for ridge, $\|w\|_1 \le t$ (a diamond) for the lasso. The answer is the first point where an error contour, an ellipse around the OLS solution, touches the budget.

~~~dg-fig 02.02-geometry.svg
The loss ellipses grow outward from the OLS solution until they touch the budget. A disc has no corners, so the contact point usually has both weights non-zero. A diamond has corners on the axes, and the ellipse very often touches one first, giving w₂ = 0 exactly.
~~~

**The one-feature formula.** For one standardised feature, write $\rho$ for its least-squares weight, and scale both penalties per example as the lasso objective does. Each penalty turns $\rho$ into a weight in a different way:

- Ridge: $w = \rho/(1 + \lambda)$. Every weight is scaled by the same fraction, so it gets small but never reaches zero.
- Lasso: $w = S(\rho, \lambda) = \mathrm{sign}(\rho)\max(|\rho| - \lambda, 0)$, the **soft-threshold**. It subtracts $\lambda$ from the size of the weight, and if that crosses zero it stops at zero.

~~~dg-fig 02.02-shrink.svg
The four rules side by side with λ = 1. Ridge is a straight line through the origin with a smaller slope. The lasso is the no-penalty line shifted toward zero by λ, with a flat dead zone where the weight is exactly 0. Elastic net thresholds by a smaller amount and then also shrinks.
~~~

The kink of $|w|$ at zero is the reason. Near zero the ridge penalty $w^2$ is almost flat, so a tiny weight costs almost nothing. The lasso charges $\lambda$ per unit of weight however small, so a feature that cannot pay that price in reduced error gets nothing. Number check with $\lambda = 0.3$: $\rho = 0.8$ gives $w = 0.5$; $\rho = 0.2$ gives exactly $0$; $\rho = -0.5$ gives $-0.2$.

## Step 5: solving the lasso by coordinate descent

The kink rules out a closed formula for many features at once, but the one-feature answer above is all we need. **Coordinate descent** updates one weight at a time while holding the others fixed:

1. Remove feature $j$'s contribution from the current residual: $r_{(j)} = y - \sum_{k \ne j} x_k w_k$ (the **partial residual**).
2. Compute its signal $\rho_j = \tfrac1n x_j^\top r_{(j)}$.
3. Set $w_j = S(\rho_j, \lambda)$ (valid for standardised columns, where $\tfrac1n x_j^\top x_j = 1$).
4. Move to the next feature; cycle until no weight changes.

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
# on the diabetes data below, this matches sklearn's Lasso to within 0.001
```

Two facts drop out of the update. From $w = 0$, every feature thresholds to zero when $\lambda \ge \lambda_{\max} = \max_j |x_j^\top y|/n$: that is where the path begins. And a feature whose correlation with the residual stays below $\lambda$ never enters. A second solver, **ISTA**, takes a gradient step on all weights at once and then soft-thresholds each one; it is slower but works for any smooth loss.

## Step 6: elastic net, both penalties at once

The lasso has two weaknesses: with $d > n$ it selects at most $n$ features, and from a group of correlated features it keeps one, somewhat arbitrarily. **Elastic net** adds a little ridge to fix both:

$$\tfrac{1}{2n}\|y - Xw\|^2 + \lambda\Big(\alpha\|w\|_1 + \tfrac{1-\alpha}{2}\|w\|_2^2\Big).$$

Here $\alpha \in [0, 1]$ is the mix: $\alpha = 1$ is the lasso, $\alpha = 0$ is ridge. The coordinate update becomes $w_j = S(\rho_j, \lambda\alpha)/(1 + \lambda(1-\alpha))$: threshold, then shrink. The L2 part makes the objective strictly convex, so correlated features get similar weights (the **grouping effect**). Watch the names: scikit-learn's `alpha` is this $\lambda$, and its `l1_ratio` is this $\alpha$.

| Property | Ridge | Lasso | Elastic net
| --- | --- | --- | --- |
| Closed form | yes | no | no
| Exact zeros | no | yes | yes
| Group of correlated features | shares the weight | keeps one | keeps the group, shrunk
| Features selected when d > n | all, none zero | at most n | can exceed n
| Usual solver | Cholesky or SVD | coordinate descent | coordinate descent
| Reach for it when | many small effects, collinearity | few features really matter | sparse but correlated groups

## Step 7: the path, and choosing λ

Never fit just one $\lambda$. Fit about 100 values on a log grid from $\lambda_{\max}$ down to about $\lambda_{\max}/1000$, starting each fit from the previous solution (a **warm start**). The whole **regularisation path** then costs little more than a few single fits.

~~~dg-fig 02.02-paths.svg
Coefficient paths on scikit-learn's diabetes data (10 standardised features), with the penalty weakening from left to right. Ridge weights grow from zero together and smoothly; lasso weights enter one at a time (bmi first, then s5, then bp) and are exactly zero before they enter.
~~~

~~~dg-flow The lasso path: start where every weight is zero and relax the penalty, letting features enter one at a time.
lambda max: all weights 0 -(lower lambda)-> few strong features -(lower)-> *CV minimum -(lower)-> near OLS, overfit
~~~

To choose a point, use **K-fold cross-validation**: for each $\lambda$, fit on $K-1$ folds, measure error on the held-out fold, average, and plot against $\log\lambda$.

~~~dg-fig 02.02-cv.svg
Ridge on a degree-12 polynomial with 40 points. Training error only rises as λ grows. Cross-validation error falls (less overfitting) and then rises (underfitting), giving a U shape. The diamond marks the one-standard-error choice: a stronger penalty with error statistically indistinguishable from the minimum.
~~~

The curve is usually flat near its bottom, so many values of $\lambda$ tie within noise. The **one-standard-error rule** picks the largest $\lambda$ whose CV error is within one standard error of the minimum: the same accuracy, a simpler and more stable model, and for the lasso fewer features.

## Ridge, lasso and elastic net in code

The scaler sits in a pipeline so each CV fold is standardised with its own statistics.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV

X, y = load_diabetes(return_X_y=True, as_frame=True)       # 442 patients, 10 features
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

for est in [Ridge(alpha=10), Lasso(alpha=1.0), ElasticNet(alpha=1.0, l1_ratio=0.5)]:
    pipe = make_pipeline(StandardScaler(), est).fit(Xtr, ytr)
    print(type(est).__name__, round(pipe.score(Xte, yte), 3), (pipe[-1].coef_ == 0).sum())
# Ridge 0.358 0        test R², number of zero weights
# Lasso 0.356 2        s2 and s4 dropped
# ElasticNet 0.379 1

cv = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0)).fit(Xtr, ytr)
lasso = cv[-1]
print(lasso.alpha_, list(X.columns[lasso.coef_ == 0]))     # 1.86 ['age', 's4', 's6']

mse = lasso.mse_path_.mean(1)                               # one-standard-error rule
se = lasso.mse_path_.std(1, ddof=1) / np.sqrt(5)
i = mse.argmin()
a1 = lasso.alphas_[mse <= mse[i] + se[i]].max()
print(round(a1, 2))                                         # 8.06 -> keeps bmi, bp, s3, s5
```

Plain least squares scores 0.359 here: with 331 rows and 10 features there is little overfitting to remove, though the 1-SE lasso matches it closely (0.336) with 4 features instead of 10. Regularisation pays off most with many, noisy or correlated features, as in the polynomial example.

## Reading the result

- **Where did CV land on the grid?** At either end of the grid, the optimum is probably outside it: extend and refit.
- **Compare with an unpenalised baseline** on the same folds. A tie just means there is enough data per feature.
- **Look at the path.** Features that enter early and stay are well supported; late entrants or sign flips are weak.
- **Check selection stability.** Refit on each CV fold or bootstrap sample and count how often each feature is selected; one picked in 3 of 10 folds is not a finding.
- **Remember the weights are shrunk** toward zero by design: fine for prediction, misleading as effect sizes.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Features not standardised | selection changes when you change units | `StandardScaler` first
| Scaler fitted on all the data | CV error looks better than the test error | put the scaler in a `Pipeline` so each fold fits its own
| Reusing `alpha` across `Ridge`, `Lasso` or libraries | very different shrinkage for the "same" value | check whether the loss is divided by $n$ or $2n$
| Penalising the intercept in hand-written code | predictions pulled toward 0, worse with large $\bar y$ | centre $y$, or leave the bias out of the penalty
| Treating the lasso's selection as the truth | different features chosen on resamples | elastic net, stability checks
| Reporting lasso weights as effects | effects look smaller than they are | refit OLS on the selected features (the relaxed lasso) and validate that refit too
| `ConvergenceWarning` | solver stopped before weights settled | standardise, raise `max_iter`, warm-start from larger $\lambda$

## Going deeper

**Penalties as priors.** Ridge is the maximum a posteriori (MAP) estimate under a Gaussian prior on each weight; the lasso uses a Laplace prior, whose sharp peak at zero is the probabilistic face of the kink. A large $\lambda$ means a strong belief that weights are small.

**Ridge is least squares on padded data.** Append $d$ fake rows $\sqrt{\lambda}\, I$ to $X$ and $d$ zeros to $y$. Plain least squares on the padded data minimises exactly the ridge objective, and the padded matrix always has full column rank.

**Ridge beats OLS for some λ.** Hoerl and Kennard (1970) showed some $\lambda > 0$ always gives ridge a lower mean squared error than OLS. Ridge accepts a little bias for a larger cut in variance; Gauss-Markov (02.01) only promises OLS is best among *unbiased* estimators.

**The dual form when d > n.** Since $(X^\top X + \lambda I)^{-1}X^\top = X^\top(XX^\top + \lambda I)^{-1}$, you can solve an $n \times n$ system instead; predictions then use only inner products $x_i^\top x_k$, the doorway to kernels (02.06).

**Optimality conditions for the lasso.** With residual $r = y - Xw$, a solution must satisfy $\tfrac1n x_j^\top r = \lambda\,\mathrm{sign}(w_j)$ for every non-zero weight and $|\tfrac1n x_j^\top r| \le \lambda$ for every zero weight. Active features are those whose correlation with the residual has hit the ceiling $\lambda$. Path solvers such as glmnet use them to skip features that cannot become active (screening rules), so a coordinate sweep costs $O(nd)$ or less. With strongly correlated features the lasso still predicts well, but which features it selects is fragile.

## Key terms

| Term | Meaning
| --- | --- |
| Regularisation | Adding a term to the training objective that prefers simpler models
| Ridge (L2) | Least squares plus $\lambda\sum w_j^2$; shrinks all weights smoothly, none to zero
| Lasso (L1) | Least squares plus $\lambda\sum \lvert w_j\rvert$; shrinks and sets some weights exactly to zero
| Elastic net | A mix of L1 and L2; sparse, but keeps correlated features together
| Penalty strength λ | The price of weight size relative to error; `alpha` in scikit-learn
| Soft-threshold | $S(\rho, \lambda)$: subtract $\lambda$ from the size, stop at zero; the lasso's per-coordinate answer
| Regularisation path | The weights as a function of $\lambda$ over a log grid
| One-standard-error rule | Choose the largest λ whose CV error is within one SE of the minimum

<div class="take">
<div class="lab">Key takeaway</div>

ridge rescales the directions of the data and mutes the poorly measured ones, the lasso soft-thresholds each weight and so selects features, and elastic net does both. Standardise inside a pipeline, fit the whole path with warm starts, choose λ by cross-validation, and prefer the stronger-penalty end of the flat region.

</div>
