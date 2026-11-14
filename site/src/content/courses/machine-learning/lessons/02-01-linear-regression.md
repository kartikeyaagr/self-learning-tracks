---
id: "02.01"
course: "machine-learning"
module: "02"
slug: "linear-regression"
title: "Linear regression"
keyIdeas: ["Linear regression predicts $\\hat y = w^\\top x + b$ and picks $w$ by minimising squared error $\\|Xw - y\\|^2$; with an appended column of ones the intercept is just another weight.", "Setting the gradient to zero gives the **normal equations** $X^\\top X w = X^\\top y$; solve them with QR or SVD (`lstsq`), never by inverting $X^\\top X$, which squares the condition number.", "Geometrically, $\\hat y$ is the **orthogonal projection** of $y$ onto the column space of $X$, so the residual is orthogonal to every feature column.", "Prediction needs few assumptions; interpreting coefficients and their standard errors needs many (independent, constant-variance errors, no collinearity, no omitted confounders)."]
question: "You add a feature that is an exact copy of an existing one in different units (price in dollars and price in cents). What happens to the normal equations, to the predictions, and to the coefficients returned by `np.linalg.inv(X.T @ X) @ X.T @ y` versus `np.linalg.lstsq`?"
answer: "The new column is 100 times an existing one, so $X$ loses full column rank and $X^\\top X$ is singular: the normal equations have infinitely many solutions, all satisfying $w_{\\text{dollars}} + 100\\,w_{\\text{cents}} = \\text{const}$. The predictions do not change, because the column space is the same and the projection of $y$ onto it is unique. The explicit inverse either raises a singular-matrix error or, because rounding makes the matrix only nearly singular, returns enormous coefficients of opposite sign that are dominated by floating-point noise. `lstsq` uses the SVD, reports rank $d-1$, and returns the minimum-norm solution, which splits the weight between the two columns in proportion to their scale. Neither coefficient is individually meaningful; drop one column, or use ridge, which makes the system invertible and picks a unique split."
sources: [{"title": "Cornell CS4780 lecture notes: Linear Regression", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote08.html", "note": "OLS as maximum likelihood under Gaussian noise and its MAP (ridge) counterpart, in a few pages."}, {"title": "The Elements of Statistical Learning, ch. 3", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Sections 3.1-3.2: least squares geometry, the hat matrix and the Gauss-Markov theorem. Free PDF."}, {"title": "scikit-learn: Ordinary Least Squares", "url": "https://scikit-learn.org/stable/modules/linear_model.html", "note": "The LinearRegression section, including the note on collinearity and solver complexity."}, {"title": "Dive into Deep Learning: Linear Regression", "url": "https://d2l.ai/chapter_linear-regression/linear-regression.html", "note": "The same model fitted by minibatch SGD, as a bridge to the optimisation view."}]
practice: ["linear-regression", "least-squares-solution", "orthogonal-projection-matrix", "moore-penrose-pseudoinverse", "qr-decomposition"]
tags: ["least squares", "normal equations"]
tools: ["NumPy"]
---

## Where this module starts

Module 01 framed learning as choosing a hypothesis class and a loss, then minimising empirical risk. Linear models are the simplest class worth using: the prediction is a weighted sum of features. They are fast, convex to fit, easy to debug, and the baseline every other model in this course has to beat. This lesson covers the regression case; later lessons swap the loss (logistic, hinge) or add a penalty (ridge, lasso) while keeping the same linear core. The ML Maths course (lesson 08.02) derives least squares from the linear-algebra side; here the focus is on what an engineer needs to fit, check and trust it.

## The model and the loss

For $n$ examples with $d$ features, stack inputs into $X \in \mathbb{R}^{n \times d}$ and targets into $y \in \mathbb{R}^n$. Append a column of ones to $X$ and the intercept $b$ becomes one more entry of $w$. The model is $\hat y = Xw$ and ordinary least squares (OLS) minimises

$$L(w) = \tfrac{1}{n}\|Xw - y\|^2 = \tfrac{1}{n}\sum_i (w^\top x_i - y_i)^2.$$

Why squared error? Three reasons. It is the negative log-likelihood if $y = w^\top x + \varepsilon$ with Gaussian noise, so OLS is the maximum-likelihood estimate under that model. It is smooth and convex, so there is one minimum and calculus finds it. And its minimiser is the conditional mean $\mathbb{E}[y \mid x]$, which is usually what you want to predict. The price: a residual of 10 costs 100 times a residual of 1, so a few outliers can drag the fit. Absolute or Huber loss (module 06) trades the closed form for robustness.

"Linear" means linear in the parameters, not in the raw inputs. Features can be $x^2$, $\log x$, interactions $x_1 x_2$ or one-hot encodings, and the fitting machinery is unchanged.

## The normal equations and how to solve them

The gradient is $\nabla L = \tfrac{2}{n} X^\top (Xw - y)$. Setting it to zero gives

$$X^\top X\, w = X^\top y.$$

If $X$ has full column rank ($n \ge d$ and no feature is a linear combination of others), $X^\top X$ is invertible and the solution is unique. In code, do not form the inverse:

```python
import numpy as np
X1 = np.c_[np.ones(len(X)), X]                  # add intercept column
w, res, rank, sv = np.linalg.lstsq(X1, y, rcond=None)
print(rank, sv.max() / sv.min())                # rank and condition number of X
```

`lstsq` uses an SVD, which is stable and also reports rank and singular values. Forming $X^\top X$ squares the condition number: if $\kappa(X) = 10^5$, then $\kappa(X^\top X) = 10^{10}$ and in float64 you have lost about ten of your sixteen significant digits. Cost is $O(nd^2)$ for the factorisation, fine for $d$ in the thousands. For very large $n$ or streaming data, minimise the same loss with (mini-batch) gradient descent; the Hessian is $\tfrac{2}{n}X^\top X$, so its largest eigenvalue caps the stable step size and badly scaled features make convergence slow. Standardising features fixes most of that.

## The geometric view

$Xw$ ranges over the column space of $X$, a $d$-dimensional subspace of $\mathbb{R}^n$. Least squares picks the point in that subspace closest to $y$, which is the orthogonal projection $\hat y = Hy$ with the hat matrix $H = X(X^\top X)^{-1}X^\top$. Rewriting the normal equations as $X^\top (y - \hat y) = 0$ says the residual is orthogonal to every column.

~~~dg-flow Least squares projects y onto the space the features can reach; what is left is orthogonal to every feature.
Targets y -(project onto column space of X)-> *Fitted y-hat -(subtract from y)-> Residual r -(check)-> X-transpose r = 0
~~~

Useful consequences:

- With an intercept column, residuals sum to exactly zero and the fitted line passes through $(\bar x, \bar y)$.
- $R^2 = 1 - \|r\|^2 / \|y - \bar y\|^2$ is the fraction of variance the projection captures. Adding any feature can only enlarge the subspace, so training $R^2$ never drops; judge features on validation data.
- The diagonal $h_{ii}$ of $H$ is the **leverage** of point $i$: how much its own label moves its own prediction. Points with extreme $x$ have high leverage and deserve a look.
- Exact duplicates or collinear features make the subspace smaller than $d$. The projection $\hat y$ is still unique, but $w$ is not: infinitely many weight vectors give the same predictions.

## Assumptions and what breaks

OLS will happily fit any data. The assumptions matter for what you claim afterwards.

| Assumption | Needed for | Symptom when violated | Typical fix
| --- | --- | --- | --- |
| Linear in the chosen features | unbiased predictions | curved pattern in residuals vs fitted | add transforms or interactions
| Independent errors | valid standard errors | autocorrelated residuals in time series | time-aware models, clustered errors
| Constant variance | efficient fit, honest intervals | residual spread grows with fitted value | log the target, weighted least squares
| Normal errors | exact small-sample intervals only | heavy tails in residual plot | usually ignore for prediction; robust loss
| No strong collinearity | stable, interpretable coefficients | huge coefficients with opposite signs | drop or merge features, ridge (02.02)
| No omitted confounders | causal reading of a coefficient | coefficient sign flips when a feature is added | domain knowledge, experiments

Under the first three, the Gauss-Markov theorem says OLS has the lowest variance among unbiased linear estimators. That is a statement about unbiased estimators only; ridge accepts some bias to cut variance further, which is often a better deal (01.02).

<div class="ex">
<div class="lab">Worked example</div>

**four points by hand**

Data $x = (1, 2, 3, 4)$, $y = (2, 3, 5, 6)$. Means $\bar x = 2.5$, $\bar y = 4$. $\sum (x - \bar x)(y - \bar y) = 3 + 0.5 + 0.5 + 3 = 7$ and $\sum (x - \bar x)^2 = 5$, so the slope is $1.4$ and the intercept is $4 - 1.4 \cdot 2.5 = 0.5$.

Fitted values $(1.9, 3.3, 4.7, 6.1)$, residuals $(0.1, -0.3, 0.3, -0.1)$. They sum to 0, and $\sum x_i r_i = 0.1 - 0.6 + 0.9 - 0.4 = 0$: orthogonal to both columns, as the geometry promises.

SSE $= 0.2$, total sum of squares $= 10$, so $R^2 = 0.98$.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

least squares is a projection. Solve it with lstsq or QR, read the residuals rather than the coefficients first, and only interpret coefficients when the error and collinearity assumptions hold.

</div>
