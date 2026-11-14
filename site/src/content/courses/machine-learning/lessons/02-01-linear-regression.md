---
id: "02.01"
course: "machine-learning"
module: "02"
slug: "linear-regression"
title: "Linear regression"
keyIdeas: ["A linear model predicts $\\hat y = w_1 x_1 + \\dots + w_d x_d + b$: each **weight** is the change in prediction per unit of its feature with the others held fixed, and the **bias** shifts the whole prediction.", "Ordinary least squares picks the weights that minimise the mean squared residual. The loss is a bowl with a single bottom, so there is one exact answer, given by the **normal equations** $X^\\top X w = X^\\top y$.", "Fit with `LinearRegression` or `np.linalg.lstsq` (stable SVD), never by inverting $X^\\top X$, which squares the condition number and loses precision.", "\"Linear\" means linear in the weights: adding columns like $x^2$ or $\\log x$ lets the same algorithm fit curves.", "Check residual plots and held-out error before trusting the model; outliers drag the fit because errors are squared, and collinear features make individual weights meaningless even when predictions are fine."]
question: "You fit rent on flat size and get rent = 4.7 + 0.195 x size with training R² = 0.97. A colleague then adds a column 'size_in_sq_ft' (size multiplied by 10.76) and reruns the fit with `np.linalg.inv(X.T @ X) @ X.T @ y`. Describe what happens to (a) the predictions, (b) the two size coefficients, and (c) what `np.linalg.lstsq` would return instead. What would you do?"
answer: "The new column is an exact multiple of an existing one, so the columns of $X$ are linearly dependent and $X^\\top X$ is singular. (a) Predictions do not change: the set of reachable predictions (the column space) is the same, and the projection of $y$ onto it is unique, so training $R^2$ stays 0.97. (b) The explicit inverse either raises a singular-matrix error or, because rounding makes the matrix only *nearly* singular, returns two enormous coefficients of opposite sign whose combination $w_{m^2} + 10.76\\, w_{ft^2}$ still equals about 0.195; the individual values are floating-point noise and will change wildly if a single row is added. (c) `lstsq` uses the SVD, detects that the rank is one less than the number of columns, and returns the minimum-norm solution, which splits the effect between the two columns (in proportion to their scale) so that predictions are unchanged. Neither coefficient is meaningful on its own. Drop the duplicate column; more generally, check for near-duplicates with a correlation matrix or the condition number reported by `lstsq`, and use ridge regression (02.02) when features are strongly but not exactly correlated."
sources: [{"title": "Cornell CS4780 lecture notes: Linear Regression", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote08.html", "note": "OLS as maximum likelihood under Gaussian noise and its MAP (ridge) counterpart, in a few pages."}, {"title": "The Elements of Statistical Learning, ch. 3", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Sections 3.1-3.2: least squares geometry, the hat matrix and the Gauss-Markov theorem. Free PDF."}, {"title": "scikit-learn: Ordinary Least Squares", "url": "https://scikit-learn.org/stable/modules/linear_model.html", "note": "The LinearRegression section, including the note on collinearity and solver complexity."}, {"title": "Dive into Deep Learning: Linear Regression", "url": "https://d2l.ai/chapter_linear-regression/linear-regression.html", "note": "The same model fitted by minibatch SGD, as a bridge to the optimisation view."}]
practice: ["linear-regression", "least-squares-solution", "orthogonal-projection-matrix", "moore-penrose-pseudoinverse", "qr-decomposition"]
tags: ["least squares", "normal equations"]
tools: ["NumPy"]
---

## What you will learn

- What a linear model is, in plain words and in symbols, and why it is the first model you should try on any regression problem.
- How "best fit" is defined (squared error) and why that choice has one clean answer.
- How to fit the line by hand on four points, then with NumPy and scikit-learn on real-sized data.
- How to read a fitted model: residual plots, $R^2$, and when you may (and may not) interpret a coefficient.
- The two classic ways the fit goes wrong: outliers and collinear features.

*Before you start:* you need sums, averages and the idea of a derivative. Matrix notation appears in the second half; the ML Maths course (module 08) covers it if it is new to you. Module 01 of this course explained the general recipe (pick a model family, pick a loss, minimise it on training data); this lesson is the first full example of that recipe.

## Where this module starts

Linear models are the simplest models worth using: the prediction is a weighted sum of the inputs. They train in milliseconds, they have one best answer instead of many local ones, and they are easy to inspect. That makes them the **baseline** every fancier model in this course has to beat. The rest of the module keeps the same linear core and changes one thing at a time: a penalty on the weights (02.02), a different loss for yes/no labels (02.03), many classes (02.04), a margin-based loss (02.05), and finally a trick that makes linear methods fit curves (02.06).

## The idea in one picture

Suppose you want to predict the monthly rent of a flat from its size. You collect ten flats:

| Size (m²) | 30 | 42 | 48 | 55 | 61 | 68 | 75 | 83 | 90 | 104
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rent (thousand / month) | 11 | 13.5 | 13 | 16 | 15.5 | 18.5 | 19 | 20 | 23.5 | 25

Plot them and the pattern is obvious: bigger flats cost more, roughly in a straight line. Linear regression turns "roughly a straight line" into a precise rule.

~~~dg-fig 02.01-fit.svg
The least-squares line through ten flats. Each dashed segment is a residual: how far the real rent is from the line's prediction.
~~~

The line in the figure is

$$\widehat{\text{rent}} = 4.69 + 0.195 \times \text{size}.$$

Read it like a sentence: a flat starts at about 4.7 thousand, and every extra square metre adds about 0.195 thousand (195 per month). A 70 m² flat is predicted at $4.69 + 0.195 \times 70 \approx 18.4$ thousand.

Two words you will see everywhere:

- The **weight** (or coefficient) $w = 0.195$ is the slope: how much the prediction changes per unit of input.
- The **bias** (or intercept) $b = 4.69$ is the prediction when every input is zero. Here that is a 0 m² flat, which does not exist; the intercept is just where the line has to sit so that it fits the data you do have. Do not over-interpret it.

## Step 1: one feature, one line

With a single input $x$ the model is

$$\hat y = w x + b,$$

where $\hat y$ ("y-hat") is the prediction and $y$ is the true label. Each choice of $(w, b)$ is a different line. Learning means choosing the pair that fits the training data best, so we need to define "best".

## Step 2: measure how wrong a line is

For each example the **residual** is $r_i = y_i - \hat y_i$, the vertical gap in the figure. A good line has small residuals, but they come with signs, so we cannot just add them (big positive and big negative errors would cancel). Linear regression squares them and averages:

$$\text{MSE}(w, b) = \frac{1}{n}\sum_{i=1}^{n} \big(y_i - (w x_i + b)\big)^2.$$

This is the **mean squared error**, and the method is called **ordinary least squares (OLS)**. Squaring has three good consequences and one bad one:

- It is smooth, so calculus can find the minimum exactly.
- The loss has a single minimum (see the next figure), so there is no risk of getting stuck.
- The best prediction under squared error is the *average* of $y$ for a given $x$, which is usually what you want to predict. (If the noise is Gaussian, OLS is also the maximum-likelihood fit, the view introduced in lesson 01.01.)
- The bad one: a residual of 10 costs 100 times a residual of 1, so a few wild points can drag the line. We come back to this below.

## Step 3: find the bottom of the bowl

Picture the MSE as a landscape over all possible $(w, b)$. Because the loss is a quadratic in the parameters, the landscape is a bowl with one lowest point.

~~~dg-fig 02.01-loss.svg
Left: hold the intercept fixed and the loss is a parabola in the slope. Right: over both parameters the loss is a bowl; its contour lines are ellipses around the single best point.
~~~

At the bottom of a bowl the slope of the surface is zero in every direction. Setting both partial derivatives of the MSE to zero and solving gives the textbook formulas:

$$w = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}, \qquad b = \bar y - w \bar x,$$

where $\bar x$ and $\bar y$ are the averages. The slope is "how much $x$ and $y$ move together" divided by "how much $x$ moves on its own". The intercept formula says the line always passes through the point of averages $(\bar x, \bar y)$.

<div class="ex">
<div class="lab">Worked example</div>

**fit a line by hand on four points**

Data: $x = (1, 2, 3, 4)$ and $y = (2, 3, 5, 6)$. Averages: $\bar x = 2.5$, $\bar y = 4$.

Deviations from the mean: $x - \bar x = (-1.5, -0.5, 0.5, 1.5)$ and $y - \bar y = (-2, -1, 1, 2)$.

Numerator: $\sum (x - \bar x)(y - \bar y) = 3 + 0.5 + 0.5 + 3 = 7$. Denominator: $\sum (x - \bar x)^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5$. So $w = 7/5 = 1.4$.

Intercept: $b = 4 - 1.4 \times 2.5 = 0.5$. The line is $\hat y = 0.5 + 1.4x$.

Predictions $(1.9, 3.3, 4.7, 6.1)$, residuals $(0.1, -0.3, 0.3, -0.1)$. They sum to exactly 0, which always happens when the model has an intercept.

Squared residuals sum to $0.2$. The total variation of $y$ around its mean is $4 + 1 + 1 + 4 = 10$, so the line explains $1 - 0.2/10 = 98\%$ of it. That number is $R^2$.

</div>

## Step 4: many features at once

Real problems have many inputs: size, floor, distance to the metro, age of the building. The model simply adds one weight per feature:

$$\hat y = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b.$$

Each weight now means "the change in prediction when this feature goes up by one *and all the other features stay the same*". That "holding the rest fixed" clause matters: the weight on size in a model that also knows the number of rooms is not the same number as the weight on size alone, because size and rooms carry overlapping information.

It is convenient to write this with vectors. Put the features of example $i$ in a vector $x_i$, stack all examples as rows of a matrix $X$ (shape $n \times d$), and append a column of ones so that $b$ becomes just another weight. Then all predictions at once are $\hat y = Xw$, and the loss is

$$L(w) = \frac{1}{n}\,\|Xw - y\|^2.$$

The same "set the gradient to zero" step gives the **normal equations**

$$X^\top X\, w = X^\top y.$$

For one feature plus an intercept these reduce exactly to the two formulas in step 3.

## Fitting it in code

You will almost never solve the normal equations by hand. Three standard ways, from most to least common:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

size = np.array([30, 42, 48, 55, 61, 68, 75, 83, 90, 104.])
rent = np.array([11, 13.5, 13, 16, 15.5, 18.5, 19, 20, 23.5, 25.])

model = LinearRegression().fit(size.reshape(-1, 1), rent)   # X must be 2-D: (n, d)
print(model.intercept_, model.coef_)    # 4.69  [0.195]
print(model.predict([[70]]))            # [18.36]

X1 = np.c_[np.ones_like(size), size]                       # intercept column + feature
w, *_ = np.linalg.lstsq(X1, rent, rcond=None)              # same answer: [4.69, 0.195]
```

`np.linalg.lstsq` solves least squares with a stable matrix factorisation (the SVD). Avoid the tempting `np.linalg.inv(X.T @ X) @ X.T @ y`: forming $X^\top X$ squares the problem's sensitivity to rounding (its *condition number*), so if $X$ is even mildly ill-conditioned you can lose most of your significant digits. For very large datasets that do not fit in memory, minimise the same loss with mini-batch gradient descent (`SGDRegressor`, or a one-layer network in PyTorch); remember to standardise features first so the bowl is round rather than a long thin valley.

## Curves are allowed: "linear" means linear in the weights

"Linear" refers to the weights, not to the inputs. You are free to feed the model $x^2$, $\log x$, $x_1 x_2$ or one-hot encoded categories as extra columns. The fitting procedure does not change at all, only the columns of $X$ do.

~~~dg-fig 02.01-poly.svg
With features [1, x] least squares can only draw a straight line; adding an x² column lets the same algorithm fit the curve.
~~~

This is how linear models stay useful on non-linear data, and it is the seed of the kernel idea in lesson 02.06. The danger is that every new column also adds flexibility to fit noise, which is what regularisation (02.02) controls.

## Reading a fitted model

A model that trains is not yet a model you can trust. Check these in order.

**1. Look at the residuals.** Plot residual against predicted value. If the model has captured the pattern, what is left is shapeless noise.

~~~dg-fig 02.01-residuals.svg
Residual plots tell you what the model missed. A random band is healthy; a curve means a missing non-linear feature; a funnel means the noise grows with the prediction.
~~~

- A **curve** says the relationship is not a straight line in your current features: add a transformed feature (square, log) or an interaction.
- A **funnel** says errors grow with the size of the prediction (common for prices and counts). Predicting $\log y$ instead of $y$ often fixes it, and turns the weights into "percentage change" effects.
- **Clusters or stripes** often reveal a categorical variable you left out.

**2. Measure the fit on held-out data.** $R^2 = 1 - \frac{\sum r_i^2}{\sum (y_i - \bar y)^2}$ is the fraction of the target's variance the model explains: 1 is perfect, 0 is no better than always predicting the mean, and on new data it can even go negative. Adding any feature can only increase *training* $R^2$, so always judge on a validation set (module 06). Report an error in the target's own units too, such as RMSE or MAE, because "$R^2 = 0.8$" tells a stakeholder much less than "off by 1.2 thousand per month on average".

**3. Only then read coefficients**, and only under the conditions in the table below.

## What goes wrong

**Outliers pull hard.** Because errors are squared, a single extreme point can move the whole line. Below, one mistyped rent of 45 at 60 m² lifts and tilts the fit for every flat.

~~~dg-fig 02.01-outlier.svg
One data-entry error shifts the whole least-squares line, because its squared residual dominates the loss.
~~~

Look at the data before fitting, check points with large residuals, and if extremes are genuine, use a loss that grows more slowly, such as Huber or absolute error (`HuberRegressor`, covered with losses in module 06).

**Collinear features make weights meaningless.** If two columns carry the same information (price in dollars and price in cents, or "rooms" and "bedrooms + 1"), then infinitely many weight combinations give identical predictions. The predictions are still fine, but the individual weights can be huge, opposite in sign, and change completely when you add a few rows. Drop one of the columns or use ridge regression (02.02).

**Assumptions, and what each one buys you.** OLS fits any data; the assumptions only matter for the claims you make afterwards.

| Assumption | Needed for | Symptom when violated | Typical fix
| --- | --- | --- | --- |
| Linear in the chosen features | unbiased predictions | curved pattern in residuals | add transforms or interactions
| Independent errors | honest standard errors | residuals correlated over time | time-aware validation and models
| Constant variance | efficient fit, honest intervals | funnel-shaped residuals | log the target, weighted least squares
| Normal errors | exact small-sample intervals | heavy-tailed residuals | usually ignore for prediction; robust loss
| No strong collinearity | stable, interpretable weights | huge weights with opposite signs | drop or merge features, ridge
| No omitted confounders | causal reading of a weight | weight flips sign when a feature is added | domain knowledge, experiments

The last row is the most important in practice. A weight tells you how predictions change, not what would happen if you intervened. Ice-cream sales "predict" drownings because both rise in summer.

## Going deeper: the geometric view

Think of the target $y$ as one point in $n$-dimensional space (one coordinate per example). Every possible prediction $Xw$ lies in the flat subspace spanned by the columns of $X$. Least squares picks the point in that subspace closest to $y$: the **orthogonal projection**.

~~~dg-flow Least squares projects y onto the space the features can reach; what is left over is perpendicular to every feature.
Targets y -(project onto column space of X)-> *Fitted y-hat -(subtract from y)-> Residual r -(check)-> X-transpose r = 0
~~~

Rewriting the normal equations as $X^\top (y - \hat y) = 0$ says exactly this: the residual is perpendicular to every feature column. Consequences you can use:

- With an intercept column, residuals sum to zero (they are perpendicular to the column of ones), as in the hand example.
- The projection matrix $H = X(X^\top X)^{-1}X^\top$ is called the **hat matrix** ($\hat y = Hy$). Its diagonal entry $h_{ii}$ is the **leverage** of point $i$: how strongly its own label pulls its own prediction. Points with unusual feature values have high leverage and deserve a look.
- When columns are collinear the subspace is smaller than $d$ dimensions. The projection $\hat y$ is still unique, but many $w$ produce it, which is the collinearity problem seen geometrically.
- Under the first three assumptions in the table, the **Gauss-Markov theorem** says OLS has the smallest variance among all *unbiased* linear estimators. Ridge regression deliberately accepts a little bias to cut variance further, which is often the better trade (module 01, lesson 02).

## Key terms

| Term | Meaning
| --- | --- |
| Weight / coefficient | How much the prediction changes per unit of a feature, others held fixed
| Bias / intercept | The prediction when all features are zero; a shift, rarely meaningful alone
| Residual | Actual minus predicted value for one example
| MSE / RMSE | Mean of squared residuals / its square root, in the target's units
| OLS | Choosing weights that minimise squared error
| Normal equations | $X^\top X w = X^\top y$, the condition for the minimum
| $R^2$ | Fraction of the target's variance explained by the model
| Leverage | How strongly one point's label determines its own fitted value

<div class="take">
<div class="lab">Key takeaway</div>

linear regression fits the weighted sum of features that minimises squared error, which has one exact answer (a projection). Fit it with `LinearRegression` or `lstsq`, check the residual plot and held-out error before anything else, and read individual weights only when features are not collinear and confounders are accounted for.

</div>
