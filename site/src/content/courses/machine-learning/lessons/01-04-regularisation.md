---
id: "01.04"
course: "machine-learning"
module: "01"
slug: "regularisation"
title: "Regularisation"
keyIdeas: ["Regularisation keeps a flexible model but adds a preference for simple solutions: minimise $\\hat R_S(w) + \\lambda\\,\\Omega(w)$, accepting a little more bias for much lower variance.", "**L2** ($\\sum_j w_j^2$) shrinks every weight proportionally and rarely to exactly zero; **L1** ($\\sum_j |w_j|$) pushes with constant force and drives many weights exactly to zero (sparsity).", "The strength $\\lambda$ is a hyperparameter: search it on a log grid and pick the bottom of the U-shaped validation curve, never by training or test error. Standardise features and leave the intercept unpenalised.", "A penalty is equivalent to a hard constraint $\\Omega(w) \\le t$ and to a prior in MAP estimation: Gaussian for L2, Laplace for L1.", "**Early stopping** limits how far gradient descent travels from its start and behaves much like L2 with $\\lambda \\approx 1/(\\eta T)$; tree depth, $k$ in k-NN, dropout and data augmentation are regularisers in the same sense."]
question: "A teammate fits a linear model on 1,500 rows and 400 features (some in raw units such as income in dollars, others 0/1 flags). Training error is tiny but validation error is three times larger. They add an L2 penalty, try lambda = 0.01, 0.1 and 1 on the test set, pick lambda = 1 because it scored best, and report that test score. What is right and what is wrong with this procedure, what would you do instead, and how would you check that regularisation actually helped?"
answer: "The diagnosis is right: a large gap between tiny training error and much higher validation error is high variance, and a penalty is a sensible treatment. Several things are wrong with how it was applied. (1) No standardisation: the penalty treats every weight alike, but a weight's size depends on its feature's units. Income in dollars needs only a tiny weight, so it is barely penalised, while the 0/1 flags have weights of normal size and get shrunk hard. Put a `StandardScaler` before the model inside a pipeline so it is fitted on training folds only, and leave the intercept unpenalised (the library default). (2) The grid is too coarse and the winner is at its edge: lambda = 1 is the largest value tried, so the optimum may be larger. Use a log grid such as 1e-4 to 1e3 in steps of about 3x and extend it until the minimum is inside. (3) Choosing lambda on the test set turns the test set into a validation set, so the reported score is optimistically biased. Tune on a validation split or with k-fold cross-validation on the training data, refit the chosen model on training plus validation data, and evaluate on the test set once. To check that it helped: plot training and validation error against log lambda and look for a U with an interior minimum; the train/validation gap should shrink; the weight norm should drop and huge opposite-sign weights should disappear; predictions should change less across bootstrap refits; and the final model must beat a simple baseline. If many of the 400 features are probably irrelevant, also try L1 or elastic net (lesson 02.02), checking that the selected features are stable across resamples."
sources: [{"title": "Deep Learning (Goodfellow, Bengio, Courville), chapter 7: Regularization", "url": "https://www.deeplearningbook.org/contents/regularization.html", "note": "Sections 7.1 and 7.8: L2/L1 analysis and the early-stopping-as-L2 derivation."}, {"title": "Dive into Deep Learning: Weight decay", "url": "https://d2l.ai/chapter_linear-regression/weight-decay.html", "note": "A short runnable demonstration of L2 on a high-dimensional regression."}, {"title": "The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Free PDF. Section 3.4 on shrinkage methods and the constraint geometry."}]
practice: ["ridge-regression", "lasso-regression", "early-stopping", "lagrange-multipliers-regularization-as-constrain", "adamw-decoupled-weight-decay"]
tags: ["regularisation", "early stopping"]
tools: ["NumPy"]
---

## What you will learn

- What regularisation is: keep a flexible model, but charge it for being complicated.
- How the objective "loss + $\lambda$ × penalty" works.
- L2 (shrink everything) versus L1 (switch some weights off), worked by hand on three points.
- How to choose the strength $\lambda$ with a validation set and read the U-shaped curve.
- Why early stopping, tree depth limits and dropout are the same idea in different clothes.

*Before you start:* you need lesson 01.02 (bias, variance and overfitting) and lesson 01.03 (validation sets and cross-validation). Knowing what a linear model and gradient descent are helps; the maths is limited to sums, squares and one derivative.

## From diagnosis to treatment

Lesson 01.02 showed the disease: a model with too much freedom fits the noise in its training data (high **variance**). Lesson 01.03 gave the thermometer: a validation set reveals it. This lesson is the first treatment.

The obvious fix is a smaller model, but that is crude: you pick from a few sizes, and a small model may be unable to express the real pattern. **Regularisation** keeps the big, flexible model and adds a preference for "simple" versions of it, with one continuous dial for how strong that preference is.

## The idea in one picture

Take twelve noisy measurements of a smooth wave and fit a degree-9 polynomial to them. A degree-9 polynomial has ten coefficients, almost one per data point, so it can bend through nearly every point. We fit it three times, changing only the penalty strength $\lambda$ (Greek "lambda"), and score each fit on 200 fresh points it never saw.

| Penalty strength $\lambda$ | Training MSE | Validation MSE | Largest weight (absolute)
| --- | --- | --- | --- |
| 0 (no penalty) | 0.017 | 0.71 | 8,536
| 0.001 | 0.035 | 0.14 | 6.6
| 10 | 0.146 | 0.51 | 0.07

~~~dg-fig 01.04-fits.svg
The same degree-9 polynomial fitted three ways. With no penalty it threads through the points with wild swings; a small penalty gives a smooth curve close to the true wave; a large penalty flattens it into almost a straight line.
~~~

Read the table row by row:

- No penalty: training error is almost zero, but the curve swings between the points and validation error is the worst. The weights are in the thousands, with alternating signs that cancel to make those swings.
- A small penalty raises training error a little (0.017 to 0.035) and cuts validation error fivefold (0.71 to 0.14).
- A huge penalty squashes every weight towards zero; the curve is too flat to follow the wave. That is **underfitting**, or high **bias**.

The middle row is the whole point: slightly worse training error for much better error on new data.

## Step 1: add a price for complexity

Ordinary training minimises the average loss on the training set $S$, written $\hat R_S(w)$ ("empirical risk", lesson 01.01). Regularisation adds a second term:

$$\min_w \; \hat R_S(w) + \lambda\,\Omega(w).$$

- $w$ is the vector of model parameters (the weights).
- $\hat R_S(w)$ is how badly the model fits the training data, for example the mean squared error.
- $\Omega(w)$ (Greek "omega") is the **penalty** or **regulariser**: a number that is large when the model is complicated.
- $\lambda \ge 0$ is the **regularisation strength**, the exchange rate between "fit the data" and "stay simple". It is a **hyperparameter**: you choose it, the optimiser does not.

$\lambda = 0$ is plain training; as $\lambda \to \infty$ the optimiser drives $w$ to 0, a constant model. The useful models lie in between.

Why do small weights mean "simple"? The weights set how fast the output changes when an input changes. Small weights give a smooth function that cannot jump about to visit individual noisy points; the wild curve needed weights in the thousands to make its jumps.

**Number check** (scikit-learn adds the penalty to the *sum* of squared errors). The unpenalised fit has error 0.21 but squared weights summing to about 215 million, so at $\lambda = 0.001$ its cost is $0.21 + 0.001 \times 214{,}859{,}220 \approx 214{,}859$. The smooth fit has error 0.42 and squared weights summing to 68.3: cost $0.42 + 0.001 \times 68.3 \approx 0.49$. Even a tiny $\lambda$ makes the wild solution hopelessly expensive.

## Step 2: choose how to measure complexity

The two standard penalties both measure the size of the weight vector, in different ways:

| Property | L2 (ridge, weight decay) | L1 (lasso)
| --- | --- | --- |
| Penalty $\Omega(w)$ | $\sum_j w_j^2$ | $\sum_j |w_j|$
| Effect on weights | shrinks all toward 0, rarely exactly 0 | sets many exactly to 0
| Gradient of penalty | $2\lambda w_j$, proportional to size | $\lambda\,\text{sign}(w_j)$, constant push
| Correlated features | spreads weight across them | tends to pick one arbitrarily
| Closed form (least squares) | yes | no; coordinate descent or proximal methods
| Prior (MAP view) | Gaussian | Laplace

The **gradient row** explains the behaviour. Think of the penalty as a spring pulling each weight towards zero:

- Under **L2** the pull is proportional to the weight, so it fades as the weight nears zero. Weights get close to zero but stop short.
- Under **L1** the pull is $\lambda$ however small the weight is. If the data's push on a weight is weaker than that, the weight is driven to exactly zero: the feature is switched off. This is **sparsity**.

Below, a linear model predicts diabetes progression from ten standardised measurements (age, BMI, blood pressure, six blood serum values), refitted for many $\lambda$ values.

~~~dg-fig 01.04-paths.svg
Weight paths on the diabetes data. Read left (weak penalty) to right (strong). Under L2 every weight glides towards zero together; under L1 weights hit exactly zero one after another. The two axes differ in scale because the libraries define the objectives differently.
~~~

With L1 at a moderate strength the model keeps 7 of the 10 measurements; at a strong one, only 3 (BMI, blood pressure and one serum value). L2 never drops a feature. Prefer L2 when you believe most features matter a little, L1 when a few matter a lot; **elastic net** mixes the two. Lesson 02.02 covers all three for linear regression in depth.

<div class="ex">
<div class="lab">Worked example</div>

**L2 versus L1 on one weight, by hand**

Data: inputs $x = (1, 2, 3)$, targets $y = (1, 3, 7)$, model $\hat y = w x$ with no intercept. The two sums we need are $\sum x_i^2 = 1 + 4 + 9 = 14$ and $\sum x_i y_i = 1 + 6 + 21 = 28$.

No penalty. Minimise $\sum (y_i - w x_i)^2$. Setting the derivative $-2\sum x_i(y_i - w x_i)$ to zero gives $w = \sum x_i y_i / \sum x_i^2 = 28 / 14 = 2$.

L2. Minimise $\sum (y_i - w x_i)^2 + \lambda w^2$. The derivative is $-2(28 - 14w) + 2\lambda w = 0$, so $w = 28 / (14 + \lambda)$. At $\lambda = 14$: $w = 28/28 = 1.0$. At $\lambda = 56$: $w = 28/70 = 0.4$. However large $\lambda$ gets, $w$ stays positive.

L1. Minimise $\sum (y_i - w x_i)^2 + \lambda |w|$. For $w > 0$ the derivative is $-2(28 - 14w) + \lambda = 0$, so $w = (28 - \lambda/2)/14$. At $\lambda = 14$: $w = (28 - 7)/14 = 1.5$. At $\lambda = 56$: $w = (28 - 28)/14 = 0$. For any $\lambda \ge 56$ the weight stays exactly 0, because the data's pull ($2 \times 28$) can no longer beat the constant push $\lambda$.

Now a weak relationship. Keep $x = (1, 2, 3)$ but use $y = (1, 1, 0)$, so $\sum x_i y_i = 1 + 2 + 0 = 3$ and the unpenalised weight is $3/14 \approx 0.214$. At $\lambda = 14$, L2 gives $3/28 \approx 0.107$ (exactly half, like the strong weight). L1 would give $(3 - 7)/14 < 0$, which contradicts $w > 0$; checking $w < 0$ gives $(3 + 7)/14 > 0$, also a contradiction, so the minimum is at the kink: $w = 0$.

Summary at $\lambda = 14$: L2 multiplied both weights by $14/28 = 0.5$ (2 to 1.0, 0.214 to 0.107). L1 subtracted $\lambda / (2 \times 14) = 0.5$ from both and clipped at zero (2 to 1.5, 0.214 to 0). Multiply versus subtract-and-clip is the whole difference.

</div>

## Step 3: choose the strength with validation data

The penalty shape decides *how* the model is simplified; $\lambda$ decides *how much*. You cannot pick $\lambda$ by training error, because training error is always lowest at $\lambda = 0$. Instead, fit the model for a range of $\lambda$ values and score each one on validation data (or with k-fold cross-validation, lesson 01.03).

~~~dg-fig 01.04-lambda.svg
Error against penalty strength for the degree-9 polynomial. Training error only goes up as lambda grows. Validation error falls while the penalty removes variance, bottoms out, then rises as the penalty adds too much bias. The best lambda sits at the bottom of the U.
~~~

This is the bias-variance curve from lesson 01.02, mirrored: small $\lambda$ overfits, large $\lambda$ underfits. Here the minimum is at $\lambda \approx 3 \times 10^{-4}$ with a validation MSE of 0.13, close to the 0.09 that even a perfect model would score because of the noise in the data.

- **Use a log grid** ($10^{-4}, 10^{-3}, \dots, 10^{2}$): the effect of $\lambda$ is multiplicative.
- **Keep the best value inside the grid**; if it sits at an end, extend the grid.
- **Refit, then test once**, on training plus validation data, as in lesson 01.03.

## Step 4: regularisation beyond weight penalties

The underlying idea, *limit how freely the model can fit the training data*, appears in every model family:

| Model family | Regularisation knob | Stronger regularisation means
| --- | --- | --- |
| Linear and logistic models | L2 / L1 penalty $\lambda$ (scikit-learn's `C` is $1/\lambda$) | larger $\lambda$, smaller `C`
| k-nearest neighbours | number of neighbours $k$ | larger $k$ (averages over more points)
| Decision trees | max depth, min samples per leaf, pruning | shallower trees, bigger leaves
| Gradient boosting | learning rate, number of trees, subsampling | smaller steps, fewer trees
| Support vector machines | `C` (inverse penalty) | smaller `C`
| Neural networks | weight decay, dropout, early stopping, data augmentation | more decay, more dropout, earlier stop

**Early stopping** is the most widely used regulariser for anything trained by gradient descent. Training starts from small weights and every step moves them further out; stop early and they have not travelled far enough to become extreme. In practice you evaluate on validation data regularly, keep the best model so far, and stop once validation error has not improved for a set number of evaluations, the **patience**.

~~~dg-fig 01.04-early.svg
Left: gradient descent on the degree-9 polynomial from zero weights. Training error keeps falling; validation error bottoms out, then climbs. Right: on a two-weight problem, gradient descent from zero follows a route similar to the ridge solutions as lambda is relaxed, so stopping early acts like an L2 penalty.
~~~

Stopping at the best step gives validation error 0.13, matching the best ridge model; running to the end gives 0.22. Later you will meet **data augmentation**, **dropout**, **label smoothing** and **ensembling**: all reduce variance at some cost in bias.

## Regularisation in code

In scikit-learn the penalty is part of the model; put a scaler in front, inside a pipeline.

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import load_diabetes

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 1, 12)); y = np.sin(2*np.pi*x) + rng.normal(0, 0.3, 12)
xv = rng.uniform(0, 1, 200); yv = np.sin(2*np.pi*xv) + rng.normal(0, 0.3, 200)

for lam in [1e-6, 1e-4, 1e-3, 1e-2, 1, 10]:
    model = make_pipeline(PolynomialFeatures(9, include_bias=False),
                          StandardScaler(),          # put every column on the same scale
                          Ridge(alpha=lam))          # alpha is sklearn's name for lambda
    model.fit(x[:, None], y)
    val = np.mean((model.predict(xv[:, None]) - yv) ** 2)
    print(f"lambda={lam:g}  val MSE={val:.3f}  max|w|={np.abs(model[-1].coef_).max():.2f}")
# lambda=1e-06  val MSE=0.258  max|w|=42.74
# lambda=0.0001  val MSE=0.148  max|w|=12.71
# lambda=0.001  val MSE=0.136  max|w|=6.56
# lambda=0.01  val MSE=0.176  max|w|=3.10
# lambda=1  val MSE=0.409  max|w|=0.22
# lambda=10  val MSE=0.509  max|w|=0.07

X, t = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)
for lam in [0.1, 1, 5, 20]:
    lasso = Lasso(alpha=lam).fit(X, t)
    print(f"lasso lambda={lam:<4}  non-zero weights: {np.sum(lasso.coef_ != 0)} of 10")
# lasso lambda=0.1   non-zero weights: 9 of 10
# lasso lambda=1     non-zero weights: 7 of 10
# lasso lambda=5     non-zero weights: 5 of 10
# lasso lambda=20    non-zero weights: 3 of 10
```

Both estimators leave the intercept unpenalised. In PyTorch, L2 arrives as the optimiser's `weight_decay` argument (prefer `AdamW`), and early stopping is a few lines that track the best validation loss.

## Checking that it worked

Regularisation is working when these signs appear together:

- **The gap closes.** Training error rises a little, validation error falls, and the gap shrinks.
- **The validation curve has an interior minimum.** A curve that only rises with $\lambda$ means the model was not overfitting; one that only falls means you need larger values.
- **Weights become sensible.** Huge alternating weights disappear (8,536 to 6.6 above).
- **Predictions are stable** across bootstrap refits of the training data.
- **It beats a baseline** such as predicting the mean (lesson 01.05).

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Features not standardised | a feature in small units (millimetres) is crushed while one in large units barely moves | put `StandardScaler` before the model, inside a pipeline
| Penalising the intercept | predictions biased towards zero when targets have a large mean | leave the intercept out of the penalty (library defaults do this)
| Choosing $\lambda$ on the test set | test score looks great, real-world performance does not | tune on validation or cross-validation, test once
| Best $\lambda$ at the edge of the grid | validation curve still falling (or rising) at the end | extend the log grid until the minimum is inside
| Copying $\lambda$ between libraries or data sizes | the "same" $\lambda$ gives a very different fit | check whether the loss is a sum or a mean; Ridge uses a sum, Lasso a mean with factor 1/2
| Regularising an underfitting model | both errors high and rising with $\lambda$ | add capacity or features instead
| Reading L1 zeros as "irrelevant" | a dropped feature reappears on a resample | with correlated features lasso picks one arbitrarily; use elastic net or check stability
| L2 term added to the loss with Adam | weight decay weaker than expected for some weights | use decoupled weight decay (AdamW)

## Going deeper

**Three equivalent views.** The same regulariser can be read three ways, and each answers a different question.

~~~dg-layers Three views of the same regulariser. Each is useful for a different question.
*Penalty | minimise loss + lambda times norm; what you implement
Constraint | minimise loss subject to norm at most t; explains the geometry (corners of the L1 ball give zeros)
Prior | MAP estimate with a Gaussian or Laplace prior on weights; explains what lambda means
~~~

**Constraint view.** Minimising $\hat R + \lambda\|w\|^2$ gives the same solution as minimising $\hat R$ subject to $\|w\|^2 \le t$, for some budget $t$ that depends on $\lambda$ (a Lagrange multiplier argument: $\lambda$ is the multiplier of the constraint). Geometrically, grow the loss's elliptical contours outward from the unpenalised optimum until they first touch the allowed region.

~~~dg-fig 01.04-geometry.svg
The constraint view. Loss contours are ellipses around the unpenalised best point. The L2 region is a round disc, so the first contact is usually at a point where both weights are non-zero. The L1 region is a diamond with corners on the axes, and the contours often touch it first at a corner, here (1, 0), where the second weight is exactly zero.
~~~

The disc has no corners, so contact almost never lies on an axis. The diamond's corners stick out, so a whole range of contour positions touches the same corner; in $d$ dimensions the L1 ball has $2d$ corners plus edges and faces, all where some coordinates are zero. That is the geometric reason for sparsity.

**Bayesian view.** Suppose $y = w^\top x + \varepsilon$ with Gaussian noise of variance $\sigma^2$, and put a prior $w \sim \mathcal{N}(0, \tau^2 I)$ on the weights. The **MAP** (most probable) weights maximise $\log p(y \mid w) + \log p(w)$, which up to constants is $-\frac{1}{2\sigma^2}\sum_i (y_i - w^\top x_i)^2 - \frac{1}{2\tau^2}\|w\|^2$. Multiply by $-2\sigma^2$ and you get ridge regression with $\lambda = \sigma^2 / \tau^2$. A tight prior (small $\tau$) or noisy data (large $\sigma$) means strong regularisation. A Laplace prior, $p(w_j) \propto e^{-|w_j|/b}$, gives the L1 penalty instead.

**Early stopping as L2, made precise.** Take a quadratic loss with Hessian eigenvalues $h_i$ and unpenalised optimum $w^*$. In the eigenbasis, $T$ steps of gradient descent with learning rate $\eta$ from $w = 0$ give component $i$ equal to $\big(1 - (1 - \eta h_i)^T\big) w^*_i$, while ridge gives $\frac{h_i}{h_i + \lambda} w^*_i$. Both factors are close to 1 for directions where the loss is steep ($h_i$ large) and close to 0 for flat directions. Using $(1 - \eta h)^T \approx e^{-\eta h T}$, the switch between the two regimes happens at $h_i \approx 1/(\eta T)$ for gradient descent and at $h_i \approx \lambda$ for ridge, so early stopping behaves like L2 with $\lambda \approx 1/(\eta T)$: more steps or a larger learning rate means weaker regularisation. Goodfellow et al. (Deep Learning, chapter 7) give the full derivation.

**Weight decay is not always L2.** With plain gradient descent, adding $\frac{\lambda}{2}\|w\|^2$ to the loss equals shrinking every weight by $(1 - \eta\lambda)$ per step. Adam rescales each gradient coordinate, the penalty's included, so the two differ; AdamW applies the decay directly to the weights (Deep Learning course).

## Key terms

| Term | Meaning
| --- | --- |
| Regularisation | Any change to training that limits how freely the model fits the training data, to reduce variance
| Penalty / regulariser $\Omega(w)$ | A term added to the loss that is large for complicated models
| Regularisation strength $\lambda$ | Hyperparameter setting the exchange rate between fitting data and staying simple
| L2 / ridge / weight decay | Penalty $\sum_j w_j^2$; shrinks all weights proportionally, rarely to exactly zero
| L1 / lasso | Penalty $\sum_j |w_j|$; constant pull that sets many weights exactly to zero
| Sparsity | Having many weights exactly zero, so the model ignores those features
| Early stopping | Stopping gradient descent at the best validation score; acts like L2
| Patience | How many evaluations without improvement to wait before stopping
| MAP estimate | The most probable weights under a prior; L2 and L1 correspond to Gaussian and Laplace priors

<div class="take">
<div class="lab">Key takeaway</div>

regularisation keeps a flexible model but charges it for complexity, trading a little bias for much lower variance. L2 shrinks all weights, L1 switches some off, early stopping limits how far the optimiser travels, and in every case the strength is chosen on validation data, never on training or test data.

</div>
