---
id: "04.05"
course: "machine-learning"
module: "04"
slug: "gradient-boosting"
title: "Gradient boosting"
keyIdeas: ["Gradient boosting builds a model as a sum: start from the best constant, then repeatedly fit a small tree to what the model still gets wrong and add a fraction $\\nu$ of it.", "\"What the model gets wrong\" is the negative gradient of the loss at each training point (the **pseudo-residual**): $y - F$ for squared error, $y - p$ for log-loss. The same algorithm therefore handles any differentiable loss.", "Shrinkage (learning rate 0.01-0.1) plus many trees plus early stopping is the main regulariser; halving the learning rate roughly doubles the trees needed. Tree size, row and column subsampling and leaf constraints add more.", "XGBoost uses a second-order expansion: a leaf's weight is $w^* = -G/(H+\\lambda)$ and a split's gain compares $G^2/(H+\\lambda)$ before and after, with a penalty $\\gamma$ per leaf.", "LightGBM and scikit-learn's HistGradientBoosting bin features into at most 255 bins and grow trees leaf-wise, which makes training on millions of rows fast."]
question: "You train LightGBM with `learning_rate=0.3`, `num_leaves=255` on 20,000 rows and see validation loss bottom out at round 40 and then climb. Your manager suggests reducing the number of trees to 40. Is that the best fix? What would you change and why?"
answer: "Stopping at 40 trees is better than training to 1,000, but it keeps a model built from 40 large, aggressive steps. Each tree has up to 255 leaves fitted on 20,000 rows, so many leaves contain few samples (fewer than 80 on average, far fewer in the deep branches that leaf-wise growth creates) and fit noise in the pseudo-residuals. At a learning rate of 0.3 each of those noisy trees also makes a large change to the predictions, so the errors of individual trees are not averaged out by later ones. Better: lower the learning rate to about 0.05 and let early stopping on a separate validation set choose the number of trees (it will now be several hundred, roughly six times as many); shrink `num_leaves` to 15-63 and raise `min_child_samples` (say 50-100) so leaves average over more rows; add row and column subsampling (`bagging_fraction` with `bagging_freq=1`, `feature_fraction` around 0.7-0.8) and perhaps `lambda_l2`, which shrinks the weights of small leaves via $-G/(H+\\lambda)$. Then confirm on a test set that was not used for early stopping or tuning. Many small, regularised steps usually reach a lower minimum of validation loss than a few large ones, not just the same minimum reached sooner, as the learning-rate curves in this lesson show."
sources: [{"title": "Friedman (2001): Greedy Function Approximation: A Gradient Boosting Machine", "url": "https://doi.org/10.1214/aos/1013203451", "note": "The original gradient boosting paper: pseudo-residuals, per-leaf line search, shrinkage."}, {"title": "Chen and Guestrin (2016): XGBoost: A Scalable Tree Boosting System", "url": "https://arxiv.org/abs/1603.02754", "note": "Section 2 derives the second-order leaf weights and split gain."}, {"title": "XGBoost docs: Introduction to Boosted Trees", "url": "https://xgboost.readthedocs.io/en/stable/tutorials/model.html", "note": "A clean derivation of the regularised objective and structure score."}, {"title": "LightGBM docs: Features", "url": "https://lightgbm.readthedocs.io/en/stable/Features.html", "note": "Histogram algorithm, leaf-wise growth, and how categorical and missing values are handled."}]
practice: ["gradient-boosted-regressor", "decision-tree-regressor"]
tags: ["gradient boosting", "XGBoost", "LightGBM"]
tools: ["XGBoost", "LightGBM", "scikit-learn"]
---

## What you will learn

- The core loop: start from a constant, fit a small tree to what the model still gets wrong, add a fraction of it, repeat.
- Why "what the model gets wrong" is the negative gradient of the loss, so one algorithm handles regression, classification and robust losses.
- How to run two boosting rounds by hand, for squared error and for log-loss.
- The knobs that matter (learning rate, number of trees, tree size, subsampling) and how early stopping picks the number of trees for you.
- What XGBoost, LightGBM, CatBoost and `HistGradientBoosting` add: second-order leaf weights, regularised gain, histograms.

*Before you start:* you should know how a regression tree picks splits and leaf values (04.01, 04.02), the boosting idea of adding weak learners one at a time (04.04), log-loss and the sigmoid from logistic regression (02.03), gradient descent (small steps against the slope), and validation splits (01.03).

## From AdaBoost to any loss

Lesson 04.04 ended with a surprise: AdaBoost, which looked like a clever reweighting scheme, is really adding one weak learner at a time to lower one particular loss, the exponential loss. That loss is harsh on mislabelled points and only fits binary classification.

Friedman (2001) asked the general question. Pick any differentiable loss $L(y, F)$ that measures how bad a prediction $F$ is for a label $y$. How do you add trees one at a time so that the total loss $\sum_i L(y_i, F(x_i))$ keeps going down? His answer, **gradient boosting**, is today the default model for tabular data such as credit, pricing, ranking and fraud.

## The idea in one picture

A courier company wants to predict delivery time (minutes) from distance (km). Seven of its 80 recorded trips:

| Distance (km) | 0.8 | 2.2 | 3.9 | 5.4 | 6.2 | 7.7 | 9.4
| --- | --- | --- | --- | --- | --- | --- | --- |
| Delivery time (min) | 16 | 22 | 19 | 25 | 30 | 43 | 39

The pattern is not a straight line: a plateau from 2 to 5 km, a jump near 6 km (crossing the river), another plateau. Gradient boosting builds a model in rounds:

1. Start with the dumbest model: predict the average time for everyone.
2. Look at the **residuals**, actual minus predicted. They show where the model is too low and too high.
3. Fit a small regression tree (here depth 2, so at most 4 leaves) to those residuals.
4. Add a fraction of that tree's output to the model, and go back to step 2.

~~~dg-fig 04.05-stages.svg
Top row: the model after 0, 3 and 10 trees. Bottom row: the residuals left at that point (dots) and the small tree fitted to them (black steps). Each tree only has to describe what the previous trees missed, and the residuals shrink towards pure noise.
~~~

Read the figure column by column. At the start the residuals carry the whole pattern and the first tree finds the jump at 6 km. After three trees the large structure is in place. After ten, training error has fallen from 89 to about 2 and the residuals look like noise.

Two words to fix now:

- An **ensemble** here is a sum: $F_M(x) = F_0 + \nu h_1(x) + \nu h_2(x) + \dots + \nu h_M(x)$, where each $h_m$ is a small tree.
- The **learning rate** (or **shrinkage**) $\nu$ ("nu"), like 0.1, is the fraction of each tree actually added (the figure uses 0.3).

Contrast a random forest (04.03): it averages deep, independent trees, mainly reducing variance. Boosting adds shallow trees *in sequence*, each correcting the sum so far, mainly reducing bias.

## Step 1: start from the best constant

Before any tree, the model is one number $F_0$ chosen to minimise the loss over the training set:

$$F_0 = \arg\min_c \sum_{i=1}^n L(y_i, c).$$

$\arg\min_c$ means "the $c$ that makes the sum smallest": the mean of $y$ for squared error, the median for absolute error, the log-odds of the base rate for log-loss. For the deliveries it is the mean, 29 minutes.

## Step 2: the pseudo-residuals are the downhill direction

Treat the $n$ training predictions $F(x_1), \dots, F(x_n)$ as $n$ free parameters. Gradient descent would nudge each against its own slope. The amount for example $i$ is the **pseudo-residual**

$$r_i = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}.$$

In words: how much, and in which direction, should the prediction for example $i$ move to reduce its loss? For squared error $L = \tfrac12 (y - F)^2$, the derivative is $-(y - F)$, so $r_i = y_i - F(x_i)$: the ordinary residual. That is why the figure looked like "fitting residuals". For other losses the pseudo-residual is a different quantity, but it always plays the same role.

Number check: with $y = 10$ and current prediction $F = 6$, squared error gives $r = 4$ (move up by 4); absolute error gives $r = \operatorname{sign}(4) = 1$ (move up, by a fixed amount).

## Step 3: fit a tree to the pseudo-residuals

The $r_i$ exist only for training points, but the model must predict for new inputs. So we fit a small regression tree $h_m$ to the pairs $(x_i, r_i)$ with ordinary squared-error splitting (04.01): a version of "the downhill direction" defined for every $x$.

After the tree's *structure* is chosen, each leaf's value is usually re-optimised for the actual loss (the constant that lowers $L$ most for that leaf's samples), a **line search per leaf**. For squared error the leaf mean of the residuals is already optimal, so nothing changes; for other losses it matters (see Step 4).

## Step 4: take a small step

Update the model by adding a shrunken copy of the tree:

$$F_m(x) = F_{m-1}(x) + \nu\, h_m(x).$$

$F_{m-1}$ is the model after $m-1$ trees, $h_m$ is the new tree, and $\nu$ is the learning rate, typically 0.01 to 0.1. Then recompute pseudo-residuals at the new predictions and repeat for $M$ rounds.

~~~dg-flow Gradient boosting loop. The tree only ever sees pseudo-residuals, so the loss can be anything differentiable.
Current F -(gradient of loss)-> Pseudo-residuals r -(fit small tree)-> h_m -(F += ν·h_m)-> *Updated F
~~~

Why not add the whole tree ($\nu = 1$)? Each tree is fitted to noisy residuals, so it is partly wrong. Adding only a tenth means no single tree's mistakes dominate and later trees can correct course.

<div class="ex">
<div class="lab">Worked example</div>

**two rounds with squared error, by hand**

Three training points: $x = (1, 2, 3)$ with targets $y = (3, 5, 10)$. Learning rate $\nu = 0.1$, trees are stumps (one split).

Start: $F_0 = \text{mean}(y) = (3 + 5 + 10)/3 = 6$. Training MSE $= (9 + 1 + 16)/3 = 8.67$.

Round 1 residuals: $r = y - F_0 = (-3, -1, 4)$. The best stump splits at $x = 2.5$: the left leaf holds $(-3, -1)$, mean $-2$; the right leaf holds $4$. (The other split, at 1.5, leaves $(-1, 4)$ together and has a much larger squared error.)

Update: $F_1 = (6 - 0.1 \times 2,\ 6 - 0.1 \times 2,\ 6 + 0.1 \times 4) = (5.8, 5.8, 6.4)$. Training MSE $= (2.8^2 + 0.8^2 + 3.6^2)/3 = (7.84 + 0.64 + 12.96)/3 = 7.15$.

Round 2 residuals: $r = (3 - 5.8,\ 5 - 5.8,\ 10 - 6.4) = (-2.8, -0.8, 3.6)$. The same split wins again; leaf means are $(-2.8 - 0.8)/2 = -1.8$ and $3.6$.

Update: $F_2 = (5.8 - 0.18,\ 5.8 - 0.18,\ 6.4 + 0.36) = (5.62, 5.62, 6.76)$. Training MSE $= (2.62^2 + 0.62^2 + 3.24^2)/3 = (6.86 + 0.38 + 10.50)/3 = 5.92$.

Each round closes about a tenth of the gap. With $\nu = 1$ the first stump would jump straight to $(4, 4, 10)$; the slowness is deliberate and is what stops boosting memorising noise.

</div>

## Losses and their pseudo-residuals

Because the tree only ever sees pseudo-residuals, switching the task is a matter of switching one formula.

| Loss | $L(y, F)$ | Pseudo-residual $r$ | Initial $F_0$
| --- | --- | --- | --- |
| Squared error | $\tfrac12 (y-F)^2$ | $y - F$ | mean of $y$
| Absolute error | $\lvert y - F\rvert$ | $\operatorname{sign}(y - F)$ | median of $y$
| Log-loss, $y \in \{0,1\}$ | $-y\ln p - (1-y)\ln(1-p)$, $p = \sigma(F)$ | $y - p$ | log-odds of base rate
| Huber | quadratic near 0, linear in tails | clipped residual | median

~~~dg-fig 04.05-residuals.svg
What each loss asks the next tree to fit. Squared error passes errors through at full size; absolute and Huber loss cap them, so a wild point cannot dominate. For log-loss the target is y − p, which fades to zero once a point is confidently correct.
~~~

For classification, $F$ is a **log-odds score** and $p = \sigma(F) = 1/(1 + e^{-F})$ is the probability, as in logistic regression. A positive example at $p = 0.9$ has pseudo-residual $0.1$ (nudge up a little); at $p = 0.2$ it has $0.8$.

For log-loss the best leaf value is not the mean of the residuals. The per-leaf line search is done with one Newton step, which divides the summed gradient by the summed curvature:

$$\text{leaf value} = \frac{\sum_{i \in \text{leaf}} r_i}{\sum_{i \in \text{leaf}} p_i(1-p_i)}.$$

<div class="ex">
<div class="lab">Worked example</div>

**one round of log-loss boosting**

Four customers, $x = (1, 2, 3, 4)$, churn labels $y = (1, 1, 1, 0)$. Base rate $3/4$, so $F_0 = \ln(0.75/0.25) = \ln 3 = 1.099$ and every $p = 0.75$. Log-loss $= -(3\ln 0.75 + \ln 0.25)/4 = 0.562$.

Pseudo-residuals $y - p = (0.25, 0.25, 0.25, -0.75)$. A stump at $x = 3.5$ separates the last customer.

Each point has curvature $p(1-p) = 0.75 \times 0.25 = 0.1875$. Left leaf: $0.75 / (3 \times 0.1875) = 1.333$. Right leaf: $-0.75 / 0.1875 = -4$.

With $\nu = 0.1$: left scores become $1.099 + 0.133 = 1.232$, so $p = 0.774$; the right score becomes $1.099 - 0.4 = 0.699$, so $p = 0.668$. Log-loss falls from 0.562 to 0.468.

With $\nu = 1$ the probabilities would jump to 0.919 and 0.052 (log-loss 0.077) in a single round, which on real, noisy data is how boosting overfits.

</div>

Robust losses limit the influence of outliers, fixing the noise sensitivity of AdaBoost's exponential loss:

~~~dg-fig 04.05-huber.svg
Three bad records (crosses). Under squared-error loss their large residuals keep attracting trees and the fit spikes towards them; under Huber loss their pseudo-residuals are capped and the model follows the bulk of the data.
~~~

## Gradient boosting in code

First the whole algorithm for squared error, from scratch, on a standard synthetic benchmark (Friedman #1: five useful features, five noise features). Then the same model from scikit-learn.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

X, y = make_friedman1(n_samples=800, noise=2.0, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.5, random_state=0)

nu, n_trees = 0.1, 120
F_tr = np.full(len(y_tr), y_tr.mean())          # F0 = mean of y
F_va = np.full(len(y_va), y_tr.mean())
for m in range(n_trees):
    r = y_tr - F_tr                               # pseudo-residuals (squared error)
    tree = DecisionTreeRegressor(max_depth=3).fit(X_tr, r)
    F_tr += nu * tree.predict(X_tr)
    F_va += nu * tree.predict(X_va)
print(round(np.mean((y_va - F_va)**2), 2))        # 6.51

gb = GradientBoostingRegressor(n_estimators=120, learning_rate=0.1, max_depth=3, random_state=0)
gb.fit(X_tr, y_tr)
print(round(np.mean((y_va - gb.predict(X_va))**2), 2))   # 6.5
```

The loop and the library agree (predicting the mean would score about 29). On larger data, use a histogram-based implementation and let early stopping choose the number of trees:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

clf = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=2000,            # generous cap; early stopping decides
    max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=50,
    random_state=0)
clf.fit(X_tr, y_tr)
print(clf.n_iter_)                                # 398 rounds were run before stopping
p = clf.predict_proba(X_te)[:, 1]
print(round(log_loss(y_te, p), 3), round(accuracy_score(y_te, p > 0.5), 3))   # 0.162 0.951
```

XGBoost and LightGBM have the same fit/predict interface; their early stopping takes an explicit `eval_set`.

## Regularisation: the knobs and how they interact

Gradient boosting overfits if run too long: every extra tree also fits some noise. Its knobs interact, so learn them as a set.

~~~dg-fig 04.05-curves.svg
Validation error as trees are added (Friedman #1, 400 training rows). Left: a learning rate of 1 bottoms out after 4 trees at a poor value; 0.3, 0.1 and 0.03 reach their best after 25, 122 and 422 trees, and 0.1 and 0.03 end lowest. Right: stumps learn slowly, depth 3 is best, and depth-8 trees drive training error (dotted) to zero while validation error stalls high.
~~~

- **Learning rate and number of trees** trade off almost exactly: halving $\nu$ roughly doubles the trees needed (25 trees at 0.3, 122 at 0.1, 422 at 0.03 in the figure). Smaller $\nu$ almost always generalises a little better, at the cost of training time; below about 0.05 the gain is usually small.
- **Early stopping**: hold out a validation set and stop when its loss has not improved for, say, 50 rounds. This makes the number of trees a fitted quantity rather than a guess. Keep the test set out of it.
- **Tree size**: depth 3-8, or 15-63 leaves. Depth controls the order of interactions the model can represent per tree: stumps give an additive model (each feature's effect separate), depth 2 allows pairwise interactions, and so on.
- **Subsampling**: fitting each tree on a random 50-80% of rows (stochastic gradient boosting, Friedman 2002) and a fraction of columns both reduce variance and speed training.
- **Leaf constraints**: minimum samples or minimum hessian sum per leaf, and L2 penalties on leaf values.

A sensible default: rate 0.05-0.1, a large tree cap, early stopping with patience 50, 15-31 leaves, subsampling around 0.8; then tune leaf settings (06.05).

## Reading the result

- **Plot training and validation loss against the number of trees.** A healthy run falls fast, then flattens. If validation turns up early while training keeps dropping, trees are too big or the rate too high.
- **Check where early stopping fired.** If it stopped within the first 20 rounds, the learning rate is too high. If it never fired and hit the cap, raise the cap or the rate.
- **Compare against a baseline** (01.05): constant, linear, random forest.
- **For classifiers, check calibration** (03.04); class weights can distort probabilities.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Validation loss bottoms out in the first few dozen rounds | learning rate too high | lower $\nu$ to 0.05-0.1, let early stopping pick more trees
| Training loss near zero, validation much worse | leaves too large or too deep for the data size | fewer leaves, larger minimum leaf size, L2 on leaf values, subsampling
| Fit chases a handful of extreme targets | squared error on heavy-tailed data | Huber or absolute loss, or model $\log y$
| Great validation score, poor in production | early stopping and tuning on the same split, or leaky features | separate validation and test sets; check features for leakage
| Training takes hours | exact split finding on millions of rows | histogram method (`HistGradientBoosting`, `tree_method="hist"`, LightGBM)
| Deep, narrow branches on a small dataset | leaf-wise growth with no limits | cap `num_leaves`, raise `min_child_samples`

## Going deeper: XGBoost's second-order view

XGBoost (Chen and Guestrin, 2016) writes the objective at round $m$ with a second-order Taylor expansion of the loss around the current predictions. Let $g_i$ and $h_i$ be the first and second derivatives of the loss at the current prediction (the slope and the curvature for example $i$), and add a penalty $\gamma T + \tfrac12 \lambda \sum_j w_j^2$ for a tree with $T$ leaves and leaf weights $w_j$. For a fixed tree structure, write $G_j = \sum_{i \in j} g_i$ and $H_j = \sum_{i \in j} h_i$ for the sums over the samples in leaf $j$. Then the optimum is closed form:

$$w_j^* = -\frac{G_j}{H_j + \lambda} \qquad \text{Gain} = \frac12\Big[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big] - \gamma$$

Here $L$ and $R$ are the left and right child of a candidate split. Splits are scored with this gain, and a split that does not beat $\gamma$ is not made.

For squared error, $g_i = F_i - y_i$ (minus the residual), $h_i = 1$ and $H$ is the sample count, so $\lambda$ shrinks small leaves towards zero: a leaf with one residual of 4 gets weight $4/(1+1) = 2$ when $\lambda = 1$, not 4. Applied to round 1 of the hand example ($g = (3, 1, -4)$, $\lambda = 1$, $\gamma = 0$): the split at 2.5 has $G_L = 4, H_L = 2$ and $G_R = -4, H_R = 1$, so the leaf weights are $-4/3 = -1.33$ and $4/2 = 2$ instead of $-2$ and $4$, and the gain is $\tfrac12(16/3 + 16/2 - 0) = 6.67$. For log-loss, $h_i = p_i(1-p_i)$, recovering the Newton step above.

Making it fast: the expensive part is split finding. Modern libraries bucket each feature into at most 255 quantile bins once, then build per-node histograms of $(G, H)$ and scan bins instead of sorted values. A useful trick: a child's histogram equals the parent's minus its sibling's, so only the smaller child needs a pass over data.

| Library | Distinctive ideas
| --- | --- |
| XGBoost | Second-order objective with $\lambda$, $\gamma$; sparsity-aware default directions for missing values; `hist` tree method
| LightGBM | Histograms; leaf-wise (best-first) growth capped by `num_leaves`; GOSS keeps large-gradient rows and samples small ones; bundles mutually exclusive sparse features
| CatBoost | Symmetric (oblivious) trees; ordered boosting and ordered target statistics to avoid target leakage with categorical features
| scikit-learn `HistGradientBoosting*` | LightGBM-style histograms, native missing values and categoricals, built-in early stopping

Leaf-wise growth reaches lower loss with the same number of leaves than depth-wise growth, but can grow deep, narrow branches on small data; cap `num_leaves` and set `min_child_samples`. All four handle missing values natively by learning which side of each split they go to.

## Key terms

| Term | Meaning
| --- | --- |
| Gradient boosting | Building a sum of small trees, each fitted to the negative gradient of the loss at the current predictions
| Pseudo-residual | $-\partial L / \partial F$ for one example: the direction and size its prediction should move
| Learning rate / shrinkage $\nu$ | Fraction of each new tree added to the model, typically 0.01-0.1
| Early stopping | Stopping when validation loss has not improved for a set number of rounds
| Line search per leaf | Re-fitting each leaf's value to minimise the actual loss after the split is chosen
| Stochastic gradient boosting | Fitting each tree on a random subsample of rows (and optionally columns)
| Gain (XGBoost) | Loss reduction of a split from gradient and hessian sums, minus the per-leaf penalty $\gamma$
| Histogram method | Binning features into at most 255 buckets so split search scans bins, not sorted values

<div class="take">
<div class="lab">Key takeaway</div>

gradient boosting adds small trees fitted to the negative gradient of any differentiable loss; tune the learning rate, tree size and subsampling, and let early stopping pick the number of trees.

</div>
