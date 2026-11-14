---
id: "04.05"
course: "machine-learning"
module: "04"
slug: "gradient-boosting"
title: "Gradient boosting"
keyIdeas: ["Gradient boosting is gradient descent in function space: each new tree fits the negative gradient of the loss with respect to the current predictions (the pseudo-residuals), and is added with a small step size.", "For squared error the pseudo-residuals are ordinary residuals $y - F$; for log-loss they are $y - p$. The same algorithm handles any differentiable loss.", "Shrinkage (learning rate 0.01-0.1) plus many trees plus early stopping is the main regulariser; row and column subsampling add more.", "XGBoost uses a second-order expansion: a leaf's weight is $w^* = -G/(H+\\lambda)$ and a split's gain compares $G^2/(H+\\lambda)$ before and after, with a penalty $\\gamma$ per leaf.", "LightGBM and scikit-learn's HistGradientBoosting bin features into at most 255 bins and grow trees leaf-wise, which makes training on millions of rows fast."]
question: "You train LightGBM with `learning_rate=0.3`, `num_leaves=255` on 20,000 rows and see validation loss bottom out at round 40 and then climb. Your manager suggests reducing the number of trees to 40. Is that the best fix? What would you change and why?"
answer: "Stopping at 40 trees is better than training to 1,000, but it keeps a model built from 40 large, aggressive steps. Each tree has up to 255 leaves fitted on 20,000 rows, so many leaves contain few samples and fit noise. At a learning rate of 0.3 each of those noisy trees also makes a large change to the predictions. Better: lower the learning rate to about 0.05 and let early stopping choose the number of trees (it will now be several hundred); shrink `num_leaves` to 15-63 and raise `min_child_samples` so leaves average over more rows; add row and column subsampling (`bagging_fraction`, `feature_fraction` around 0.7-0.8) and perhaps `lambda_l2`. Many small, regularised steps usually reach a lower minimum of validation loss than a few large ones, not just the same minimum reached sooner."
sources: [{"title": "Friedman (2001): Greedy Function Approximation: A Gradient Boosting Machine", "url": "https://doi.org/10.1214/aos/1013203451", "note": "The original gradient boosting paper: pseudo-residuals, per-leaf line search, shrinkage."}, {"title": "Chen and Guestrin (2016): XGBoost: A Scalable Tree Boosting System", "url": "https://arxiv.org/abs/1603.02754", "note": "Section 2 derives the second-order leaf weights and split gain."}, {"title": "XGBoost docs: Introduction to Boosted Trees", "url": "https://xgboost.readthedocs.io/en/stable/tutorials/model.html", "note": "A clean derivation of the regularised objective and structure score."}, {"title": "LightGBM docs: Features", "url": "https://lightgbm.readthedocs.io/en/stable/Features.html", "note": "Histogram algorithm, leaf-wise growth, and how categorical and missing values are handled."}]
practice: ["gradient-boosted-regressor", "decision-tree-regressor"]
tags: ["gradient boosting", "XGBoost", "LightGBM"]
tools: ["XGBoost", "LightGBM", "scikit-learn"]
---

## Boosting as gradient descent

AdaBoost (lesson 04.04) turned out to be stagewise minimisation of one particular loss. Friedman (2001) asked the general question: for any differentiable loss $L(y, F)$, how do you add one tree at a time to reduce $\sum_i L(y_i, F(x_i))$?

Think of the vector of predictions $(F(x_1), \dots, F(x_n))$ as the parameters. Gradient descent would move it along the negative gradient, whose $i$-th component is

$$r_i = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$$

But we need a function that also works on new inputs, not just $n$ numbers. So we fit a regression tree $h_m$ to the targets $r_i$ (the pseudo-residuals), which gives the gradient direction as a function of $x$, and take a step:

$$F_m(x) = F_{m-1}(x) + \nu\, h_m(x)$$

The learning rate (shrinkage) $\nu$ is typically 0.01-0.1. After fitting the tree's structure, each leaf's value is usually re-optimised for the actual loss (a line search per leaf), since the tree was fitted to gradients with squared error.

~~~dg-flow Gradient boosting loop. The tree only ever sees pseudo-residuals, so the loss can be anything differentiable.
Current F -(gradient of loss)-> Pseudo-residuals r -(fit small tree)-> h_m -(F += ν·h_m)-> *Updated F
~~~

## Losses and their residuals

| Loss | $L(y, F)$ | Pseudo-residual $r$ | Initial $F_0$ |
| --- | --- | --- | --- |
| Squared error | $\tfrac12 (y-F)^2$ | $y - F$ | mean of $y$ |
| Absolute error | $\lvert y - F\rvert$ | $\operatorname{sign}(y - F)$ | median of $y$ |
| Log-loss, $y \in \{0,1\}$ | $-y\ln p - (1-y)\ln(1-p)$, $p = \sigma(F)$ | $y - p$ | log-odds of base rate |
| Huber | quadratic near 0, linear in tails | clipped residual | median |

For log-loss, $F$ is the log-odds, and the optimal leaf value is a Newton step: $\sum r_i / \sum p_i(1-p_i)$ over the leaf's samples. Robust losses limit the influence of outliers, which addresses the noise sensitivity of exponential loss.

<div class="ex">
<div class="lab">Worked example</div>

**two rounds with squared error**

Targets y = [3, 5, 10]. Start at F0 = mean = 6, so residuals are [−3, −1, 4].

A stump separates the third point: leaf values are the mean residuals, −2 for the first two and 4 for the third.

With ν = 0.1, F1 = [6 − 0.2, 6 − 0.2, 6 + 0.4] = [5.8, 5.8, 6.4]. New residuals [−2.8, −0.8, 3.6] are only slightly smaller.

It takes many small steps to fit the data. That slowness is the point: each tree contributes a little, so no single tree's errors dominate.

</div>

## Regularisation

Gradient boosting overfits if run for too long, so its knobs interact:

- **Learning rate and number of trees** trade off almost exactly: halving $\nu$ roughly doubles the trees needed. Smaller $\nu$ almost always generalises a little better, at the cost of training time.
- **Early stopping**: hold out a validation set and stop when its loss has not improved for, say, 50 rounds. This makes the number of trees a fitted quantity rather than a guess.
- **Tree size**: depth 3-8 or 15-63 leaves. Depth controls the order of interactions the model can represent per tree.
- **Subsampling**: fitting each tree on a random 50-80% of rows (stochastic gradient boosting, Friedman 2002) and a fraction of columns both reduce variance and speed training.
- **Leaf constraints**: minimum samples or minimum hessian sum per leaf, and L2 penalties on leaf values.

## XGBoost: a second-order view

XGBoost (Chen and Guestrin, 2016) writes the objective at round $m$ with a second-order Taylor expansion. Let $g_i$ and $h_i$ be the first and second derivatives of the loss at the current prediction, and add a penalty $\gamma T + \tfrac12 \lambda \sum_j w_j^2$ for a tree with $T$ leaves and leaf weights $w_j$. For a fixed structure, with $G_j = \sum_{i \in j} g_i$ and $H_j = \sum_{i \in j} h_i$, the optimum is closed form:

$$w_j^* = -\frac{G_j}{H_j + \lambda} \qquad \text{Gain} = \frac12\Big[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big] - \gamma$$

Splits are scored with this gain, and a split that does not beat $\gamma$ is not made. For squared error, $h_i = 1$ and $H$ is the sample count, so $\lambda$ shrinks small leaves towards zero: a leaf with one residual of 4 gets weight $4/(1+1) = 2$ when $\lambda = 1$, not 4. For log-loss, $h_i = p_i(1-p_i)$, recovering the Newton step above.

## Making it fast: histograms and leaf-wise growth

The expensive part is split finding. Modern libraries bucket each feature into at most 255 quantile bins once, then build per-node histograms of $(G, H)$ and scan bins instead of sorted values. A useful trick: a child's histogram equals the parent's minus its sibling's, so only the smaller child needs a pass over data.

| Library | Distinctive ideas |
| --- | --- |
| XGBoost | Second-order objective with $\lambda$, $\gamma$; sparsity-aware default directions for missing values; `hist` tree method |
| LightGBM | Histograms; leaf-wise (best-first) growth capped by `num_leaves`; GOSS keeps large-gradient rows and samples small ones; bundles mutually exclusive sparse features |
| CatBoost | Symmetric (oblivious) trees; ordered boosting and ordered target statistics to avoid target leakage with categorical features |
| scikit-learn `HistGradientBoosting*` | LightGBM-style histograms, native missing values and categoricals, built-in early stopping |

Leaf-wise growth reaches lower loss with the same number of leaves than depth-wise growth, but can grow deep, narrow branches on small data; cap `num_leaves` and set `min_child_samples`. All four handle missing values natively by learning which side of each split they go to.

<div class="take">
<div class="lab">Key takeaway</div>

gradient boosting adds small trees fitted to the negative gradient of any differentiable loss; tune the learning rate, tree size and subsampling, and let early stopping pick the number of trees.

</div>
