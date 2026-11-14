---
id: "04.03"
course: "machine-learning"
module: "04"
slug: "bagging-and-random-forests"
title: "Bagging and random forests"
keyIdeas: ["Averaging $B$ predictors with variance $\\sigma^2$ and pairwise correlation $\\rho$ gives variance $\\rho\\sigma^2 + (1-\\rho)\\sigma^2/B$: more trees kill the second term, only decorrelation shrinks the first.", "Bagging trains each tree on a bootstrap sample (size $n$, with replacement). Each sample contains about 63.2% of the distinct points, because $(1-1/n)^n \\to e^{-1} \\approx 0.368$.", "Random forests also sample a random subset of features at every split (classically $\\sqrt{d}$ for classification), which decorrelates trees that would otherwise all split on the same strong feature.", "Out-of-bag (OOB) error scores each point using only trees that did not see it: a nearly free validation estimate.", "Adding trees never causes overfitting in the usual sense, but returns diminish quickly; `max_features` and leaf size are the knobs that matter."]
question: "A colleague's random forest regressor on 40 features barely beats a single tuned tree, and adding trees from 100 to 2000 changes nothing. They used scikit-learn defaults. What is the likely cause, and what would you change first?"
answer: "The flat curve with more trees says the variance term $(1-\\rho)\\sigma^2/B$ is already negligible; what remains is the correlation floor $\\rho\\sigma^2$ plus bias. With scikit-learn's `RandomForestRegressor`, the default `max_features=1.0` considers all 40 features at every split, so the model is plain bagging: if one or two features dominate, almost every tree has the same top splits and the trees are highly correlated. I would first lower `max_features`, trying values such as 0.33 (the classic $d/3$), `\"sqrt\"` and 0.5, and compare by OOB score. If that does not help, I would check `min_samples_leaf` (noisy targets favour larger leaves) and whether the problem needs extrapolation or a smooth trend, which no tree ensemble provides."
sources: [{"title": "Breiman (2001): Random Forests", "url": "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf", "note": "The original paper: strength vs correlation bound, feature subsampling, OOB estimates."}, {"title": "Breiman (1996): Bagging Predictors", "url": "https://www.stat.berkeley.edu/~breiman/bagging.pdf", "note": "Why bagging helps unstable procedures and does little for stable ones."}, {"title": "scikit-learn User Guide: Ensembles (forests)", "url": "https://scikit-learn.org/stable/modules/ensemble.html", "note": "Section on random forests and Extra-Trees, including current max_features defaults and parameter advice."}, {"title": "scikit-learn example: OOB errors for random forests", "url": "https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html", "note": "OOB error as a function of the number of trees for different max_features."}]
practice: ["random-forest", "bagging-classifier", "bootstrap-confidence-interval"]
tags: ["bagging", "random forest", "OOB error"]
tools: ["scikit-learn"]
---

## Why averaging helps

Lesson 04.02 ended with a tension: deep trees have low bias but high variance. Bagging resolves it by averaging many deep trees. The arithmetic is worth knowing exactly. If each of $B$ predictors has variance $\sigma^2$ and every pair has correlation $\rho$, the variance of their average is

$$\operatorname{Var}\Big(\frac{1}{B}\sum_{b=1}^{B} f_b(x)\Big) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2$$

The second term goes to zero as you add trees. The first does not: it is a floor set by how similar the trees are. Bias is unchanged by averaging, since each tree has the same expected prediction. So an ensemble of averaged trees wants members that are individually low-bias (deep) and mutually as different as possible.

<div class="ex">
<div class="lab">Worked example</div>

**why decorrelation beats more trees**

Take σ² = 1 for a single deep tree's prediction at some point.

Plain bagging, trees correlated at ρ = 0.5, B = 100: variance = 0.5 + 0.5/100 = 0.505. Going to B = 1000 only reaches 0.5005.

A random forest that cuts correlation to ρ = 0.2, still B = 100: variance = 0.2 + 0.8/100 = 0.208.

Decorrelation more than halved the variance; a tenfold increase in trees bought almost nothing.

</div>

## Bootstrap aggregation

Breiman's bagging (1996) creates diversity by giving each tree a different dataset. Each tree is trained on a bootstrap sample: $n$ draws with replacement from the $n$ training points. The chance that a given point is never drawn is $(1 - 1/n)^n$, which tends to $e^{-1} \approx 0.368$. So each bootstrap sample contains about 63.2% of the distinct points, some of them several times, and leaves out roughly a third.

For classification the ensemble either takes a majority vote or, better, averages the trees' class-probability estimates (scikit-learn does the latter). For regression it averages predictions.

~~~dg-flow Random forest training and prediction. Trees are independent, so training is embarrassingly parallel.
Training set -(bootstrap b)-> Sample b -(random √d features per split)-> Deep tree b -(average over b)-> *Forest prediction
~~~

Bagging helps unstable learners: those whose output changes a lot when the data changes, such as deep trees. It does little for stable ones like linear regression or k-NN with large $k$, because their bootstrap replicas are nearly identical ($\rho$ close to 1).

## Random forests: feature subsampling

Bagged trees remain highly correlated when one or two features dominate: almost every bootstrap tree puts the same strong feature at the root and grows similar structure beneath it. Breiman's random forest (2001) adds a second source of randomness. At every split, only a random subset of `max_features` = $m$ features is considered. Classic defaults are $m = \sqrt{d}$ for classification and $m = d/3$ for regression. In current scikit-learn, `RandomForestClassifier` uses `"sqrt"` but `RandomForestRegressor` defaults to all features (`1.0`), which makes it plain bagging unless you change it.

Smaller $m$ lowers $\rho$ but raises each tree's bias and variance, because trees sometimes have to split on weak features. The best $m$ is data-dependent and worth tuning. One failure mode is a dataset with a few informative features buried among hundreds of noise features: with $m = \sqrt{d}$, many splits see only noise candidates, and the forest degrades. There, larger $m$ helps.

Extremely randomised trees (Extra-Trees) push further: they also draw the threshold at random for each candidate feature and usually skip bootstrapping. They are faster to train and often equally accurate.

## Out-of-bag error

Because each tree skipped about 37% of the data, each training point has roughly $0.37B$ trees that never saw it. Predicting each point with only those trees and scoring the result gives the out-of-bag (OOB) error. It behaves much like a cross-validation estimate, comes nearly free during training (`oob_score=True`), and is a convenient way to choose the number of trees or `max_features` on moderately sized data. With very few trees it is noisy, and some points may have no OOB trees at all.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                            min_samples_leaf=1, oob_score=True, n_jobs=-1, random_state=0)
rf.fit(X_tr, y_tr)
print(rf.oob_score_)        # accuracy on out-of-bag predictions
```

## Practical behaviour and limits

- **Number of trees**: error falls fast then flattens; 200-1000 is typical. More trees never make test error worse in expectation, but cost memory and latency linearly.
- **Leaf size and depth**: defaults (fully grown) are strong; raise `min_samples_leaf` for noisy regression targets and smaller models.
- **No extrapolation**: like single trees, a forest's predictions stay inside the training target range, which matters for trending time series.
- **Size**: 500 deep trees on a million rows can occupy gigabytes. Capping `max_leaf_nodes` or depth shrinks models with little loss.
- **Probabilities**: averaged leaf frequencies are typically under-confident near 0 and 1; calibrate if you need probabilities (module 03).

Random forests are a strong default baseline for tabular data: few critical hyperparameters, little tuning, and OOB error for free. Boosting (next two lessons) usually wins on accuracy once tuned.

<div class="take">
<div class="lab">Key takeaway</div>

averaging deep trees removes variance only down to the floor $\rho\sigma^2$, so random forests add per-split feature sampling to lower $\rho$; bootstrapping also gives OOB error as a free validation estimate.

</div>
