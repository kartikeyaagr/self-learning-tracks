---
id: "06.04"
course: "machine-learning"
module: "06"
slug: "cross-validation-strategies"
title: "Cross-validation strategies"
keyIdeas: ["Cross-validation estimates how well *the procedure* does when trained on about $(k-1)/k$ of your data. It is not the error of the one model you will deploy, and its fold-to-fold spread understates the real uncertainty.", "The splitter must reproduce deployment: **stratified** for rare classes, **grouped** when rows share an entity, **forward-chaining with a gap** for time, and stratified-grouped when you need both.", "When you tune on CV scores, the best score is biased upwards. **Nested CV** wraps the whole tuning procedure in an outer loop to estimate it honestly, at the cost of (outer folds) x (inner folds) x (configs) fits.", "To compare two models, use the same folds and look at **paired** per-fold differences, with a variance correction for overlapping training sets (Nadeau and Bengio).", "After CV picks a configuration, refit it on all the training data; the CV score is the estimate you report for that refit."]
question: "You have 600 labelled patients (several visits each, 8% positive), and you will tune a gradient-boosted model over 40 configurations. A teammate runs GroupKFold(5) over the grid and reports the best mean AUC, 0.88, as the expected deployment performance. What is wrong with the number, and what would you run instead, roughly how many fits would it take, and what would you do if that is too expensive?"
answer: "The 0.88 is the maximum over 40 noisy CV estimates, so it is biased upwards: with only about 48 positive patients per fold, the noise between configurations is large and the winner partly won by luck. Also plain GroupKFold does not balance the rare class, so some folds may have very few positives. Run **nested CV** with StratifiedGroupKFold in both loops (grouped by patient): for each of 5 outer folds, tune over the 40 configs with inner 5-fold CV, refit the winner on the outer training fold and score it on the outer test fold. That is about 5 x (40 x 5 + 1) = 1,005 fits. Report the mean and spread of the 5 outer scores. If too expensive, use random search over fewer configs or successive halving in the inner loop, or reserve a grouped held-out test set once and tune with ordinary CV on the rest."
sources: [{"title": "Cross-validation: what does it estimate and how well does it do it? (Bates, Hastie, Tibshirani)", "url": "https://arxiv.org/abs/2104.00673", "note": "Why CV estimates the procedure's average error and why naive CV intervals undercover; proposes nested CV for intervals."}, {"title": "Bias in error estimation when using cross-validation for model selection (Varma and Simon)", "url": "https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-91", "note": "Open-access demonstration of optimistic bias from tuning on CV and the nested CV fix."}, {"title": "scikit-learn: Statistical comparison of models using grid search", "url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html", "note": "Implements the Nadeau-Bengio corrected t-test and a Bayesian comparison on CV results."}, {"title": "scikit-learn: Nested versus non-nested cross-validation", "url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html", "note": "Short runnable example showing the optimism of non-nested scores."}]
practice: ["k-fold-cross-validation", "stacking-ensemble", "hashed-dataset-split", "bootstrap-confidence-interval"]
tags: ["cross-validation", "nested CV", "model comparison"]
tools: ["scikit-learn"]
---

## What lesson 01.03 left open

Lesson 01.03 introduced k-fold CV, its stratified, grouped and time-series variants, and the leakage that random splits invite. This lesson is about using CV well: what number it actually estimates, how to choose $k$ and the splitter, how nested CV keeps tuning honest, and how to decide whether one model really beats another.

## What CV estimates

Each fold trains a fresh model on $n(k-1)/k$ rows. The average fold score is therefore an estimate of the expected error of *the training procedure* applied to datasets of that size, not of the particular model you will refit on all $n$ rows. Bates, Hastie and Tibshirani (2021) show this formally and also show why the naive standard error, the fold standard deviation divided by $\sqrt{k}$, is too small: the folds share most of their training data, so their errors are positively correlated.

The choice of $k$ trades three things:

- **Bias**: with $k = 5$ each model sees 80% of the data, so if the learning curve (lesson 01.05) is still rising, CV is pessimistic. Larger $k$ reduces this.
- **Variance**: leave-one-out ($k = n$) is nearly unbiased but its estimate can be highly variable, and each fold's score is a single 0/1 value for classification.
- **Cost**: $k$ fits. For most tabular problems $k = 5$ or $10$ is a good default; **repeated** k-fold (several shuffles of the fold assignment) reduces the variance that comes from one particular partition.

## Choosing the splitter

| Data | Splitter | Why
| --- | --- | --- |
| Rare class, i.i.d. rows | StratifiedKFold | every fold has enough positives to score
| Regression with a skewed target | stratify on binned target | folds see similar target ranges
| Several rows per user, patient, device | GroupKFold | no entity on both sides
| Rare class and groups | StratifiedGroupKFold | both constraints at once
| Time-ordered data | TimeSeriesSplit with gap | train on the past only
| Labels that span time windows | forward chaining with purge and embargo | no training label overlaps the test window

For time series the **gap** matters whenever features or labels use windows. If a label is "churn in the next 30 days" and features use the last 90 days, a training row from the day before the cut has a label window that overlaps the test period. Drop (purge) training rows whose label windows reach into the test block, and leave a gap (embargo) after it before training rows resume in expanding-window schemes.

```python
from sklearn.model_selection import TimeSeriesSplit, StratifiedGroupKFold
tss = TimeSeriesSplit(n_splits=5, gap=30)            # rows sorted by time; 30 rows skipped at each cut
sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for tr, va in sgk.split(X, y, groups=user_id):
    ...
```

## Nested CV: honest numbers after tuning

If you try 50 configurations and report the best mean CV score, you have selected the maximum of 50 noisy estimates, and that maximum is optimistic. Varma and Simon (2006) showed the bias can be large on small datasets. **Nested CV** fixes this by treating the entire selection procedure as the thing being evaluated.

~~~dg-flow Nested CV: the inner loop chooses, the outer loop grades the choosing. The outer test fold never influences any choice.
Data -(outer split)-> Outer train fold -(inner CV over configs)-> *Best config -(refit on outer train)-> Model -(score)-> Outer test fold
Outer test scores -(average over outer folds)-> Estimate of tuned-procedure error
~~~

Each outer fold may pick a *different* best configuration; that is expected. The outer score is the estimate for "run this tuning procedure and deploy what it picks". Cost multiplies: 5 outer x 5 inner x 20 configs = 500 fits, plus refits. With a large dataset a single untouched test set does the same job much more cheaply; nested CV earns its cost when data is scarce.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
inner = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1, 10]},
                     cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="roc_auc")
outer_scores = cross_val_score(inner, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=2), scoring="roc_auc")
```

## Comparing two models

Use the **same folds** for both models and analyse the per-fold *differences* $d_j$. Pairing removes the variation that comes from fold difficulty, which is usually larger than the difference between models. A plain paired t-test on $d_j$ is still overconfident because the training sets overlap. Nadeau and Bengio's corrected resampled t-test inflates the variance:

$$t = \frac{\bar d}{\sqrt{\left(\frac1J + \frac{n_{te}}{n_{tr}}\right)\hat\sigma_d^2}}$$

where $J$ is the number of resamples and $n_{te}/n_{tr}$ the test-to-train size ratio (0.25 for 5-fold). scikit-learn's model-comparison example implements this test and a Bayesian alternative.

<div class="ex">
<div class="lab">Worked example</div>

**is 0.6 of a point real?**

Five-fold CV repeated twice gives $J = 10$ paired AUC differences, model B minus model A: mean $\bar d = 0.006$, standard deviation 0.008.

Naive paired t: $0.006 / (0.008/\sqrt{10}) = 2.37$, "significant at 5%". Corrected: the variance factor is $1/10 + 0.2/0.8 = 0.35$, so the standard error is $0.008 \times \sqrt{0.35} = 0.0047$ and $t = 1.27$, nowhere near significant. Unless B is cheaper or simpler, keep A, or collect more data before claiming a win.

</div>

## After CV

CV picks a configuration; it does not hand you the model. Refit the chosen pipeline on all training data and report the CV (or nested CV, or test) score as its expected performance. Keep fold assignments fixed (seed them, or hash ids as in lesson 01.03) so that later experiments are compared on the same folds. And keep the predictions each fold made on its held-out rows: these **out-of-fold predictions** are what stacking and target encoding (lesson 07.02) are built from.

<div class="take">
<div class="lab">Key takeaway</div>

choose the splitter that mimics deployment, wrap tuning in nested CV or a final untouched test set, and compare models with paired, variance-corrected differences on shared folds.

</div>
