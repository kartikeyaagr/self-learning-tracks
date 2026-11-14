---
id: "06.04"
course: "machine-learning"
module: "06"
slug: "cross-validation-strategies"
title: "Cross-validation strategies"
keyIdeas: ["Cross-validation estimates how well *the procedure* does when trained on about $(k-1)/k$ of your data. It is not the error of the one model you will deploy, and its fold-to-fold spread understates the real uncertainty because the folds share training rows.", "$k = 5$ or $10$ is a good default: smaller $k$ is pessimistic (less training data), leave-one-out is noisy and expensive, and **repeated** k-fold averages away the luck of one partition.", "The splitter must reproduce deployment: **stratified** for rare classes, **grouped** when rows share an entity, **forward-chaining with a gap** for time, and stratified-grouped when you need both.", "When you tune on CV scores, the best score is biased upwards. **Nested CV** wraps the whole tuning procedure in an outer loop to estimate it honestly, at the cost of (outer folds) x (inner folds) x (configs) fits.", "To compare two models, use the same folds and look at **paired** per-fold differences, with a variance correction for overlapping training sets (Nadeau and Bengio).", "After CV picks a configuration, refit it on all the training data; the CV score is the estimate you report for that refit, and the out-of-fold predictions are reusable for stacking."]
question: "You have 600 labelled patients (several visits each, 8% positive), and you will tune a gradient-boosted model over 40 configurations. A teammate runs GroupKFold(5) over the grid and reports the best mean AUC, 0.88, as the expected deployment performance. What is wrong with the number, and what would you run instead, roughly how many fits would it take, and what would you do if that is too expensive?"
answer: "Two things are wrong. First, 0.88 is the **maximum over 40 noisy CV estimates**, so it is biased upwards: with 600 patients and 8% positives there are only about 48 positive patients in all, roughly 10 per fold, so AUC on each fold is noisy and the winning configuration won partly by luck (the same effect that pushes the best score above chance even on random labels). Second, plain GroupKFold keeps patients together, which is right, but it does not balance the rare class, so some folds may hold only a handful of positives and give erratic AUCs.\n\nRun **nested CV** with StratifiedGroupKFold in both loops, grouped by patient. For each of 5 outer folds: hold the outer test fold aside, tune over the 40 configs with inner 5-fold CV on the outer training fold, refit the winner on the whole outer training fold, and score it once on the outer test fold. That is about 5 x (40 x 5 + 1) = 1,005 fits. Report the mean and the spread of the 5 outer AUCs, and note which configuration each outer fold picked (if they disagree wildly, the tuning itself is unstable). Then rerun the tuning once on all 600 patients and deploy that refit, quoting the nested estimate as its expected performance.\n\nIf 1,005 boosted fits are too expensive: use random search over fewer configurations or successive halving in the inner loop; use 3 inner folds instead of 5; or set aside a grouped, stratified held-out test set of patients once, tune with ordinary CV on the rest, and report the score on the held-out patients only (accepting a noisier number because it rests on fewer positives). Also check that all preprocessing sits inside the pipeline so that nothing leaks across patients."
sources: [{"title": "Cross-validation: what does it estimate and how well does it do it? (Bates, Hastie, Tibshirani)", "url": "https://arxiv.org/abs/2104.00673", "note": "Why CV estimates the procedure's average error and why naive CV intervals undercover; proposes nested CV for intervals."}, {"title": "Bias in error estimation when using cross-validation for model selection (Varma and Simon)", "url": "https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-91", "note": "Open-access demonstration of optimistic bias from tuning on CV and the nested CV fix."}, {"title": "scikit-learn: Statistical comparison of models using grid search", "url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html", "note": "Implements the Nadeau-Bengio corrected t-test and a Bayesian comparison on CV results."}, {"title": "scikit-learn: Nested versus non-nested cross-validation", "url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html", "note": "Short runnable example showing the optimism of non-nested scores."}]
practice: ["k-fold-cross-validation", "stacking-ensemble", "hashed-dataset-split", "bootstrap-confidence-interval"]
tags: ["cross-validation", "nested CV", "model comparison"]
tools: ["scikit-learn"]
---

## What you will learn

- What the number that comes out of cross-validation actually estimates, and why it is not the score of the model you will deploy.
- How to pick $k$, and why repeating k-fold with new shuffles gives a steadier answer.
- How to choose the splitter (stratified, grouped, time-ordered) so that every fold looks like deployment.
- Why the best score after tuning is too optimistic, and how **nested CV** gives an honest number.
- How to decide whether model B really beats model A, using paired per-fold differences and a corrected t-test.

*Before you start:* lesson 01.03 (train/validation/test splits, plain k-fold, leakage) and lesson 01.05 (learning curves). Lessons 06.01 to 06.03 gave you the metrics (accuracy, AUC, MAE) that CV averages. A t-test is used once near the end; the example spells out every step.

Lesson 06.03 chose the right *metric*; you also need a trustworthy way to *measure* it on unseen data. Lesson 01.03 introduced k-fold cross-validation for that. This lesson is about using it well: what the number means, which splitter to use, how to keep tuning honest, and how to compare models without fooling yourself.

## The idea in one picture

You have 150 labelled customers and want to know how accurate a logistic regression will be on next month's customers. Holding out 30 rows wastes data and gives a noisy answer from only 30 rows. **K-fold cross-validation (CV)** does better: cut the data into $k$ equal parts called **folds**, train on $k-1$ of them, score on the one left out, and repeat until every fold has been the test fold once. With $k = 5$:

| Round | Train on folds | Test on fold | Accuracy
| --- | --- | --- | --- |
| 1 | 2, 3, 4, 5 | 1 | 0.82
| 2 | 1, 3, 4, 5 | 2 | 0.78
| 3 | 1, 2, 4, 5 | 3 | 0.85
| 4 | 1, 2, 3, 5 | 4 | 0.80
| 5 | 1, 2, 3, 4 | 5 | 0.75

Every row is tested once and trained on four times. The **CV score** is the average, 0.80; the spread (0.75 to 0.85) hints at how much luck is in any single split.

How the folds are cut matters most: a fold is a fair exam only if it resembles the data the model will meet later. The figure shows four standard ways to cut 24 rows into 4 folds.

~~~dg-fig 06.04-splitters.svg
Each grid row is one round of CV: blue rows train, orange rows test, grey rows are unused. Stratified folds each get the same share of positives; group folds keep all of a patient's rows together; time-series folds train only on earlier rows and leave a gap before the test block.
~~~

New words:

- A **splitter** is the rule that assigns rows to folds (KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit in scikit-learn).
- **Out-of-fold (OOF) predictions** are the predictions each row received while it was in the test fold. Every row gets exactly one, from a model that never saw it.

## Step 1: know what the CV number estimates

Each round trains a *fresh* model on $n(k-1)/k$ rows. With $n = 150$ and $k = 5$ that is 120 rows. So the CV score answers this question: "if I run this whole training recipe on a random dataset of about 120 rows like mine, how well does the result do on new data, on average?"

**It grades the procedure, not one model.** The five fold models are thrown away; the model you deploy is a sixth one, refitted on all 150 rows, which CV never scored. Bates, Hastie and Tibshirani (2021) show that CV tracks the *average* error of the procedure much better than the error of the particular model in your hand.

**Its error bar is too narrow.** The obvious standard error is the fold standard deviation divided by $\sqrt{k}$:

$$\text{SE}_{\text{naive}} = \frac{s}{\sqrt{k}}, \qquad s = \sqrt{\frac{1}{k-1}\sum_{j=1}^{k}\big(a_j - \bar a\big)^2},$$

where $a_j$ is the score on fold $j$ and $\bar a$ their mean. It assumes the $k$ scores are independent. They are not: any two training sets share $(k-2)/k$ of the data (60% for $k = 5$), so the fold models are right or wrong together. Treat $\text{SE}_{\text{naive}}$ as a lower bound.

<div class="ex">
<div class="lab">Worked example</div>

**score and naive error bar for the table above**

Fold accuracies: $0.82, 0.78, 0.85, 0.80, 0.75$. Sum $= 4.00$, so the CV score is $\bar a = 4.00 / 5 = 0.80$.

Deviations from the mean: $+0.02, -0.02, +0.05, 0.00, -0.05$. Squares: $0.0004, 0.0004, 0.0025, 0, 0.0025$, which sum to $0.0058$.

Sample variance: $0.0058 / (5 - 1) = 0.00145$, so $s = \sqrt{0.00145} = 0.038$.

Naive standard error: $0.038 / \sqrt{5} = 0.038 / 2.236 = 0.017$. A naive 95% interval would be about $0.80 \pm 2 \times 0.017$, that is 0.77 to 0.83.

The folds share training rows, so the true uncertainty is wider. Report "0.80, fold SD 0.04", not three-digit precision.

</div>

## Step 2: choose k (and consider repeating)

The choice of $k$ trades three things.

- **Bias.** Each fold model sees only a fraction $(k-1)/k$ of the data. If the learning curve is still rising at your dataset size, smaller training sets mean worse models, so CV is **pessimistic**. $k = 2$ trains on half the data and is the most pessimistic; larger $k$ shrinks this bias.
- **Variance.** **Leave-one-out (LOO)**, where $k = n$, is nearly unbiased, but its $n$ fold models are almost identical, so their errors are correlated and the estimate can swing a lot between datasets. In classification each LOO fold scores one row: 0 or 1.
- **Cost.** $k$ folds means $k$ fits.

~~~dg-fig 06.04-choose-k.svg
Left: the learning curve is still rising, so models trained on k=2 or k=5 training folds are a little worse than one trained on all 150 rows; CV inherits that pessimism. Right: the same 150 rows, cut 40 different ways, give noticeably different CV scores.
~~~

In numbers: a model trained on all 150 rows reaches about 0.77, while the average CV estimate is 0.72 with $k = 2$, 0.745 with $k = 5$ and 0.75 with $k = 10$. The right panel shows a second source of noise: one particular random partition. With $k = 2$ the 40 shuffles range from 0.66 to 0.78. **Repeated k-fold** reshuffles the fold assignment several times (say 5 x 5-fold = 25 fits) and averages, which smooths out partition luck. It does not remove the noise from having only 150 rows in the first place.

Default for tabular data: 5-fold or 10-fold, repeated if fits are cheap and data is small. With hundreds of thousands of rows a single validation split is already precise.

## Step 3: choose the splitter that mimics deployment

Plain KFold assumes rows are independent and interchangeable. Real data often breaks that assumption in one of three ways, and each needs its own splitter.

| Data | Splitter | Why
| --- | --- | --- |
| Rare class, independent rows | StratifiedKFold | every fold has enough positives to score
| Regression with a skewed target | stratify on a binned target | folds see similar target ranges
| Several rows per user, patient, device | GroupKFold | no entity on both sides
| Rare class and groups | StratifiedGroupKFold | both constraints at once
| Time-ordered data | TimeSeriesSplit with gap | train on the past only
| Labels that span time windows | forward chaining with purge and embargo | no training label overlaps the test window

**Stratification.** With 8% positives, random folds can differ twofold in positives, making per-fold AUC or recall noisy. Stratified folds copy the class proportions into every fold.

**Groups.** If a patient contributes eight visits and random CV puts five in training and three in testing, the model can recognise the patient instead of learning the disease. The score then measures memory, not generalisation. The figure shows how large this effect can be.

~~~dg-fig 06.04-leak.svg
The same forest, three numbers. Random KFold lets each patient's visits sit on both sides of the split and reports an AUC of 1.00; GroupKFold reports 0.64, close to the 0.58 actually achieved on brand-new patients.
~~~

**Time.** If you predict the future from the past, every fold must train only on the past: TimeSeriesSplit grows the training window and tests on the next block. The **gap** matters when labels use windows. With a label "churn in the next 30 days", a training row from the day before the cut has a label window inside the test period. Drop (**purge**) such training rows, and leave a gap (**embargo**) after the test block before training rows resume.

```python
from sklearn.model_selection import TimeSeriesSplit, StratifiedGroupKFold
tss = TimeSeriesSplit(n_splits=5, gap=30)            # rows sorted by time; 30 rows skipped at each cut
sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for tr, va in sgk.split(X, y, groups=user_id):
    ...
```

Quick test: say what the model faces in production ("a new patient", "next week"). Each test fold should contain exactly that kind of unseen thing.

## Step 4: keep tuning honest with nested CV

Tuning means trying many configurations and keeping the best CV score. That score is the *maximum* of many noisy estimates, and a maximum of noisy numbers is biased upwards: the winner won partly because its folds were kind to it. Varma and Simon (2006) showed this bias can be large on small datasets.

The figure makes it stark. The labels are coin flips, so no configuration can truly beat 50% accuracy, yet the best CV score climbs steadily as more configurations are tried, reaching 0.64 after 40. Nested CV stays at 0.50.

~~~dg-fig 06.04-optimism.svg
On data whose labels are pure noise, the best CV score creeps upward with every extra configuration tried. Nested CV evaluates the whole "try them all, keep the best" procedure and stays at the true 0.5.
~~~

**Nested CV** fixes this by treating the entire selection procedure as the thing being evaluated. An inner CV loop chooses; an outer CV loop grades the choosing.

~~~dg-flow Nested CV: the inner loop chooses, the outer loop grades the choosing. The outer test fold never influences any choice.
Data -(outer split)-> Outer train fold -(inner CV over configs)-> *Best config -(refit on outer train)-> Model -(score)-> Outer test fold
Outer test scores -(average over outer folds)-> Estimate of tuned-procedure error
~~~

Step by step, for each outer fold:

1. Set the outer test fold aside. Do not look at it.
2. On the outer training fold, run ordinary CV over every configuration and pick the best.
3. Refit that configuration on the whole outer training fold.
4. Score it once on the outer test fold.

Average the outer scores. Outer folds may pick *different* configurations; that is expected and shows how stable tuning is. The outer score estimates "run this tuning procedure and deploy what it picks".

The cost multiplies: 5 outer x 5 inner folds x 20 configurations = 500 fits, plus 5 refits. With a large dataset a single untouched test set does the same job far more cheaply; nested CV earns its cost when data is scarce.

## Step 5: compare two models fairly

Model B scores 0.006 AUC higher than model A. Is that real? Two rules.

**Pair the folds.** Run both models on the *same* folds and look at the per-fold differences $d_j = b_j - a_j$. Fold difficulty usually varies far more than the gap between models; pairing cancels it out. In the figure, model A's AUC ranges from 0.78 to 0.96 across folds, yet B beats A on 9 of 10 folds by an average of 0.018.

~~~dg-fig 06.04-paired.svg
Left: per-fold AUC for two models. The folds differ a lot in difficulty, and both models move together. Right: the per-fold differences remove the shared difficulty and show the actual comparison.
~~~

**Correct the variance.** A plain paired t-test on the $d_j$ is still overconfident, for the same reason as in step 1: overlapping training sets make the differences correlated. Nadeau and Bengio's **corrected resampled t-test** inflates the variance:

$$t = \frac{\bar d}{\sqrt{\left(\frac1J + \frac{n_{te}}{n_{tr}}\right)\hat\sigma_d^2}}$$

Here $J$ is the number of resamples (folds times repeats), $\bar d$ the mean difference, $\hat\sigma_d^2$ the sample variance of the differences, and $n_{te}/n_{tr}$ the ratio of test to training size (0.25 for 5-fold, because each test fold is a quarter the size of its training set). Compare $t$ against a t-distribution with $J - 1$ degrees of freedom. Without the $n_{te}/n_{tr}$ term this is the ordinary paired t-test.

<div class="ex">
<div class="lab">Worked example</div>

**is 0.6 of a point real?**

Five-fold CV repeated twice gives $J = 10$ paired AUC differences, model B minus model A: mean $\bar d = 0.006$, standard deviation $\hat\sigma_d = 0.008$.

Naive paired t: standard error $0.008/\sqrt{10} = 0.008/3.162 = 0.00253$, so $t = 0.006 / 0.00253 = 2.37$. With 9 degrees of freedom that gives $p \approx 0.04$: "significant at 5%".

Corrected: the variance factor is $1/10 + 0.2/0.8 = 0.10 + 0.25 = 0.35$. The standard error becomes $0.008 \times \sqrt{0.35} = 0.008 \times 0.592 = 0.0047$, and $t = 0.006 / 0.0047 = 1.27$, giving $p \approx 0.24$. Nowhere near significant.

Decision: unless B is cheaper or simpler, keep A, or collect more data before claiming a win.

</div>

## Cross-validation in code

The snippet runs the three ideas end to end on 600 rows from 150 "patients" with about 10% positives: grouped and stratified CV, nested CV around a grid search, and a corrected comparison.

```python
import numpy as np
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedGroupKFold, GridSearchCV,
                                     cross_val_score, RepeatedStratifiedKFold)

X, y = make_classification(n_samples=600, n_features=20, n_informative=5,
                           weights=[0.9], flip_y=0.05, random_state=0)
patient = np.arange(600) // 4                       # 150 patients, 4 rows each

# 1) plain CV with the right splitter: stratified AND grouped
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
s = cross_val_score(pipe, X, y, cv=sgk, groups=patient, scoring="roc_auc")
print(s.round(3), round(s.mean(), 3))   # [0.883 0.9 0.889 0.82 0.894] 0.877

# 2) nested CV: the inner GridSearchCV is the "model" the outer loop grades
inner = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1, 10]},
                     cv=StratifiedGroupKFold(5, shuffle=True, random_state=1),
                     scoring="roc_auc")
outer = cross_val_score(inner, X, y, cv=sgk, groups=patient, scoring="roc_auc",
                        params={"groups": patient})   # inner search splits by patient too
print(outer.round(3), round(outer.mean(), 3))  # [0.871 0.933 0.893 0.827 0.9] 0.885

# 3) paired comparison on shared folds, Nadeau-Bengio corrected t-test
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
a = cross_val_score(pipe, X, y, cv=rcv, scoring="roc_auc")
b = cross_val_score(RandomForestClassifier(300, random_state=0), X, y, cv=rcv, scoring="roc_auc")
d = b - a; J = len(d); ratio = 1 / 4                 # n_test / n_train for 5-fold
t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1))
p = 2 * stats.t.sf(abs(t), J - 1)
print(round(d.mean(), 4), round(t, 2), round(p, 3))  # -0.0206 -0.63 0.547
```

The scaler sits *inside* the pipeline, so it is refitted on each training fold; the shared `rcv` gives both models identical folds. The forest is 0.02 AUC worse but $p = 0.55$: we cannot tell them apart, so the simpler model wins by default.

## Reading the result

Before trusting a CV number, check these:

- **Look at the per-fold scores, not just the mean.** One fold far below the others often points to a slice (a region, a period) the model handles badly.
- **Compare with a dumb baseline** (`DummyClassifier`) on the same folds. The improvement is what the model adds.
- **Sanity-check the splitter.** Print each fold's positives, groups and date range; assert no group appears on both sides.
- **Expect the final test score to land in the CV range.** Clearly below it means leakage or a splitter that did not mimic deployment.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Preprocessing fitted on all rows before CV | CV clearly above the test score | put scaling, imputation, feature selection and encoding inside a Pipeline
| Random KFold on grouped rows | near-perfect CV, poor on new entities | GroupKFold or StratifiedGroupKFold
| Shuffled KFold on time series | great CV, weak after launch | TimeSeriesSplit with a gap; purge overlapping labels
| Reporting the best tuned CV score | production below every reported number | nested CV or a final untouched test set
| Different folds for each model | comparisons flip when the seed changes | one fixed `cv` object or seeded fold ids shared by all runs
| Naive t-test on fold scores | "significant" wins that do not replicate | corrected resampled t-test, repeated CV

A useful smoke test for the whole pipeline: shuffle the labels and run your exact CV code. The score should drop to chance. If it does not, something is leaking.

## After CV

CV picks a configuration; it does not hand you the model. Refit the chosen pipeline on all training data and report the CV (or nested CV, or test) score as its expected performance. The refit sees more data than any fold model, so it is usually slightly better than the estimate: the safe direction to be wrong in. Keep fold assignments fixed (seed them, or hash ids as in lesson 01.03) so that later experiments are compared on the same folds. And keep the out-of-fold predictions: they are what stacking and target encoding (lesson 07.02) are built from.

## Going deeper

**LOO for linear models.** For least squares the LOO residual of row $i$ is $r_i / (1 - h_{ii})$, where $h_{ii}$ is its leverage (lesson 02.01). So LOO costs one fit, but the estimate leans heavily on a few high-leverage rows.

**No unbiased variance estimator exists.** Bengio and Grandvalet (2004) proved that no estimator of the variance of the k-fold CV estimate is unbiased for every distribution. The Nadeau-Bengio factor $n_{te}/n_{tr}$ is a heuristic that works well in practice, not an exact fix.

**Better intervals.** Bates, Hastie and Tibshirani propose a nested scheme that estimates the error of the CV estimate itself and widens the interval so it reaches its stated coverage. scikit-learn's model-comparison example also shows a Bayesian alternative to the t-test.

**Selection without nesting.** Cheaper options: tune with ordinary CV and report a separate held-out test set (most common), or apply a bias correction to the best score (Tibshirani and Tibshirani, 2009).

## Key terms

| Term | Meaning
| --- | --- |
| K-fold CV | Split data into $k$ folds; train on $k-1$, test on the rest, rotate, average
| Fold | One of the $k$ parts; each serves as the test set once
| Splitter | The rule that assigns rows to folds (stratified, grouped, time-ordered)
| Out-of-fold prediction | The prediction a row received while it was held out
| Repeated k-fold | Several reshuffled k-fold runs averaged to cut partition noise
| Nested CV | Inner CV chooses the configuration; outer CV scores the whole choosing procedure
| Selection bias | The upward bias of the best score among many noisy estimates
| Purge / embargo | Dropping training rows whose label windows overlap the test period, and a gap after it
| Corrected resampled t-test | Paired t-test on fold differences with variance inflated for overlapping training sets

<div class="take">
<div class="lab">Key takeaway</div>

CV estimates how well your training procedure does on data like yours, so choose the splitter that mimics deployment, wrap tuning in nested CV or a final untouched test set, and compare models with paired, variance-corrected differences on shared folds.

</div>
