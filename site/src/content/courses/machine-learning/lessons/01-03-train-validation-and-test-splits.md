---
id: "01.03"
course: "machine-learning"
module: "01"
slug: "train-validation-and-test-splits"
title: "Train, validation and test splits"
keyIdeas: ["Use three roles: **train** to fit parameters, **validation** to choose hyperparameters and models, **test** touched once to report an unbiased number. Choosing the best of many options on validation makes its score optimistic, which is why the test set must stay out of every decision.", "Size evaluation sets by absolute count, not percentage: the standard error of an accuracy is $\\sqrt{p(1-p)/m}$, so 1,000 examples at 90% accuracy give about $\\pm 2$ points (95%), and a half-point gain is noise.", "k-fold cross-validation reuses every point for validation once; it gives a mean and a spread, at k times the cost. Treat configurations within about one fold standard deviation as tied.", "**Leakage** is any path by which information about the evaluation data reaches training: preprocessing fitted on all rows, duplicates across splits, features from the future, target proxies. Put every transform inside a `Pipeline`.", "The split must mimic deployment: split by time if you predict the future, by group (user, patient, site) if you predict for unseen groups, and by a stable hash of the id if splits must survive retraining."]
question: "You build a model that predicts machine failure from sensor readings sampled every minute on 40 machines, and random 5-fold CV gives 97% accuracy. Explain why that number is suspect and design a split that answers the real question \"will this work on next month's data from the same fleet\" and, separately, \"will it work on a new machine\"."
answer: "Minute-level readings are strongly autocorrelated, so a random split puts almost identical neighbouring minutes in train and validation; the model can interpolate between them instead of learning failure signals. Both machine identity and time leak, and if features such as rolling averages were computed over the whole series, information from the future leaks too. For **next month on the same fleet**, split by time: train on months 1..t, validate on month t+1, and roll forward (time-series CV), with a gap between the windows at least as long as the longest feature lookback so no window spans the cut; keep the latest month or two as a final test set that is scored once. For **a new machine**, group by machine id: leave-one-machine-out or GroupKFold over the 40 machines, so each validation machine is unseen; ideally also keep the time ordering inside, training on earlier periods of the training machines. Expect both numbers to be well below 97%; the gap between them tells you how much the model relies on machine-specific behaviour. Fit any scaling, imputation or feature selection inside each training fold via a Pipeline. Also check the evaluation size: failures are rare, so stratify or at least verify each fold contains failures, and judge the result with a metric suited to rare events (recall at a fixed false-alarm rate) and its spread across folds, since with 40 machines a few machines can swing the score by several points."
sources: [{"title": "scikit-learn: Cross-validation", "url": "https://scikit-learn.org/stable/modules/cross_validation.html", "note": "k-fold, stratified, group and time-series iterators, and why the test set stays separate."}, {"title": "scikit-learn: Common pitfalls and recommended practices", "url": "https://scikit-learn.org/stable/common_pitfalls.html", "note": "Concrete preprocessing-leakage examples and the Pipeline fix."}, {"title": "Leakage and the reproducibility crisis in ML-based science (Kapoor and Narayanan)", "url": "https://arxiv.org/abs/2207.07048", "note": "A taxonomy of eight leakage types with real case studies."}, {"title": "scikit-learn: TimeSeriesSplit", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html", "note": "Forward-chaining splits, including the gap parameter."}]
practice: ["hashed-dataset-split", "k-fold-cross-validation", "normalize-columns", "point-in-time-feature-join"]
tags: ["cross-validation", "data leakage"]
tools: ["scikit-learn"]
---

## What you will learn

- Why one held-out set is not enough, and the separate jobs of the **training**, **validation** and **test** sets.
- How big each split should be, using the standard error of an accuracy estimate rather than a fixed percentage.
- How k-fold cross-validation works, how to read its mean and spread, and which variant (stratified, grouped, time-ordered) fits your data.
- What **leakage** is, how it usually happens, and how a scikit-learn `Pipeline` stops the most common kind.
- How to design a split that copies deployment.

*Before you start:* you should know what a loss and an empirical risk are (lesson 01.01) and why training error is too optimistic, which is the overfitting story of lesson 01.02. Basic averages and square roots are enough for the maths here.

Lesson 01.02 showed that training error keeps falling as a model gets more flexible, while error on new data eventually rises. To find the sweet spot you must measure error on data the model has not fitted, and do it honestly. It sounds like bookkeeping, but most models that scored 95% in the notebook and 70% in production failed right here.

## The idea in one picture

You have 105 noisy measurements of a curve and want to fit a polynomial. The open question is the **degree**: 1 is a straight line, 10 is very wiggly. The degree is a **hyperparameter**: a setting you choose *before* fitting, as opposed to the polynomial's coefficients (the **parameters**), which the fitting algorithm sets.

You cut the rows into three piles: 25 for training, 40 for validation, 40 for testing. For each degree you fit on the 25 training rows, then measure mean squared error on the training rows and on the 40 validation rows:

| Degree | 1 | 2 | 3 | 4 | 5 | 7 | 10
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training error | 0.205 | 0.187 | 0.072 | 0.071 | 0.069 | 0.055 | 0.044
| Validation error | 0.249 | 0.278 | 0.086 | 0.084 | 0.091 | 0.161 | 0.299

~~~dg-fig 01.03-select.svg
Training error only ever falls as the degree grows, so it cannot pick the degree. Validation error is U-shaped and picks degree 4. The diamond is the test error of that one chosen model, measured once at the end.
~~~

Read the table like this:

- Training error keeps dropping all the way to degree 10. If you chose by training error, you would always pick the most flexible model.
- Validation error drops, bottoms out at degree 4 (0.084), then climbs. This is the U-curve from lesson 01.02, now measured instead of imagined.
- You pick degree 4 and evaluate it once on the 40 test rows. Its test error is 0.120, noticeably worse than the 0.084 that won the contest.

That last point is the whole lesson in miniature. The validation error of the winner is flattering, because you chose it *for* being low. The test set, which played no part in the choice, gives the honest number.

## Step 1: give each split exactly one job

Every number you look at and then act on becomes part of the fitting process. The optimiser fits the parameters to the training set; *you* fit the hyperparameters to the validation set. So you need three roles:

- **Training set**: the optimiser sees it and sets the parameters.
- **Validation set** (or dev set): you use it to choose hyperparameters, features, model families and when to stop. Because you select on it, it also becomes optimistic, just more slowly.
- **Test set**: looked at once, at the end, to report a number you will stand behind. Tune anything after seeing it and it has silently become a second validation set.

~~~dg-flow Each split has one job. Information flows only left to right; the test set feeds nothing back.
Training set -(fit parameters)-> Model -(score)-> Validation set -(pick hyperparameters)-> *Final model -(score once)-> Test set
~~~

Why does choosing make validation optimistic? Each validation score is true performance plus luck from which rows landed in the set. Keep the best of many options and you partly select for luck.

~~~dg-fig 01.03-optimism.svg
A simulation where every configuration is truly 80% accurate. Pick the best of k on the same 200 validation rows and its validation score climbs to 87% at k = 100, while its accuracy on fresh test rows stays at 80%.
~~~

No configuration is truly better, yet the best of 100 shows a 7-point apparent gain. Real searches are milder, but the bias always points the same way, which is why the test set stays out of every decision.

## Step 2: size the splits by the error bar you can tolerate

Textbooks often quote 70/15/15 or 80/10/10. Those are fine defaults for a few thousand rows, but the right way to think is in *absolute* numbers of evaluation examples, because that is what controls how noisy a score is.

Treat each test example as a coin flip that comes up "correct" with probability $p$, the model's true accuracy. The measured accuracy on $m$ examples, $\hat p$, then has a **standard error** of

$$\text{SE}(\hat p) = \sqrt{\frac{p(1-p)}{m}}.$$

Here $m$ is the number of evaluation examples and $p$ the true accuracy (you plug in the measured one). A 95% interval is roughly $\hat p \pm 1.96\,\text{SE}$. Number check: at $m = 100$ and $p = 0.9$, $\text{SE} = \sqrt{0.09/100} = 0.03$, so the 95% interval is about $\pm 5.9$ points. A tiny test set cannot tell 85% from 95%.

~~~dg-fig 01.03-stderr.svg
The 95% margin shrinks with the square root of the test-set size: about 6 points at 100 examples, 2 at 1,000 and 0.6 at 10,000 for a 90%-accurate model. Accuracies nearer 50% have wider margins.
~~~

Quadrupling the test set only halves the error bar. With a million rows, a 1% test set (10,000 examples) already gives about $\pm 0.6$ points. With 500 rows in total, no fixed split is satisfying, which is where cross-validation comes in.

<div class="ex">
<div class="lab">Worked example</div>

**is a half-point improvement real?**

Your current model scores 90.0% on a 1,000-example test set. A new version scores 90.5%. Is it better?

Standard error: $p(1-p) = 0.9 \times 0.1 = 0.09$. Divide by $m$: $0.09 / 1000 = 0.00009$. Square root: $\sqrt{0.00009} \approx 0.0095$, so one standard error is about 0.95 points.

95% margin: $1.96 \times 0.95 \approx 1.9$ points. The new score is $90.5 \pm 1.9$, the interval from 88.6 to 92.4, which comfortably contains 90.0.

Verdict: a 0.5-point gain on 1,000 examples is indistinguishable from noise. To shrink the margin to 0.5 points you would need $m = 0.09 \times (1.96 / 0.005)^2 \approx 13{,}800$ examples.

</div>

## Step 3: cross-validation when data is scarce

With little data, a single validation split wastes rows and gives a noisy score. **k-fold cross-validation** fixes both.

1. Shuffle the development data (everything except the test set) and cut it into $k$ equal **folds**.
2. For each fold $j = 1, \dots, k$: train on the other $k-1$ folds, score on fold $j$. Call that score $s_j$.
3. Report the mean $\bar s = \frac{1}{k}\sum_{j=1}^{k} s_j$ and the standard deviation of the $s_j$.

Every row is validated exactly once, at the price of $k$ trainings. Common choices are $k = 5$ or $10$. The mean is your estimate; the spread tells you how much to trust differences.

~~~dg-fig 01.03-folds.svg
Three ways to cut 24 rows into rounds. Shuffled k-fold mixes rows freely. Group k-fold keeps every row of a patient on the same side. A time-series split always trains on the past and validates on the block that comes next.
~~~

The figure shows the three variants you will use most:

- **Stratified k-fold** (like the top panel) keeps class proportions equal in every fold; essential when a class is rare.
- **Group k-fold** keeps all rows of a group (user, patient, hospital) in one fold, so validation groups are always unseen.
- **Time-series split** (forward chaining) trains on everything up to a point, validates on the next block, then grows the window.

<div class="ex">
<div class="lab">Worked example</div>

**reading five fold scores by hand**

Configuration A gets fold accuracies $(0.82, 0.79, 0.85, 0.80, 0.84)$. Sum $= 4.10$, so the mean is $4.10/5 = 0.82$.

Deviations from the mean: $(0, -0.03, 0.03, -0.02, 0.02)$. Squares: $(0, 0.0009, 0.0009, 0.0004, 0.0004)$, which sum to $0.0026$. Divide by 5: $0.00052$. Square root: standard deviation $\approx 0.023$, about 2.3 points. (Dividing by $k-1 = 4$ instead gives 0.025; `np.std` uses $k$ by default.)

Configuration B gets $(0.83, 0.81, 0.86, 0.80, 0.85)$: sum $4.15$, mean $0.83$.

The gap of 1 point is less than one fold standard deviation, so by the rule of thumb you treat A and B as tied and prefer the simpler or cheaper one.

</div>

After CV picks the configuration, retrain it on *all* the development data before testing.

## Step 4: make the split copy deployment

A random split answers one specific question: "how well does the model do on new rows drawn from the same pile?" Often that is not the question you care about. Ask instead: *at prediction time, what will be new?* Then make the validation and test data new in exactly that way.

| Deployment | What is new | Split
| --- | --- | --- |
| Same users, new random events | individual rows | random or stratified
| New users or patients | the group | grouped by id
| Predicting next week | time | train on past, validate on a later block
| New site or hospital | the environment | leave-one-site-out
| Stable assignment across retrains | nothing should move | hash the id into buckets

If one patient contributes twenty X-rays, a random split lets the model score well by recognising the patient instead of the disease. If you forecast demand, a random split lets the model train on Wednesday and validate on the Tuesday before, which is interpolation, not forecasting.

A **hash-based split** gives each id a permanent home: take a stable hash of the id (MD5, not Python's built-in `hash`, which changes between runs) modulo 100, and send buckets 0-79 to train, 80-89 to validation, 90-99 to test. Old customers stay put across retrains; random re-shuffling would slowly move test customers into training.

## Step 5: keep leakage out

**Leakage** is any path by which information about the evaluation data reaches training. The scores look wonderful and production does not. Kapoor and Narayanan (2022) surveyed leakage across 17 scientific fields and found it affected hundreds of published papers. The usual suspects:

1. **Preprocessing fitted on all data.** Standardising, imputing, selecting features or fitting PCA before splitting lets held-out statistics shape the training inputs. Fit every transform on the training fold only.
2. **Duplicates and near-duplicates** across splits: the same image resized, the same customer entered twice.
3. **Group leakage**: several rows per patient, split at random, so the model learns to recognise the entity rather than the outcome.
4. **Temporal leakage**: features computed with information from after the prediction time (a "total purchases" column that includes the purchase being predicted).
5. **Target proxies**: a feature that is a consequence of the label, such as "refund issued" when predicting fraud.

~~~dg-fig 01.03-leak.svg
Two leaks, measured. Left: 100 rows of pure noise with 5,000 features; picking the 20 "best" features on all rows before CV reports 89% accuracy, while doing the selection inside each fold gives the true 50%. Right: 30 patients with 20 rows each; a random split reports 99%, a grouped split 55%.
~~~

In the left panel, some of 5,000 random features correlate with 100 random labels by pure chance. Pick them using every row and the validation rows helped choose them, so CV confirms a pattern that does not exist. The fix is to make selection part of the model, redone inside each training fold, which is what a scikit-learn `Pipeline` does.

<div class="ex">
<div class="lab">Worked example</div>

**a churn model that looked 12 points too good**

Data: 2 years of monthly snapshots per customer; the label is "churns next month". A random 80/20 split gave AUC 0.91.

Problem 1: the same customer appears in both splits, with nearly identical features month to month. Problem 2: a feature "support tickets in the last 90 days" was computed at extraction time, so it included tickets filed after the snapshot.

Fix: cut by time (train on year 1, validate on the first quarter of year 2), recompute features as of each snapshot date, and keep each customer on one side. AUC fell to 0.79, which matched what the model later did in production.

</div>

## Splitting in code

The full workflow with scikit-learn: lock away a test set, choose a hyperparameter with a validation set or with cross-validation, and touch the test set once.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                           weights=[0.8], random_state=0)        # 20% positives

# 1. Lock away 15% as the test set, then carve 300 validation rows
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X_dev, y_dev, test_size=300, stratify=y_dev, random_state=0)
print(len(X_tr), len(X_val), len(X_test))          # 1400 300 300

# 2. Choose C on the validation set; the scaler is fitted on X_tr only
for C in [0.001, 0.01, 0.1, 1, 10]:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    print(C, round(pipe.fit(X_tr, y_tr).score(X_val, y_val), 3))
# 0.001 0.813 | 0.01 0.903 | 0.1 0.92 | 1 0.917 | 10 0.917

# 3. Same choice by 5-fold CV on all 1700 development rows
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
search = GridSearchCV(pipe, {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10]}, cv=cv)
search.fit(X_dev, y_dev)                 # then refits the best C on all 1700 rows
print(search.best_params_, round(search.best_score_, 3))   # {'...C': 10} 0.919
# mean CV accuracy per C: 0.806 0.896 0.915 0.919 0.919, each with std about 0.005

# 4. Only now, once
print(round(search.score(X_test, y_test), 3))              # 0.893
```

Notice three things. Validation preferred $C = 0.1$ and CV preferred $C = 10$, but all $C \ge 0.1$ are within half a point: a tie. The test score (0.893) is below validation (0.92); with 300 test rows the 95% margin is about $\pm 3.4$ points, so the gap is selection optimism plus noise. And because the scaler is inside the pipeline, `GridSearchCV` refits it on each training fold.

For grouped data, pass the ids: `cross_val_score(pipe, X, y, groups=patient_id, cv=GroupKFold(n_splits=5))`. For time-ordered data, use `TimeSeriesSplit(n_splits=5, gap=g)`, where `gap` drops `g` rows between the training window and the validation block.

## Checking the result

Before you trust an evaluation number, run through these checks:

- **Compare train, validation and test.** Train far above validation means overfitting (lesson 01.02). Validation far above test means you tuned too hard, or the two sets differ.
- **Look at the fold spread**, not just the mean. A 5-point standard deviation makes 1-2 point comparisons meaningless.
- **Suspiciously good is a bug until proven otherwise.** Inspect the top features (a target proxy usually sits first), search for duplicates across splits, and check each feature exists at prediction time.
- **Check the splits resemble each other.** Train a classifier to tell training rows from test rows (adversarial validation). If it beats chance easily, the test set comes from a different distribution.
- **Compare against a baseline** scored on the same split (lesson 01.05). A 0.93 AUC means little until you know a one-feature rule gets 0.91.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Tuning on the test set | test score drifts up over weeks of experiments; production lower | keep a final holdout nobody looks at; log every test evaluation
| Scaler or feature selection fitted before splitting | CV score far above chance on data that should be hard | put every transform in a `Pipeline`
| Random split with repeated entities | huge gap between CV and production; grouped CV much lower | `GroupKFold`, split by id
| Random split on time-ordered data | great backtests, poor live forecasts | `TimeSeriesSplit`, with a gap if features look back in time
| Test set too small | rankings of models flip when you rerun with another seed | size by standard error; use CV and report spread
| Rare class missing from a fold | metric undefined or wildly variable per fold | stratify
| Deciding on differences below one fold standard deviation | the "better" model is not better next month | treat as ties; prefer the simpler model

## Going deeper

**How optimistic is the winner?** If $k$ configurations are equally good and their validation scores have independent noise with standard deviation $\sigma$, the best of them exceeds the truth by about $\sigma$ times the expected maximum of $k$ standard normals. That expected maximum grows like $\sqrt{2 \ln k}$; for $k = 100$ it is about 2.5. In the simulation, $\sigma = \sqrt{0.8 \times 0.2 / 200} \approx 0.028$, so the expected optimism is $2.5 \times 2.8 \approx 7$ points, matching the figure. The bias grows only logarithmically in $k$ but linearly in $\sigma$, which is why larger validation sets protect you more than trying fewer things.

**Choosing k.** Each fold model trains on $(k-1)/k$ of the data, so small $k$ is slightly pessimistic. Leave-one-out ($k = n$) removes most of that bias, but its fold models share nearly all their data, so the estimate can be high-variance as well as costing $n$ fits. $k = 5$ or $10$ is the usual compromise; repeating with different shuffles reduces dependence on one partition.

**The fold standard deviation is a rough guide, not a confidence interval.** Fold scores are not independent: any two training sets overlap in $(k-2)/k$ of the rows. Bengio and Grandvalet (2004) showed there is no unbiased estimator of the variance of k-fold CV from a single run. Use the fold spread for "is this difference obviously noise?", not for formal p-values.

**Paired comparisons.** Scoring two models on the same rows and testing their per-example differences (McNemar's test for two classifiers) cancels shared difficulty and detects smaller gaps than two separate error bars.

**Nested cross-validation.** The best CV score from a search carries the winner's optimism. Nested CV wraps the search in an outer loop: the inner loop tunes on the outer training folds, the outer held-out fold scores the result. The outer mean estimates the whole procedure "search then fit", useful when data is too scarce for a test set.

**Gaps for time series.** With a 30-day rolling feature, the last training rows and first validation rows share up to 29 days of inputs. Leave a gap at least as long as the longest lookback (`TimeSeriesSplit(gap=...)`).

## Key terms

| Term | Meaning
| --- | --- |
| Parameter | A value the training algorithm sets, such as a weight or a tree split
| Hyperparameter | A setting chosen before training, such as polynomial degree or regularisation strength
| Training set | Rows the optimiser fits the parameters on
| Validation set | Rows used to choose hyperparameters, features and models; becomes optimistic through selection
| Test set | Rows scored once at the end to report an unbiased estimate
| k-fold cross-validation | Train k times, each time holding out a different fold, and average the k scores
| Standard error | Typical size of the random error in a measured score, $\sqrt{p(1-p)/m}$ for accuracy
| Leakage | Any path by which evaluation data influences training
| Group / time split | A split that keeps whole entities, or whole time periods, on one side

<div class="take">
<div class="lab">Key takeaway</div>

give train, validation and test separate jobs, size evaluation sets by the error bar you need, fit every transform inside the training fold, and design the split so that what is new at evaluation is exactly what will be new in deployment.

</div>
