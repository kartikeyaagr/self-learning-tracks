---
id: "01.03"
course: "machine-learning"
module: "01"
slug: "train-validation-and-test-splits"
title: "Train, validation and test splits"
keyIdeas: ["Use three roles: **train** to fit parameters, **validation** to choose hyperparameters and models, **test** touched once to report an unbiased number.", "k-fold cross-validation reuses every point for validation once; it gives a mean and a spread, at k times the cost.", "**Leakage** is any path by which information about the evaluation data reaches training: preprocessing fitted on all rows, duplicates across splits, features from the future.", "The split must mimic deployment: split by time if you predict the future, by group (user, patient, site) if you predict for unseen groups."]
question: "You build a model that predicts machine failure from sensor readings sampled every minute on 40 machines, and random 5-fold CV gives 97% accuracy. Explain why that number is suspect and design a split that answers the real question \"will this work on next month's data from the same fleet\" and, separately, \"will it work on a new machine\"."
answer: "Minute-level readings are strongly autocorrelated, so a random split puts almost identical neighbouring minutes in train and validation; the model can interpolate between them instead of learning failure signals. Both machine identity and time leak. For **next month on the same fleet**, split by time: train on months 1..t, validate on month t+1, and roll forward (time-series CV), with a gap between the windows if features use lookback windows so no window spans the cut. For **a new machine**, group by machine id: leave-one-machine-out or GroupKFold over the 40 machines, so each validation machine is unseen; ideally also keep the time ordering inside. Expect both numbers to be well below 97%; the gap between them tells you how much the model relies on machine-specific behaviour. Fit any scaling or feature selection inside each training fold."
sources: [{"title": "scikit-learn: Cross-validation", "url": "https://scikit-learn.org/stable/modules/cross_validation.html", "note": "k-fold, stratified, group and time-series iterators, and why the test set stays separate."}, {"title": "scikit-learn: Common pitfalls and recommended practices", "url": "https://scikit-learn.org/stable/common_pitfalls.html", "note": "Concrete preprocessing-leakage examples and the Pipeline fix."}, {"title": "Leakage and the reproducibility crisis in ML-based science (Kapoor and Narayanan)", "url": "https://arxiv.org/abs/2207.07048", "note": "A taxonomy of eight leakage types with real case studies."}, {"title": "scikit-learn: TimeSeriesSplit", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html", "note": "Forward-chaining splits, including the gap parameter."}]
practice: ["hashed-dataset-split", "k-fold-cross-validation", "normalize-columns", "point-in-time-feature-join"]
tags: ["cross-validation", "data leakage"]
tools: ["scikit-learn"]
---

## Three roles, three splits

The previous lesson argued that training error is optimistic. The fix is to measure on data the model never fitted. But every *decision* you make while looking at a number is also a form of fitting. So you need three roles:

- **Training set**: the optimiser sees it and sets the parameters.
- **Validation set**: you look at it to choose hyperparameters, features, model families and when to stop. Because you select on it, it also becomes optimistic, just more slowly.
- **Test set**: looked at once, at the end, to report a number you will stand behind. If you tune after seeing it, it has become a second validation set.

A common default is 70/15/15 or 80/10/10. With millions of rows, a 1% test set (10,000 examples) is plenty; what matters is the *absolute* number of evaluation examples, because the standard error of an accuracy estimate is about $\sqrt{p(1-p)/m}$. At $m = 1000$ and $p = 0.9$, that is about ±1 percentage point, so a 0.5-point "improvement" is noise.

~~~dg-flow Each split has one job. Information flows only left to right; the test set feeds nothing back.
Training set -(fit parameters)-> Model -(score)-> Validation set -(pick hyperparameters)-> *Final model -(score once)-> Test set
~~~

## Cross-validation

When data is scarce, a single validation split wastes rows and gives a noisy estimate. **k-fold cross-validation** partitions the training data into $k$ folds, trains $k$ times on $k-1$ folds and validates on the one held out, then averages. Every row is validated exactly once. With $k = 5$ you get five scores: report the mean and the standard deviation; if two configurations differ by less than about one standard deviation, treat them as tied.

Variants worth knowing:

- **Stratified k-fold** keeps class proportions equal in every fold, essential when a class is rare.
- **Group k-fold** keeps all rows of a group (user, patient, hospital) in the same fold.
- **Time-series split** (forward chaining) trains on the past and validates on the next block, growing the window each time.
- **Nested CV**: an inner loop picks hyperparameters, an outer loop estimates the error of the whole procedure. Needed when you want an honest number without a separate test set.

## Leakage: the silent failure

Leakage is when the evaluation data influences training through a side door. The scores look great and production does not. Kapoor and Narayanan (2022) surveyed leakage across 17 scientific fields and found it affected hundreds of published papers. The usual suspects:

1. **Preprocessing fitted on all data.** Standardising, imputing, selecting features or fitting PCA before splitting lets test statistics shape the training inputs. Fit every transform on the training fold only, then apply it.
2. **Duplicates and near-duplicates** across splits: the same image resized, the same customer twice, the same sentence in two documents.
3. **Group leakage**: several rows per patient, split at random, so the model learns to recognise patients rather than disease.
4. **Temporal leakage**: features computed with information from after the prediction time (a "total purchases" column that includes the purchase being predicted), or a random split on data where the future resembles the recent past.
5. **Target proxies**: a feature that is a consequence of the label, like "refund issued" when predicting fraud.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
# The scaler is refit inside each fold, and no patient appears on both sides.
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(pipe, X, y, groups=patient_id, cv=GroupKFold(n_splits=5))
```

## Match the split to deployment

The question to ask is: *at prediction time, what will be new?* Then make the validation data new in exactly that way.

| Deployment | What is new | Split |
| --- | --- | --- |
| Same users, new random events | individual rows | random or stratified |
| New users or patients | the group | grouped by id |
| Predicting next week | time | train on past, validate on a later block |
| New site or hospital | the environment | leave-one-site-out |
| Stable assignment across retrains | nothing should move | hash the id into buckets |

A **hash-based split** (bucket = hash(id) mod 100) gives each id the same split forever, even as new data arrives. That stops rows drifting from test into train between retrains, which random re-splitting would do.

<div class="ex">
<div class="lab">Worked example</div>

**a churn model that looked 12 points too good**

Data: 2 years of monthly snapshots per customer; label = churns next month. A random 80/20 split gave AUC 0.91. Problem 1: the same customer appears in both splits, with nearly identical features month to month. Problem 2: a feature "support tickets in the last 90 days" was computed at extraction time, so it included tickets filed after the snapshot.

Fix: cut by time (train on year 1, validate on the first quarter of year 2), recompute features as of each snapshot date, and keep each customer on one side. AUC fell to 0.79, which matched what the model later did in production.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

give train, validation and test separate jobs, fit every transform inside the training fold, and design the split so that what is new at evaluation is exactly what will be new in deployment.

</div>
