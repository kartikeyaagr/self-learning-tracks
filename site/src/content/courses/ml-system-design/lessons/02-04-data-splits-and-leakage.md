---
id: "02.04"
course: "ml-system-design"
module: "02"
slug: "data-splits-and-leakage"
title: "Data splits and leakage"
keyIdeas: ["A split must mimic deployment: the model will predict the future for new requests, often for new users or items, so test on later data and, where relevant, on unseen entities.", "**Leakage** is any information in training features that would not be available at prediction time, or test examples that are too similar to training ones. It inflates offline metrics.", "Common leaks: features computed after the prediction time, target-derived features computed on the full dataset, duplicates across splits, and the same user in train and test.", "Assign splits **deterministically** (for example by hashing an entity ID) so they are stable across rebuilds and between teams."]
question: "A model predicts which support tickets will escalate. Engineers split tickets randomly 80/20 and get 0.95 AUC; in production it is 0.78. Features include the ticket text, the customer's total tickets, and the agent's average handling time. Name three likely leaks and a split that would expose them."
answer: "(1) Temporal leakage: \"customer's total tickets\" and \"agent's average handling time\" were probably computed over the whole dataset, including tickets after the one being scored, so they encode future escalations. (2) Entity leakage: the same customer appears in train and test with near-identical tickets, so the model memorises customers rather than learning signals that generalise. (3) Post-hoc text: if the ticket text field includes later replies or notes added after escalation (\"escalated to tier 2\"), it leaks the label. A better split trains on tickets before a cut-off date and tests on tickets after it, with all features computed as of each ticket's creation time; additionally hold out a set of customers entirely to measure performance on new customers. AUC should drop toward the production number."
sources: [{"title": "Wikipedia, Leakage (machine learning)", "url": "https://en.wikipedia.org/wiki/Leakage_(machine_learning)", "note": "Types of leakage with references, including Kaufman et al. (2011)."}, {"title": "Kapoor and Narayanan, Leakage and the Reproducibility Crisis in ML-based Science (2022)", "url": "https://arxiv.org/abs/2207.07048", "note": "How common leakage is across published work."}, {"title": "scikit-learn, Cross-validation of time series data", "url": "https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split", "note": "Time-ordered cross-validation."}]
practice: ["hashed-dataset-split", "point-in-time-feature-join", "k-fold-cross-validation", "training-serving-skew"]
tags: ["data splits", "leakage"]
---

## A split is a simulation of deployment

A test set answers one question: how well will the model do on the data it will see in production? So it must be built the way production data arrives. In most products that means three things: predictions are about the future, many inputs come from users or items the model has not seen, and the model is retrained periodically on data up to some date.

## Choosing a split

| Split | When it fits | What it measures
| --- | --- | --- |
| Random by row | i.i.d. data with no time or entity structure (rare in products) | interpolation
| Time-based | almost every product system | performance on future data
| Grouped by entity | users, items, patients, sessions repeat | generalisation to new entities
| Time + entity | new users arrive over time | the hardest, most honest case

For a recommender, a common choice is to train on interactions before day $T$ and test on interactions after it, reporting separately for existing and new users. The Machine Learning course's "Train, validation and test splits" lesson (mlcore 01.03) covers the basics; this lesson focuses on what goes wrong in real pipelines.

~~~dg-flow A time-based split with a gap. The gap stops features with a look-back window from straddling the boundary.
Train: days 1-60 -(gap: 7 days)-> Validation: days 68-75 -(gap)-> *Test: days 76-90
~~~

## The usual leaks

**Future information in features.** Aggregates such as "user's total purchases" computed over the whole dataset include purchases after the prediction time. Fix with point-in-time features (01.03).

**Target leakage.** A feature that is a consequence of the label: "account status = closed" for churn, "refund issued" for fraud. Ask of every feature: would we know this value at the moment we predict?

**Preprocessing on the full dataset.** Fitting a scaler, a vocabulary or target encoding on all data before splitting lets test statistics leak into training. Fit them inside the training fold only.

**Duplicates and near-duplicates.** The same image, product listing or document appears in train and test. Deduplicate before splitting, or split by a group key that keeps duplicates together.

**Entity overlap.** The same user in train and test lets the model memorise user-specific patterns, which is fine if those users will be scored in production and misleading if the product mostly serves new users.

<div class="ex">
<div class="lab">Worked example</div>

**a 0.99 AUC that wasn't real**

A model to detect fake product reviews hit 0.99 AUC. Investigation found reviews were split randomly while reviewers wrote dozens of near-identical reviews each. The model had learned to recognise individual reviewers. Splitting by reviewer brought AUC to 0.83, which is what the moderation team saw when the model met new accounts.

</div>

## Deterministic assignment

Assign each entity to a split by hashing its ID: `split = "test" if hash(user_id) % 100 < 10 else "train"`. This is stable when the data is rebuilt, consistent across teams and tables, and needs no stored list. Combine it with a time cut-off when both matter.

## Detecting leaks

- Look for features with suspiciously high importance or near-perfect univariate AUC.
- Compare offline and online metrics; a large gap is a leak until proven otherwise.
- Retrain with a stricter split (by time, by entity); if the metric drops a lot, the easy split was leaking.
- Inspect a few high-confidence correct predictions and ask why the model knew.

## Validation sets for tuning

Hyperparameters and thresholds must be chosen on a validation set that sits between train and test in time. Choosing them on the test set leaks test information into model selection and makes the reported number optimistic. If you tune heavily, keep a final holdout that is used once.

## Splits for retrained models

If production retrains weekly on a sliding window, evaluate the same way: a rolling backtest that trains on weeks 1 to 8 and tests on week 9, then trains on 2 to 9 and tests on 10, and so on. The spread of results across weeks shows how stable the model is, and a sudden bad week often points to a data problem rather than a modelling one.

<div class="take">
<div class="lab">Key takeaway</div>

build splits that simulate deployment, usually by time and often by entity, compute every feature as of the prediction time and fit preprocessing on training data only; a large offline-online gap means a leak until proven otherwise.

</div>
