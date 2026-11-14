---
id: "09.01"
course: "machine-learning"
module: "09"
slug: "from-notebook-to-service"
title: "From notebook to service"
keyIdeas: ["A model in production is a small part of a larger system: data pipelines, feature computation, serving, logging and monitoring are most of the code and most of the failures.", "**Batch inference** precomputes predictions on a schedule and serves them by lookup; **online inference** computes per request. Choose by how fresh the inputs must be, how many entities need scores, and the latency budget.", "**Training/serving skew** is any difference between the features or data the model saw in training and what it sees live: two code paths computing the same feature, different freshness, or a changed input distribution.", "The strongest defence is to **log features at serving time** and train on those logs, so that training and serving share one computation by construction.", "Validate inputs at the boundary against a schema (types, ranges, null rates); a model fed garbage returns confident garbage without raising an error."]
question: "An e-commerce ranking model was trained on daily snapshots of item features (price, stock, rating). In production it reads the same features from a live cache. Offline AUC is 0.81, but online click-through is below the old model's. Name two specific skew mechanisms that could explain this and how you would confirm each."
answer: "First, freshness skew: in training, features came from end-of-day snapshots, so a label (a click during the day) is paired with a feature value that may reflect events after the click, such as a price drop that happened later or a rating updated by that very purchase. The model learned relationships that the live features do not have. Confirm by rebuilding the training set with point-in-time feature values and checking whether offline AUC drops towards the online result. Second, code or default skew: the live cache may return nulls or defaults (stock = 0 on a cache miss) at a different rate or in a different encoding from the warehouse snapshot. Confirm by logging served features for a day, joining them to the offline values for the same request and item, and reporting mismatch and null rates per feature. Fix by logging served features and training on them."
sources: [{"title": "Sculley et al., Hidden Technical Debt in Machine Learning Systems (2015)", "url": "https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html", "note": "The catalogue of ML-specific debt: entanglement, hidden feedback loops, pipeline jungles."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Rules 29-37 on training/serving skew and logging features at serving time."}, {"title": "TensorFlow Data Validation guide", "url": "https://www.tensorflow.org/tfx/guide/tfdv", "note": "Schema inference, anomaly detection and skew/drift comparators between training and serving data."}]
practice: ["training-serving-skew", "record-schema-validation", "point-in-time-feature-join"]
tags: ["serving", "training/serving skew"]
tools: ["pandas"]
---

## The model is the small box

A notebook ends with a fitted object and a metric. A service has to keep producing good predictions for months while its inputs change underneath it. Sculley et al.'s "Hidden Technical Debt in Machine Learning Systems" made the point with a now-famous figure: the ML code is a small box surrounded by configuration, data collection, feature extraction, verification, serving infrastructure and monitoring. Most production incidents happen in those surrounding boxes, and most of them do not raise exceptions. The model keeps returning numbers; the numbers are just wrong.

This module covers the four habits that make an ML system trustworthy: serve it the same inputs it was trained on (this lesson), build training data reproducibly (09.02), watch it in production (09.03), and change it through controlled experiments (09.04). Inference Engineering's "Production Serving & Model Lifecycle" module covers packaging, health checks and rollouts for large-model servers; this module is about the data and statistics side that applies to every model.

## Batch or online

| Dimension | Batch (offline) inference | Online (real-time) inference
| --- | --- | --- |
| When it runs | on a schedule, e.g. nightly | per request
| Serving path | key-value lookup of a stored score | feature fetch, model forward pass
| Latency budget | none for the model; lookup is ~1 ms | typically 10-100 ms end to end
| Freshness | as old as the last run | uses the current request context
| Coverage | only entities known at run time | any entity, including brand-new ones
| Cost | proportional to all entities, every run | proportional to traffic
| Failure mode | stale or missing scores | latency spikes, feature-store outages

Batch suits churn scores for 5 million customers, weekly demand forecasts, or "recommended for you" rails that can be a day old. Online is needed when the input is only known at request time: fraud scoring a transaction, ranking search results for a query, re-ranking a feed after the last click. Many systems are hybrid: precompute expensive user and item embeddings in batch, then do a cheap online computation that combines them with request context. Near-real-time features (counts over the last 5 minutes, computed by a stream processor) sit between the two.

~~~dg-flow A typical hybrid: heavy features are batch-computed, light context is added per request, and what was served is logged for the next training run.
Request -(id + context)-> Feature fetch -(batch + streaming features)-> *Model -(score)-> Response -(log features + score)-> Training log
~~~

## Training/serving skew

Skew is the gap between what the model learned from and what it is given. It comes in three forms:

- **Code skew**: the training pipeline computes `days_since_signup` in SQL over a warehouse table; the service recomputes it in Java from a different timestamp field with a different timezone. Same name, different feature.
- **Data skew**: training used features as of the end of each day, but serving reads them live, or the reverse. A "number of orders" feature that lags by a day in training but is fresh at serving is a different variable.
- **Distribution skew**: the population changes because of the model itself (it blocks fraud, so fraudsters adapt) or because the product changes. That is drift, covered in 09.03.

Google's Rules of ML give the bluntest defence: record the exact feature values the service used for each prediction, and build the next training set from those records. A feature store gives one definition per feature with two materialisations (offline for training, online for serving), which removes most code skew. Logging served features removes the rest, at the cost of a delay before you have enough logged data to train on.

```python
# Skew report: compare features computed offline vs logged at serving time
import pandas as pd
j = offline.merge(served, on=["request_id"], suffixes=("_off", "_srv"))
for f in features:
    a, b = j[f + "_off"], j[f + "_srv"]
    mismatch = (~((a - b).abs() <= 1e-6 * b.abs().clip(lower=1))).mean()
    print(f, f"mismatch={mismatch:.2%}", f"null_off={a.isna().mean():.2%}", f"null_srv={b.isna().mean():.2%}")
```

<div class="ex">
<div class="lab">Worked example</div>

**a skew bug that tests did not catch**

A credit model uses `income_monthly`. Training data had it in pounds per month. The serving team took it from an API that returned pounds per year. Every applicant suddenly looked 12 times richer, approval rates jumped from 41% to 78% within an hour, and nothing errored. A range check on the input (monthly income above 50,000 is rare) and a dashboard of approval rate by hour would each have caught it in minutes.

</div>

## Contracts at the boundary

Treat model inputs like an API contract. For each feature, record type, allowed range or categories, expected null rate and units, and check every batch or request against it: reject, impute with a logged default, or alert. TensorFlow Data Validation and similar tools infer a schema from training data and flag anomalies in new data, such as new categories, missing columns or range violations. The check is cheap and turns silent failures into visible ones.

Also version the whole path: model artifact, feature definitions and preprocessing code ship together. A model trained with one tokenizer or bucket boundary and served with another is skew of the most frustrating kind.

<div class="take">
<div class="lab">Key takeaway</div>

pick batch or online inference by freshness, coverage and latency needs, then remove training/serving skew structurally: one feature definition, served features logged for training, and a schema check at every boundary.

</div>
