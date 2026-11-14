---
id: "02.01"
course: "ml-system-design"
module: "02"
slug: "logging-training-examples"
title: "Logging training examples"
keyIdeas: ["The best training data for a deployed model is a log of what it saw and did: the request, the features as served, the prediction, what was shown, and later the outcome.", "Log at **prediction time** with a stable request ID, then join outcomes to that ID later. Do not reconstruct features afterwards if you can avoid it.", "Log what was **shown**, including position and the candidates that were not chosen. Without it you cannot correct for exposure bias or train rankers.", "Logs need a schema, a retention policy and privacy rules. A field that silently stops being logged corrupts every model trained afterwards."]
question: "Design the logging for a search ranking system so that next month's ranker can be trained from it. What fields do you log, at what point, and how do you attach clicks and purchases that happen later?"
answer: "At serving time, log one record per search with a unique request ID, timestamp, user ID (or a pseudonymous key), query text and normalised query, device and locale, the model and feature versions, and for each of the top N results shown: item ID, position, the feature vector as served and the model score. Also log a sample of candidates that were retrieved but not shown, for training retrieval and for counterfactual evaluation. Client events (impressions actually rendered, clicks, add-to-basket, purchases, dwell time) are logged separately with the same request ID and item ID. A daily job joins events to the serving log within a fixed attribution window (for example clicks within 30 minutes, purchases within 7 days) to produce labels. Validate the schema on every batch, monitor null rates per field, and keep raw logs immutable so examples can be rebuilt."
sources: [{"title": "Google, Rules of Machine Learning, Rule 29", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "\"The best way to make sure that you train like you serve is to save the set of features used at serving time.\""}, {"title": "Google Cloud, MLOps: continuous delivery and automation pipelines", "url": "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning", "note": "Where logged data enters the training pipeline."}, {"title": "Chip Huyen, Machine Learning Systems Design (free notes)", "url": "https://huyenchip.com/machine-learning-systems-design/toc.html", "note": "Data pipelines and logging for ML."}]
practice: ["training-serving-skew", "record-schema-validation", "point-in-time-feature-join", "hashed-dataset-split"]
tags: ["logging", "training data"]
---

## Your system is your data source

For a model already in production, the richest training data is its own history: every request it received, what it predicted, what the user saw and what happened next. This data matches the serving distribution by construction and keeps growing. Designing the logging well is often the most valuable thing you can do for the next version of the model.

~~~dg-flow Serving logs and outcome events are joined later by request ID to produce labelled examples.
Request -(request_id)-> Serving log: features + scores + shown items -(join on request_id, item_id)-> *Labelled examples -(train)-> Next model
Client events: impressions, clicks, purchases -(join)-> Labelled examples
~~~

## What to log

| Field | Why
| --- | --- |
| request ID, timestamp | join key and prediction time
| entity IDs (user, item, query) | grouping, splits, debugging
| features as served | removes training/serving skew
| model and feature versions | know which model made which prediction
| prediction and score | calibration checks, counterfactual evaluation
| what was shown, with position | exposure and position bias
| candidates not shown (sampled) | negatives for retrieval, off-policy evaluation
| outcomes, with timestamps | labels and attribution windows

Logging served features costs storage, but it is the single most effective defence against skew. The alternative, recomputing features later from raw tables, risks using values that changed after the prediction.

## When to log

Log the prediction-side record when the prediction is made, before anything else happens. Log outcomes as separate events with the same keys. Join them in a batch job with explicit **attribution windows**: how long after the impression a click or purchase still counts. The window is a design choice. Too short and you miss real conversions; too long and you attribute unrelated actions.

<div class="ex">
<div class="lab">Worked example</div>

**the missing impression log**

A news app logged clicks but not impressions. The team tried to train a click-through model from clicks alone, with random articles as negatives. The model learned which articles were popular, not which ones a user would click when shown. Adding an impression log, with position, let them train on real shown-but-not-clicked negatives and correct for position, and offline AUC became predictive of online gains.

</div>

## Sampling and volume

Logs grow fast: 200M impressions a day at 2 KB each is 400 GB a day. Common tactics:

- keep all positives and a random sample of negatives, storing the sampling rate so you can reweight;
- log full feature vectors for a sample of traffic and IDs only for the rest;
- compress and partition by date so old data can be dropped or archived.

Whatever you sample, record how. A model trained on downsampled negatives predicts inflated probabilities unless you correct for the rate (02.03).

## Schema and quality

Treat the log as an API with a versioned schema. Check every batch: required fields present, types correct, null rates within expected ranges, categorical values from a known set. Alert when a field's null rate jumps from 0.1% to 40%, which usually means an upstream change. The Machine Learning course's "Reproducible data" lesson (mlcore 09.02) covers snapshotting and versioning these tables.

## Privacy

Logs contain personal data. Decide what you need and drop the rest: hash or pseudonymise identifiers, avoid storing raw text where a derived feature is enough, set retention limits, and honour deletion requests across both raw logs and derived training sets. Privacy rules are requirements (01.04), and they sometimes rule out a feature entirely.

## Logging for models that do not exist yet

Log a little more than today's model needs. If the next model might use the time a user spent on the previous page, or which filters they applied, that has to be in the log now: you cannot add history later. Agree a small set of generic context fields with the product team early, and review the schema whenever a new surface launches.

## Logging in an interview

In a design interview, logging is often skipped, and mentioning it early is an easy way to stand out. Say what you log, where the join key comes from, which attribution window you use and how you sample. Then connect it to later parts of the design: the same log feeds training, offline evaluation, calibration checks and drift monitoring. One well-designed log serves all four.

<div class="take">
<div class="lab">Key takeaway</div>

log each prediction with a request ID, the features as served, the model version and what was shown, join outcomes later inside explicit attribution windows, and protect the log with a schema check, a sampling record and privacy rules.

</div>
