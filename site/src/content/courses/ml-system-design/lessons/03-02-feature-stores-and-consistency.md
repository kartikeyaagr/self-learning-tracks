---
id: "03.02"
course: "ml-system-design"
module: "03"
slug: "feature-stores-and-consistency"
title: "Feature stores and consistency"
keyIdeas: ["A **feature store** gives each feature one definition and two materialisations: an offline store with history for training and an online store with the latest value for serving.", "Its main job is consistency. It removes the risk of training and serving computing the same feature differently, and supports point-in-time joins.", "Batch features are computed by scheduled jobs; streaming features by a stream processor; request-time features in the serving path. All three must be defined once.", "You do not always need a feature store. Small systems can log served features and share one library of feature code, which solves most of the same problems."]
question: "Two teams compute \"number of orders in the last 7 days\" for a user. The training pipeline uses a nightly SQL job; the serving path increments a counter in a cache on each order. The model degrades for active users. Explain why and design a fix."
answer: "The two computations differ in at least three ways. The nightly SQL value is up to 24 hours stale, so at training time the model sees a lagged count, while serving sees a live one; for active users, who order more often, the gap is largest. The cache counter may never decrement orders older than 7 days, or may reset on cache eviction, so it drifts from the true windowed count. Cancelled orders may be excluded in SQL but counted by the increment. Fix: define the feature once, for example in a feature store, as a 7-day window over the order event stream; compute it with a streaming job that writes both to the online store (latest value) and to the offline store (timestamped history). Train with point-in-time joins against the offline history, and log the served value with each prediction so you can measure any remaining mismatch."
sources: [{"title": "Feast documentation: concepts", "url": "https://docs.feast.dev/", "note": "An open-source feature store's offline and online stores and point-in-time joins."}, {"title": "Google Cloud, MLOps: continuous delivery and automation pipelines", "url": "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning", "note": "Feature stores within an ML pipeline."}, {"title": "Wikipedia, Feature store", "url": "https://en.wikipedia.org/wiki/Feature_store", "note": "Short overview and common implementations."}]
practice: ["point-in-time-feature-join", "training-serving-skew", "record-schema-validation"]
tags: ["feature store", "training/serving skew"]
---

## The consistency problem

Every model depends on two programs agreeing about features: the one that builds training data and the one that serves predictions (01.03). In small projects one engineer writes both, and the risk is manageable. In a company with dozens of models and hundreds of features, teams reimplement the same feature in different languages, at different freshness, with different edge cases. A feature store is infrastructure to stop that.

## What a feature store does

- **Registry**: one definition per feature, with owner, entity, type, freshness and description.
- **Offline store**: full history of feature values with timestamps, usually in a data warehouse or lake, for building training sets.
- **Online store**: the latest value per entity in a low-latency key-value store, for serving.
- **Point-in-time joins**: given a list of (entity, timestamp) pairs, return each feature as of that timestamp.
- **Materialisation jobs**: compute features from source data and write them to both stores.

~~~dg-flow One definition, two materialisations. Training reads history; serving reads the latest value.
Source events -(feature definition)-> Batch or stream job -(timestamped values)-> Offline store -(point-in-time join)-> Training data
Batch or stream job -(latest value)-> *Online store -(get by key, ~ms)-> Serving
~~~

## Three kinds of features

| Kind | Computed by | Freshness | Example
| --- | --- | --- | --- |
| Batch | scheduled job, hourly or daily | hours | user's purchases last 30 days
| Streaming | stream processor | seconds to minutes | card's transactions in last 10 minutes
| Request-time | the serving code | instant | query length, time of day, distance to restaurant

Request-time features are not stored at all; they come from the request. They still need one shared implementation, typically a library used by both the training pipeline (applied to logged requests) and the server.

## Point-in-time correctness

Training data is built by asking the offline store: "for user 42 at 2024-03-01 14:05, what was `orders_7d`?" The store returns the last value written before that timestamp. This prevents future information leaking into training. It needs the offline store to keep history, not just the current value, and to record when each value became available, not only the event time it refers to. Late-arriving data is a common subtle bug: an event from 13:50 that arrived at 14:30 must not be visible at 14:05.

<div class="ex">
<div class="lab">Worked example</div>

**freshness skew without a code bug**

A feed-ranking team used a daily batch feature for "author's posts in the last 24 hours". Online it was read from the online store, which was refreshed once a day at 02:00. Offline, training used point-in-time joins, but the job that backfilled history had computed values at event time, not at availability time. Training saw a feature that was always current; serving saw one up to 24 hours old. Recording availability timestamps and joining on them removed the gap.

</div>

## Do you need one?

A feature store pays off with many models sharing features, streaming features, or several teams. For a single model, two cheaper habits get most of the benefit:

- put feature code in one library imported by both training and serving;
- log the features used at serving time and build the next training set from those logs.

In an interview, name the feature store as the component that guarantees consistency, then say what it stores, how fresh each feature is and what happens on a cache miss.

## Failure handling

The online store can be slow or missing a key. Decide per feature: use a default (and make sure training saw the same default), fall back to a coarser batch feature, or skip the model and use a fallback ranking. Monitor the miss rate as a first-class metric; a sudden rise usually means a failed materialisation job.

## Backfills and new features

When you add a feature, you need its history to train on. A backfill job computes historical values from source data, and it must use the same definition and the same availability rules as the live job, or the new feature will be skewed from day one. Some teams instead start logging the new feature at serving time and wait a few weeks to collect training data, which is slower but guaranteed consistent. Either way, compare the backfilled values with live values for a recent overlap period before trusting them.

<div class="take">
<div class="lab">Key takeaway</div>

a feature store gives each feature one definition with an offline history for point-in-time training joins and an online store for serving; whether or not you use one, share feature code, record availability times and log served values.

</div>
