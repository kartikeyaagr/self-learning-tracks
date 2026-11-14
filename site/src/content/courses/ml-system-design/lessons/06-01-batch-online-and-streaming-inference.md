---
id: "06.01"
course: "ml-system-design"
module: "06"
slug: "batch-online-and-streaming-inference"
title: "Batch, online and streaming inference"
keyIdeas: ["**Batch** inference precomputes predictions for known entities on a schedule; **online** inference computes per request; **streaming** inference reacts to events as they arrive.", "Choose by freshness needs, whether inputs are only known at request time, the number of entities, the latency budget and cost.", "Hybrid designs are the norm: batch-computed embeddings or candidate lists, streaming-updated features, and a light online model that adds request context.", "Each mode has its own failure: batch goes stale or misses new entities, online adds latency and dependency risk, streaming adds operational complexity and ordering issues."]
question: "An e-commerce site wants \"customers who viewed this also viewed\" on every product page (20M products, 5,000 page views per second) and a \"recommended for you\" rail on the home page that reflects what the user browsed in the last few minutes. Choose batch, online or streaming for each and justify."
answer: "\"Also viewed\" depends only on the product, not the user, and co-view patterns change slowly, so compute it in batch: a nightly or hourly job over co-view sessions produces the top 20 related products per product, stored in a key-value store. Serving is a lookup in about 1 ms, cheap at 5,000 QPS, with a fallback to same-category bestsellers for new products. The home rail needs the last few minutes of browsing, so it needs fresh input: a streaming job updates each user's recent-items list and session embedding within seconds; at request time an online service retrieves candidates (from batch-computed related-item lists of recently viewed products plus the user's long-term embedding) and ranks a few hundred with a light model using the session features. Batch provides the heavy candidate generation, streaming the freshness, online the final context-aware ranking."
sources: [{"title": "Chip Huyen, Real-time machine learning: challenges and solutions (2022)", "url": "https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html", "note": "Online prediction and streaming features."}, {"title": "Google Cloud, MLOps: continuous delivery and automation pipelines", "url": "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning", "note": "Batch and online serving within a pipeline."}, {"title": "Apache Flink documentation: stateful stream processing", "url": "https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/stateful-stream-processing/", "note": "How streaming features keep windowed state."}]
practice: ["dynamic-request-batching", "training-serving-skew", "point-in-time-feature-join", "p50-p95-and-p99-inference-latency"]
tags: ["batch inference", "online inference", "streaming"]
---

## Three ways to produce a prediction

The Machine Learning course introduced batch and online inference (mlcore 09.01). Here we look at the choice as a design decision with streaming as a third option.

| Mode | When it runs | Latency at use | Freshness | Coverage | Typical uses
| --- | --- | --- | --- | --- | --- |
| Batch | on a schedule | lookup, ~1 ms | as old as last run | entities known at run time | churn scores, daily recommendations, forecasts
| Online | per request | model time, 5-100 ms | current request | any input | search ranking, fraud at authorisation, ETA at checkout
| Streaming | per event | seconds after the event | near real time | entities with events | updating features, alerting, trending detection

## How to choose

Ask these questions:

- **Is the input only known at request time?** A search query, a basket or a transaction amount forces online inference.
- **How fresh must the prediction be?** If yesterday's answer is fine, batch is simplest and cheapest.
- **How many entities, and how many are actually requested?** Scoring 100M users nightly is wasteful if 5M visit; online scoring follows actual traffic.
- **What is the latency budget?** Heavy models that take 200 ms can only run in batch for interactive surfaces.
- **What does it cost?** Batch uses cheap, schedulable compute; online needs capacity for peak traffic.

~~~dg-flow A common hybrid. Batch does the heavy work, streaming keeps features fresh, online adds the request.
Nightly batch -(embeddings, candidate lists)-> Key-value store
Event stream -(windowed counts, session state)-> Online feature store
Request -(lookup + fetch)-> *Online ranker -(top items)-> Response
~~~

## Streaming in practice

A stream processor (Flink, Kafka Streams, Spark Structured Streaming) consumes events and maintains state: counts over sliding windows, the last N items a user viewed, running averages. It writes results to an online store that serving reads. Streaming brings freshness without computing everything at request time, but it adds real complexity:

- **late and out-of-order events**: choose watermarks and decide how late data updates windows;
- **exactly-once semantics**: duplicate events double-count unless processing is idempotent;
- **backfills**: replaying history to build training data must use the same logic;
- **operations**: another always-on system to monitor and scale.

Use streaming when the freshness gain is measurable, for example fraud velocity features or session-aware recommendations, and batch otherwise.

## Failure modes

| Mode | What breaks | Mitigation
| --- | --- | --- |
| Batch | job fails or is late, so scores are stale; new entities have no score | alert on freshness; fallback for missing keys
| Online | slow features or model raise p99; dependency outages | timeouts, defaults, caching, fallbacks (06.04)
| Streaming | lag builds up; state is lost on restart | lag monitoring, checkpoints, replayable sources

<div class="ex">
<div class="lab">Worked example</div>

**a batch job that served yesterday's world**

A travel site precomputed hotel rankings per city nightly. During a large event, prices and availability changed hourly, and the top results were often sold out. Adding an online re-ranking step that fetched live availability and price for the top 100 batch candidates, and removed or demoted unavailable hotels, fixed the problem without making the whole ranking online.

</div>

## Request-time features

Even mostly batch systems usually add some request context online: time of day, device, current page, location. These are cheap because they come with the request. Make sure training data has them too, by logging them at serving time, so the model learns how to use them.

## Cost comparison

For 100M users, a nightly batch job that scores 500 items per user does 50 billion scores a day on cheap batch capacity. If only 10M users visit and each visit scores 500 items online, that is 5 billion scores, but on capacity sized for peak traffic. The cheaper choice depends on the active fraction and the price difference between batch and serving compute.

## Mixing modes over time

Many systems start batch and move parts online as the product matures. A sensible order is to keep heavy candidate generation in batch, add request-time context first (it is cheap), then add streaming features only where an experiment shows the freshness pays. Each step is easy to measure on its own, and the batch output remains a natural fallback if the fresher path fails.

<div class="take">
<div class="lab">Key takeaway</div>

choose batch when yesterday's answer is good enough, online when the input exists only at request time, and streaming when freshness matters within seconds; most production systems combine all three and plan for each mode's failure.

</div>
