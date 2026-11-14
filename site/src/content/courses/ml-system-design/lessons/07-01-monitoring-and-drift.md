---
id: "07.01"
course: "ml-system-design"
module: "07"
slug: "monitoring-and-drift"
title: "Monitoring and drift"
keyIdeas: ["Monitor in layers that become available at different delays: service health (seconds), data quality (minutes), input and prediction drift (hours), and label-based quality (days to months).", "**Data drift** is a change in inputs; **concept drift** is a change in the relationship between inputs and labels. Only labels confirm concept drift, so watch proxies while labels mature.", "Drift statistics such as PSI or KS fire on harmless changes too. Alert on drift that is large, sustained and in features the model relies on, and confirm with business metrics.", "Every alert needs an owner and a runbook: what to check, how to roll back, when to retrain."]
question: "A loan-approval model's monthly default rate is only known after 90 days. Design monitoring that would catch a problem within a week of it starting, and say what each layer would catch."
answer: "Layer 1, service health (real time): request rate, error rate, p99 latency, fallback usage; catches outages and timeouts. Layer 2, data quality (per batch or hourly): schema checks, null rates, range checks and category sets for each input; catches upstream bugs such as an income field switching units. Layer 3, drift (daily): PSI or KS on the top 20 features by importance and on the score distribution, compared with a reference window, plus approval rate by segment; catches population shifts such as a new marketing channel bringing riskier applicants. Layer 4, early outcome proxies (weekly): first-payment default, missed first instalment, early fraud flags, which arrive within 30 days and correlate with 90-day default; catches concept drift earlier than the final label. Layer 5, matured labels (monthly): AUC, calibration and default rate by score band on loans old enough. Alerts on sustained deviations route to the model owner with a runbook covering rollback to the previous model or tightening the approval threshold."
sources: [{"title": "Google, Monitoring ML systems (Rules of ML, Rules 8-11)", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Freshness, silent failures and monitoring."}, {"title": "Lu et al., Learning under Concept Drift: A Review (2019)", "url": "https://arxiv.org/abs/2004.05785", "note": "Types of drift, detection and adaptation methods."}, {"title": "Evidently AI, Data drift explained", "url": "https://www.evidentlyai.com/ml-in-production/data-drift", "note": "Practical drift metrics and their pitfalls."}]
practice: ["population-stability-index", "two-sample-ks-drift", "page-hinkley-drift", "delayed-label-monitoring", "consecutive-breach-alerts"]
tags: ["monitoring", "drift"]
tools: ["NumPy", "SciPy"]
---

## What can go wrong after launch

A model that passed every offline gate can still fail in production: an upstream table changes, a new user segment arrives, a competitor changes prices, fraudsters adapt, or the model's own decisions change the data. Most of these failures are silent: the service returns predictions, error rates are zero, and quality quietly drops. The Machine Learning course covers drift statistics and calibration monitoring (mlcore 09.03); this lesson covers how to design monitoring for a system.

## Layers by delay

~~~dg-flow Signals arrive at very different delays. Monitoring needs a layer for each.
Request -(seconds)-> Service health -(minutes)-> Data quality -(hours)-> Input and prediction drift -(days)-> Outcome proxies -(weeks to months)-> *Label-based quality
~~~

| Layer | Examples | Catches
| --- | --- | --- |
| Service health | QPS, errors, p99 latency, fallback rate | outages, slow dependencies
| Data quality | schema, null rates, ranges, category sets | upstream bugs, broken joins
| Input drift | PSI or KS per feature, embedding distance | population shifts
| Prediction drift | score distribution, positive rate, action rate | model or feature changes
| Outcome proxies | early signals that correlate with labels | early concept drift
| Label-based quality | AUC, calibration, precision at the threshold | real quality loss

Each layer is cheap compared with the cost of finding a problem from a business report weeks later.

## Kinds of drift

- **Covariate (data) drift**: $P(x)$ changes. A marketing campaign brings younger users.
- **Prior (label) drift**: $P(y)$ changes. Fraud rates rise during holidays.
- **Concept drift**: $P(y \mid x)$ changes. The same behaviour now means something different, because fraudsters adapted or a product changed.

Covariate drift does not always hurt; a model may handle younger users well. Concept drift always hurts but can only be confirmed with labels.

## Measuring drift

Common statistics compare a recent window with a reference window (training data, or a stable recent period):

- **Population stability index (PSI)** over binned features: $\text{PSI} = \sum_b (p_b - q_b) \ln (p_b / q_b)$. Rules of thumb treat values above about 0.2 as a significant shift.
- **Kolmogorov-Smirnov** for continuous features.
- **Chi-square** or total variation distance for categorical ones.
- **Embedding drift**: distance between mean embeddings, or a classifier trained to tell old from new data.

With millions of rows, statistical tests flag tiny, harmless changes. Use effect sizes and thresholds tuned on history, and weight alerts by feature importance.

## Alerting that people trust

Alert fatigue kills monitoring. Good alerts are:

- **sustained**: several consecutive breaches, not one noisy hour;
- **important**: on features the model relies on and on business-relevant outputs such as approval rate;
- **actionable**: routed to an owner with a runbook;
- **segmented**: a shift in one country can vanish in global averages.

<div class="ex">
<div class="lab">Worked example</div>

**drift that did not matter, and drift that did**

A recommender's monitoring flagged high PSI on "device model" every autumn when new phones launched; quality was unaffected, and the team excluded the feature from alerting. The same month, a smaller shift in "session length" coincided with an app redesign that changed how sessions were counted. That shift silently degraded the model by 4% until the team added a data-quality check on the session definition.

</div>

## Delayed labels

When labels take weeks, use proxies that arrive early (02.02): first-payment default for loans, early disputes for fraud, short-term engagement for recommendations. Validate how well each proxy tracks the final label on historical data, then monitor it as an early warning.

## Dashboards and ownership

A model dashboard shows, per model version: traffic and latency, fallback usage, input and prediction distributions over time, action rates by segment, and label-based quality as labels mature. Every model in production has a named owner and an on-call path. Models without owners are where silent failures last longest.

## Monitoring the monitoring

Checks fail silently too. A drift job that stopped running looks exactly like a model with no drift. Alert on the freshness of every monitoring job and on the volume of data it processed, and review the alert list each quarter to remove checks nobody acts on and add checks for the incidents that were missed.

<div class="take">
<div class="lab">Key takeaway</div>

monitor in layers from service health to label-based quality, distinguish data drift from concept drift, alert only on large, sustained, important shifts with owners and runbooks, and use early proxies while labels mature.

</div>
