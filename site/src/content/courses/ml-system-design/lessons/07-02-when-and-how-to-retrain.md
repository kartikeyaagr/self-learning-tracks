---
id: "07.02"
course: "ml-system-design"
module: "07"
slug: "when-and-how-to-retrain"
title: "When and how to retrain"
keyIdeas: ["Retrain on a **schedule**, on a **trigger** (drift, quality drop, new data volume), or **continuously**. The right cadence depends on how fast the world changes and how costly retraining and validation are.", "Measure **staleness cost** directly: train models on data ending at different dates and compare their performance on the same recent period.", "Choose the training window: sliding windows forget old patterns, expanding windows keep them, and time-decay weights do both. Fine-tuning from the last model is cheaper than training from scratch but can accumulate drift.", "Every retrained model still passes evaluation gates and a staged rollout. Automatic retraining should not mean automatic deployment without checks."]
question: "A news recommender is retrained weekly from scratch on 90 days of data and takes 20 hours to train. Engagement drops noticeably mid-week. How would you decide on a better retraining strategy, and what would you change?"
answer: "First measure staleness: train models with data ending 1, 2, 4 and 7 days before a fixed evaluation day and compare offline metrics on that day; repeat for several days. If the 1-day-old model is clearly better than the 7-day-old one, freshness matters. News content turns over quickly, so it likely does. Options: (1) daily incremental fine-tuning of the weekly model on the latest day's data, with a small learning rate, which takes a fraction of the time; (2) keep the heavy model weekly but update the parts that age fastest, such as item embeddings for new articles and popularity features, through streaming or hourly jobs; (3) use a time-decayed training window so recent days carry more weight. Whatever the cadence, run each new model through automatic gates (no slice regression, calibration, prediction sanity) and a canary before full rollout, and compare its online engagement against the previous version."
sources: [{"title": "Google Cloud, MLOps: continuous delivery and automation pipelines", "url": "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning", "note": "Continuous training triggers and pipelines."}, {"title": "Chip Huyen, Real-time machine learning: challenges and solutions (2022)", "url": "https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html", "note": "Continual learning and stateful training."}, {"title": "McMahan et al., Ad Click Prediction: a View from the Trenches (2013)", "url": "https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/", "note": "Online learning for click prediction at scale."}]
practice: ["page-hinkley-drift", "delayed-label-monitoring", "learning-rate-warmup-scheduler", "hashed-dataset-split"]
tags: ["retraining", "continual learning"]
---

## Models age

A model is a snapshot of the world at training time. As the world changes, its predictions get worse. Some systems age in hours (news, trending content, ads), some in weeks (fraud, e-commerce), some in months or years (credit risk, medical imaging). Retraining is how a system keeps up, and the strategy is a design decision with real cost.

## Measuring how fast a model ages

Do not guess the cadence; measure it. Train several models on data ending at different dates, then evaluate them all on the same recent period:

| Model trained up to | Offline AUC on today's data
| --- | --- |
| yesterday | 0.812
| 3 days ago | 0.809
| 1 week ago | 0.801
| 1 month ago | 0.779

Here a week of staleness costs about 0.01 AUC. Whether that justifies daily retraining depends on what 0.01 AUC is worth in business terms and on the cost of training and validating daily.

## Triggers

| Strategy | How | Good for
| --- | --- | --- |
| Scheduled | fixed cadence (daily, weekly) | predictable systems; simple operations
| Drift-triggered | retrain when drift or quality alerts fire | slow-changing domains with occasional shifts
| Volume-triggered | retrain when enough new labelled data arrives | sparse labels
| Continuous / online | update weights from a stream of examples | fast-changing domains (ads, news)

Many teams combine a scheduled baseline with triggers for exceptional events.

~~~dg-flow A continuous training loop. Retraining is automatic; deployment still goes through gates.
New data + labels -(trigger: schedule or drift)-> Training pipeline -(candidate)-> Evaluation gates -(pass)-> Canary -(healthy)-> *Production -(monitoring)-> Trigger
~~~

## Training windows

- **Sliding window** (last N days): adapts quickly; forgets rare or seasonal patterns.
- **Expanding window** (all history): keeps everything; slow to adapt and costly.
- **Time-decay weights**: weight examples by $e^{-\lambda \cdot \text{age}}$, so recent data dominates but old data still contributes.
- **Seasonal additions**: include the same period from last year for holidays or annual events.

Choose by backtesting: simulate past retraining with each window and compare results over months.

## From scratch or incremental

Training from scratch on the full window is clean and reproducible but expensive. **Warm-starting** from the previous model and fine-tuning on new data is much cheaper and allows frequent updates, but can drift over many cycles and makes models depend on their history. A common pattern is frequent incremental updates plus a periodic full retrain to reset.

Some parts of a model age faster than others. Embeddings for new items, popularity counts and other features can be updated hourly while the main network is retrained weekly.

<div class="ex">
<div class="lab">Worked example</div>

**retraining that made things worse**

A fraud team switched to daily automatic retraining. During a holiday spike, a large campaign was mislabelled as legitimate for several days because analysts were overloaded, and the daily model learned to allow it. Adding a gate that compared the new model's block rate on known historical attacks with the current model's, and a rule that label volume per day must be within normal bounds before training, prevented a repeat.

</div>

## Gates still apply

Automatic retraining should never mean automatic deployment without checks. Every candidate passes the evaluation gates from 04.04 (comparison with production, slices, calibration, sanity) and a canary. Track how often candidates fail gates; frequent failures point to data problems.

## Costs

Frequent retraining costs compute, pipeline maintenance and validation effort, and each deploy carries risk. Estimate the value of freshness from the staleness experiment and compare it with these costs before shortening the cycle. In interviews, stating "measure the staleness curve first" is a strong answer.

## Labels and retraining together

Retraining only helps if new labels arrive in time. A model retrained daily on fraud labels that take 60 days to mature learns almost nothing new each day. Match the cadence to label availability, or train on faster proxy labels (02.02) for the frequent updates and on mature labels for the periodic full retrain. Check the volume and positive rate of each day's new labels before training; an unusual day often signals a labelling problem rather than a real change.

<div class="take">
<div class="lab">Key takeaway</div>

measure how fast your model ages before choosing a cadence, combine scheduled retraining with drift or volume triggers, choose training windows by backtesting, and send every retrained model through the same gates and canary as a new one.

</div>
