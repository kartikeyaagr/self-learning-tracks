---
id: "02.03"
course: "ml-system-design"
module: "02"
slug: "rare-events-and-class-imbalance"
title: "Rare events and class imbalance"
keyIdeas: ["Many important tasks have rare positives: fraud at 0.1%, ad conversions at 1%, harmful content at under 0.5%. Accuracy is meaningless there; use precision, recall and PR curves at the operating point.", "**Negative downsampling** cuts training cost and focuses the model on positives, but shifts predicted probabilities. Correct them if the score is used as a probability.", "**Class weights** and **focal loss** change the loss instead of the data; resampling and reweighting have similar effects for well-specified models.", "Imbalance is often less of a problem than too few positives in absolute terms. Getting more positives (better labels, longer windows) usually beats clever resampling."]
question: "An ads team trains a click model on all positives and 5% of negatives. The model predicts p = 0.20 for an impression. What is the calibrated click probability, and why does it matter for ad auctions?"
answer: "With negatives kept at rate w = 0.05, the model's odds are inflated by 1/w. The corrected probability is q = p / (p + (1 - p)/w) = 0.20 / (0.20 + 0.80/0.05) = 0.20 / 16.2 ≈ 0.0123, so about 1.2%. It matters because ad auctions rank by bid × predicted click probability and charge based on it; uncalibrated probabilities inflate expected value, misprice ads, and break comparisons between campaigns or between models trained with different sampling rates. Rankings within one model are unchanged by the correction (it is monotonic), but any use of the score as a probability needs it."
sources: [{"title": "imbalanced-learn user guide", "url": "https://imbalanced-learn.org/stable/user_guide.html", "note": "Resampling methods and their trade-offs, with code."}, {"title": "He et al., Practical Lessons from Predicting Clicks on Ads at Facebook (2014)", "url": "https://quinonero.net/Publications/predicting-clicks-facebook.pdf", "note": "Negative downsampling and the calibration correction."}, {"title": "Lin et al., Focal Loss for Dense Object Detection (2017)", "url": "https://arxiv.org/abs/1708.02002", "note": "Down-weighting easy negatives in the loss."}]
practice: ["balanced-dataloader", "stratified-sampling", "sigmoid-focal-loss", "precision-recall-curve-and-average-precision", "expected-calibration-error"]
tags: ["class imbalance", "sampling"]
---

## Why rare events are hard

When positives are 0.1% of data, a model that always predicts "negative" is 99.9% accurate and useless. The real difficulty is not the ratio itself but the absolute number of positives: 1,000 frauds in a million transactions is little to learn from, whatever you do with the 999,000 negatives. So the first question is always whether you can get more positives, for example from a longer time window, related labels or annotation.

## Metrics first

Pick metrics that ignore the easy negatives:

- **precision and recall** at the threshold you will actually use;
- the **precision-recall curve** and average precision, which react to improvements on the rare class far more than ROC AUC does;
- **cost-weighted error**: false negatives × cost + false positives × cost, using the business costs from 01.04.

The Machine Learning course covers these in "Imbalanced data" (mlcore 06.06); here the focus is the system-level choices.

## Sampling the training data

| Method | What it does | When to use
| --- | --- | --- |
| Negative downsampling | keep all positives, sample a fraction $w$ of negatives | huge datasets, to cut cost
| Positive upsampling | repeat positives | small datasets; risks overfitting duplicates
| Class weights | weight each class's loss | keep all data, shift the decision boundary
| Focal loss | down-weight easy examples in the loss | dense tasks with many trivial negatives
| Hard negative mining | train more on negatives the model finds hard | retrieval and detection

Downsampling is the workhorse in large systems. If you train on 1% of 10 billion negatives, training is 100 times cheaper with little quality loss, because most negatives are easy and redundant.

## Correcting probabilities

Downsampling negatives at rate $w$ multiplies the odds by $1/w$. To recover the true probability $q$ from the model's output $p$:
$$q = \frac{p}{p + (1 - p)/w}$$
Rankings do not change, since the map is monotonic. But any use of the score as a probability needs the correction: expected-value calculations, auction pricing, thresholds shared across models, or combining scores from several models.

~~~dg-flow Train on a downsampled set, then correct the scores before anything treats them as probabilities.
All positives + w × negatives -(train)-> Model -(score p)-> *Correct: q = p / (p + (1-p)/w) -(calibrate on true distribution)-> Probability
~~~

After correction, check calibration on an untouched holdout with the true class ratio, using a reliability diagram or expected calibration error.

<div class="ex">
<div class="lab">Worked example</div>

**two models, one threshold**

A trust team ran two spam models, one trained with 10% negative sampling and one with 50%. Both used the threshold 0.5 to block messages. The first blocked five times as many messages, because its scores were inflated by more. Correcting both to true probabilities and choosing thresholds from the business cost fixed the inconsistency.

</div>

## Choosing the operating point

With rare events the threshold matters more than the model. Plot precision and recall against the threshold on a recent holdout and choose the point that meets the business constraint: "flag at most 0.5% of transactions for review, which is the analysts' capacity" or "keep precision above 90% for automatic blocking". Many systems use two thresholds: block above a high one, send to human review between two, allow below.

## Evaluation sets for rare events

A random 10,000-example test set may contain only 10 positives, which is far too few to measure recall reliably. Stratify the test set to contain enough positives, then reweight metrics back to the true prevalence. Report confidence intervals; with 50 positives, a recall of 0.80 has a 95% interval of roughly 0.67 to 0.90.

## Getting more positives

Before tuning sampling, try to enlarge the positive set. Lengthen the time window, add related positive signals (a dispute as well as a chargeback), label a stratified sample of high-scoring unlabelled cases, or pool data across markets with a feature for the market. Each of these usually helps more than a cleverer loss, because the model's limit is the variety of positive examples it has seen.

<div class="take">
<div class="lab">Key takeaway</div>

with rare positives, measure precision, recall and cost at the operating point, downsample negatives to save compute but correct and recheck the probabilities, and make sure the test set holds enough positives to measure anything.

</div>
