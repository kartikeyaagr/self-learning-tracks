---
id: "05.01"
course: "ml-system-design"
module: "05"
slug: "classification-metrics-and-thresholds"
title: "Classification metrics and thresholds"
keyIdeas: ["Offline metrics must reflect how the score will be used: ranking quality (AUC, average precision), probability quality (log loss, calibration) or decision quality at a threshold (precision, recall, cost).", "The **operating point** is chosen from business costs and capacity, such as the number of cases reviewers can handle or a maximum false-positive rate, not from a default of 0.5.", "Base rates change between test sets and production. Precision depends on prevalence, so re-estimate it on data with the production class mix.", "Many systems use **two thresholds**: act automatically above one, send to human review between the two, and allow below."]
question: "A content-moderation model will auto-remove posts above a threshold and send borderline posts to a review team that can handle 20,000 posts a day. The platform receives 10 million posts a day, of which about 0.3% violate policy. How do you set the two thresholds, and which metrics do you report?"
answer: "Use a recent, labelled sample with the production class mix (or stratified and reweighted to 0.3% prevalence). For auto-removal, choose the lowest threshold whose precision is at or above a high bar agreed with policy, for example 98%, because wrongful removals are costly and visible; report recall at that threshold. For the review band, choose a lower threshold so that the number of posts between the two thresholds is about 20,000 a day (0.2% of traffic), which uses reviewer capacity on the most uncertain cases. Report: precision and recall of auto-removal, total recall (auto plus review, after reviewers decide), the expected daily review volume and its variance, the false-removal rate per million posts, and these numbers by language and region. Recheck the thresholds whenever the model or the traffic mix changes."
sources: [{"title": "Saito and Rehmsmeier, The Precision-Recall Plot Is More Informative than the ROC Plot on Imbalanced Datasets (2015)", "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432", "note": "Why PR curves matter for rare positives."}, {"title": "scikit-learn, Probability calibration", "url": "https://scikit-learn.org/stable/modules/calibration.html", "note": "Reliability diagrams, Platt scaling and isotonic regression."}, {"title": "Google, Machine Learning Crash Course: classification", "url": "https://developers.google.com/machine-learning/crash-course/classification", "note": "Thresholds, confusion matrices and ROC."}]
practice: ["precision-recall-curve-and-average-precision", "auc-roc", "expected-calibration-error", "log-loss-binary-cross-entropy"]
tags: ["precision and recall", "operating point"]
---

## Three questions a metric can answer

| Question | Metrics | Use when
| --- | --- | --- |
| Does the model rank positives above negatives? | ROC AUC, average precision (PR AUC) | comparing models, any threshold
| Are the scores trustworthy probabilities? | log loss, expected calibration error, reliability diagram | scores feed expected-value maths, auctions, pricing
| How good are the decisions at our threshold? | precision, recall, false-positive rate, cost | the score drives an action

Report one from each row for most systems. A model can rank well (high AUC) and still be badly calibrated, or rank well overall but poorly in the narrow region around the threshold that matters. The Machine Learning course covers the definitions (mlcore 06.01 and 06.02); this lesson is about using them in a system.

## Choosing the operating point

The threshold turns scores into actions. Choose it from what errors cost and from operational limits:

- **cost-based**: minimise $c_{FP} \cdot FP + c_{FN} \cdot FN$ on a validation set;
- **capacity-based**: flag exactly as many cases as reviewers can handle;
- **constraint-based**: maximise recall subject to precision at least 95%, or a false-positive rate below 0.1%.

Plot precision, recall and the number of flagged cases against the threshold on recent data. The right threshold is usually obvious once these curves sit next to the business constraint.

~~~dg-flow Two thresholds split traffic into three actions. The middle band uses human judgement where the model is least sure.
Score -(above high threshold)-> *Auto action -(between thresholds)-> Human review -(below low threshold)-> Allow
~~~

## Prevalence changes precision

Precision depends on how common positives are. A model with 90% recall and a 1% false-positive rate has precision 0.90 × 0.5 / (0.90 × 0.5 + 0.01 × 0.5) ≈ 99% on a balanced test set, but on production traffic with 0.3% positives:
$$\text{precision} = \frac{0.9 \times 0.003}{0.9 \times 0.003 + 0.01 \times 0.997} \approx 21\%$$
Always evaluate on data with the production class mix, or reweight a stratified sample to it.

<div class="ex">
<div class="lab">Worked example</div>

**the threshold that drifted**

A payments team set a fraud threshold to block 0.5% of transactions. After a holiday marketing campaign brought many new customers, the model's scores for new accounts rose, and the same threshold blocked 1.4% of transactions, mostly legitimate first purchases. They switched to monitoring the block rate daily and recalibrating the threshold to a target rate per customer segment, with alerts on sudden shifts.

</div>

## Calibration in a system

Calibration matters whenever scores are compared across models, combined, or used as probabilities. Retraining, sampling changes (02.03) and traffic shifts all break calibration. Recalibrate on recent data with Platt scaling or isotonic regression as a final pipeline step, and monitor calibration error in production on matured labels.

## Confidence intervals

With rare positives, metrics are noisy. A recall of 0.80 measured on 100 positives has a 95% interval of roughly 0.71 to 0.87. Report intervals, using bootstrap resampling, and do not treat small differences between models as real without them.

## Metrics per slice

Report metrics for important segments as well as overall: countries, languages, platforms, new versus established users. A threshold that is fine overall can be far too aggressive for one segment. Lesson 05.04 covers slice analysis.

## Writing it up

An evaluation summary for a classification system usually fits in one table: AUC and average precision for ranking quality; log loss and calibration error for probability quality; precision, recall and expected daily volume at each threshold; and the same three rows for the two or three most important slices, each with an interval. Put the current production model in the next column. A reviewer can then see in one glance whether the new model is better, where, and by how much.

## Thresholds per segment

A single global threshold assumes the score means the same thing everywhere. When segments differ in base rate or calibration, such as new versus established accounts, either calibrate per segment or set per-segment thresholds against the same business rule (for example the same precision target). Document the choice, because per-segment thresholds can raise fairness questions that a policy team should review.

<div class="take">
<div class="lab">Key takeaway</div>

report ranking, probability and decision metrics, choose thresholds from costs, capacity or constraints on data with the production class mix, recalibrate after every change, and give confidence intervals and per-slice numbers.

</div>
