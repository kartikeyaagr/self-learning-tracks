---
id: "06.01"
course: "machine-learning"
module: "06"
slug: "classification-metrics-and-thresholds"
title: "Classification metrics and thresholds"
keyIdeas: ["A classifier usually outputs a **score**; a **threshold** turns it into decisions, and only then is there a confusion matrix. Every count-based metric is a property of the model *and* the threshold.", "**Precision** $= TP/(TP+FP)$ answers \"when it fires, is it right?\"; **recall** $= TP/(TP+FN)$ answers \"of the real positives, how many did it catch?\". Precision depends on prevalence; recall does not.", "$F_1$ is the harmonic mean of precision and recall; $F_\\beta$ weights recall $\\beta$ times as much. Neither knows what an error costs, and both ignore true negatives.", "With calibrated probabilities and per-error costs, the cost-minimising rule is: predict positive when $p > c_{FP}/(c_{FP}+c_{FN})$. A threshold of 0.5 is only right when both errors cost the same.", "For multiclass, **macro** averaging weights every class equally, **micro** pools counts (and equals accuracy for single-label problems), **weighted** averages by support. Pick the one that matches who you are serving."]
question: "A content-moderation model is evaluated on a balanced test set (50% violating posts) and reaches precision 0.90 and recall 0.80 at its threshold. In production only 1% of posts violate policy. Estimate the production precision at the same threshold, explain why it changes, and say what the team should report instead."
answer: "Recall and the false positive rate are computed within each true class, so they carry over to production. On the balanced set, recall 0.8 gives TP = 0.4N per N posts; precision 0.9 means FP = 0.4N/0.9 - 0.4N, about 0.044N, from 0.5N negatives, so **FPR = 0.089**. In production, per 10,000 posts: 100 violating, so TP = 80; 9,900 clean, so FP = 0.089 x 9,900, about 880. Precision = 80/960, roughly **0.08**. Most flags will be false alarms, because precision depends on prevalence and the negatives now outnumber positives 99 to 1. The team should report recall and FPR (prevalence-free), state the expected production prevalence, and give precision (or reviewer workload per true catch) computed at that prevalence. They should probably re-tune the threshold on data with the real class mix."
sources: [{"title": "scikit-learn: Metrics and scoring", "url": "https://scikit-learn.org/stable/modules/model_evaluation.html", "note": "The classification metrics section: confusion matrix, averaging options, MCC, with exact definitions."}, {"title": "scikit-learn: Tuning the decision threshold", "url": "https://scikit-learn.org/stable/modules/classification_threshold.html", "note": "Why 0.5 is arbitrary and how TunedThresholdClassifierCV picks a threshold for a given scorer or cost."}, {"title": "Google ML Crash Course: Accuracy, precision, recall", "url": "https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall", "note": "Short interactive treatment of the trade-off as the threshold moves."}, {"title": "Thresholding classifiers to maximize F1 (Lipton, Elkan, Narayanaswamy)", "url": "https://arxiv.org/abs/1402.1892", "note": "Shows the F1-optimal threshold for calibrated scores is half the achievable F1, and the pitfalls of optimising F1 over many labels."}]
practice: ["cross-tabulation", "conditional-aggregation", "top-k-classification-accuracy", "mean-iou-for-segmentation"]
tags: ["confusion matrix", "precision and recall", "decision threshold"]
tools: ["scikit-learn", "NumPy"]
---

## Where this module fits

Module 01 gave you honest splits and baselines; this module is about what to compute on those splits and how to use the numbers to choose a model. The first three lessons cover metrics (classification, ranking curves, regression), then cross-validation, hyperparameter search and imbalanced data. Calibration, which several of these ideas lean on, has its own lesson in module 03.

## From scores to a confusion matrix

Most classifiers output a score $s(x)$: a probability from logistic regression, a margin from an SVM, a vote fraction from a forest. Decisions need a **threshold** $t$: predict positive when $s(x) \ge t$. Only after fixing $t$ do you get the four counts of the confusion matrix.

~~~dg-flow The model ranks; the threshold decides. Change the threshold and every count-based metric moves.
Features -(model)-> Score s(x) -(compare to t)-> *Decision -(vs label)-> Confusion matrix -(ratios)-> Metrics
~~~

| Actual class | Predicted positive | Predicted negative
| --- | --- | --- |
| Actually positive | TP (true positive) | FN (false negative, a miss)
| Actually negative | FP (false positive, a false alarm) | TN (true negative)

The common rates are ratios of these counts:

- **Accuracy** $=(TP+TN)/N$. Dominated by the majority class.
- **Precision** (positive predictive value) $= TP/(TP+FP)$: the fraction of alarms that are real.
- **Recall** (sensitivity, true positive rate) $= TP/(TP+FN)$: the fraction of real positives caught.
- **Specificity** $= TN/(TN+FP)$, and the **false positive rate** $\text{FPR} = FP/(FP+TN) = 1 - \text{specificity}$.
- **$F_1$** $= 2PR/(P+R)$, the harmonic mean, which is pulled towards the smaller of the two. $F_\beta = (1+\beta^2)PR/(\beta^2 P + R)$ treats recall as $\beta$ times as important.
- **Matthews correlation coefficient** $= (TP\cdot TN - FP\cdot FN)/\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}$: a correlation between prediction and label that uses all four cells and is 0 for any constant predictor.

A detail that trips people up: recall and specificity are computed *within* a true class, so they do not change when prevalence changes. Precision mixes the two classes, so the same model has lower precision on a population where positives are rarer. A precision number quoted without the base rate is incomplete.

## Choosing the threshold from costs

The default $t = 0.5$ is a convention, not a decision. If the model's probabilities are calibrated (module 03) and a false negative costs $c_{FN}$ while a false positive costs $c_{FP}$, predicting positive for an example with probability $p$ has expected cost $(1-p)\,c_{FP}$, and predicting negative costs $p\,c_{FN}$. Predict positive when the first is smaller:

$$p > t^* = \frac{c_{FP}}{c_{FP} + c_{FN}}$$

If a missed fraud costs 50 times a manual review, $t^* = 1/51 \approx 0.02$. When probabilities are not calibrated, or costs are fuzzy, pick the threshold empirically: sweep $t$ on validation data and take the value that minimises total cost or meets a constraint ("recall at least 0.9"). The next lesson turns that sweep into curves.

```python
import numpy as np
from sklearn.metrics import confusion_matrix
def cost_at(t, y, p, c_fp=10, c_fn=500):
    tn, fp, fn, tp = confusion_matrix(y, p >= t).ravel()
    return fp * c_fp + fn * c_fn
ts = np.linspace(0.005, 0.5, 100)
best_t = ts[np.argmin([cost_at(t, y_val, p_val) for t in ts])]   # choose on validation, report on test
```

scikit-learn wraps this as `TunedThresholdClassifierCV`, which picks the threshold by cross-validation for any scorer, including a custom cost.

## Multiclass averaging

With $K$ classes you get a $K \times K$ confusion matrix and one precision/recall pair per class (that class versus the rest). Three ways to summarise:

- **Macro**: average the per-class scores. Every class counts equally, so a rare class with recall 0.2 drags the score down. Use it when small classes matter.
- **Micro**: pool TP, FP and FN over classes, then compute. For single-label problems micro-precision, micro-recall and micro-$F_1$ all equal accuracy.
- **Weighted**: average per-class scores weighted by class support. It looks like macro but behaves like accuracy; it hides weak rare classes.

Report the per-class table next to whichever average you pick; the average alone hides which class is failing.

<div class="ex">
<div class="lab">Worked example</div>

**fraud screening at two thresholds**

10,000 transactions, 200 fraudulent (2%). At $t = 0.5$: TP = 120, FN = 80, FP = 180, TN = 9,620. Accuracy 97.4%, which is *worse* than the 98% of predicting "never fraud". Precision 120/300 = 0.40, recall 0.60, $F_1$ = 0.48, MCC about 0.48.

Costs: a missed fraud loses 500, a review costs 10. Total at $t = 0.5$: 80 x 500 + 180 x 10 = 41,800. The cost-optimal threshold for calibrated scores is 10/510, about 0.02. Suppose at $t = 0.02$ the model gives TP = 180, FN = 20, FP = 1,400. Precision collapses to 0.11 and $F_1$ falls to 0.20, yet total cost drops to 20 x 500 + 1,400 x 10 = 24,000. The "worse" model by $F_1$ saves 43% of the money.

</div>

## Which number to report

- If errors have known costs, report expected cost at the chosen threshold, plus precision and recall so people can reason about workload.
- If the threshold is not yet fixed, report a threshold-free measure (next lesson) and pick the operating point later.
- Use $F_1$ only when there is no cost information and positives and negatives of the prediction really are symmetric in importance; it is a convenient default for comparing systems, not a business objective.
- Always state prevalence alongside precision, and the threshold alongside any count-based metric.

<div class="take">
<div class="lab">Key takeaway</div>

a confusion matrix belongs to a model plus a threshold; choose the threshold from costs or constraints on validation data, and report precision with its base rate.

</div>
