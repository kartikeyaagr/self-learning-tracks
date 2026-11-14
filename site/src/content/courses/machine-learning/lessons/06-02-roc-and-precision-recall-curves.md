---
id: "06.02"
course: "machine-learning"
module: "06"
slug: "roc-and-precision-recall-curves"
title: "ROC and precision-recall curves"
keyIdeas: ["Sweeping the threshold traces a curve. The **ROC curve** plots TPR against FPR; the **PR curve** plots precision against recall. Both evaluate the ranking, not one decision.", "ROC AUC equals the probability that a random positive scores above a random negative (the Mann-Whitney statistic). It ignores calibration and is unchanged by prevalence.", "Under heavy imbalance, a tiny FPR still means many false positives, so ROC looks good while precision is poor. The PR curve exposes this; its random baseline is the prevalence, not 0.5.", "**Average precision** summarises the PR curve as a step sum $\\sum_n (R_n - R_{n-1}) P_n$; linear interpolation between PR points overstates it.", "An operating point is chosen where the curve meets your constraint or cost line, on validation data. A higher AUC does not guarantee a better model at *that* point."]
question: "Two fraud models are compared on a test set where 0.2% of transactions are fraud. Model X has ROC AUC 0.97 and AP 0.18; model Y has ROC AUC 0.94 and AP 0.31. Explain how both statements can be true, and which model you would expect to serve a team that can review only the top 500 transactions per day."
answer: "AUC averages ranking quality over the whole FPR range, weighting every negative equally, while AP is dominated by the top of the ranking, where precision is computed among the highest-scored cases. Model X ranks well overall (most fraud lands above most legitimate traffic), but its very top is contaminated with legitimate transactions; model Y places fraud more cleanly at the very top and loses ground further down. With 0.2% prevalence, a small FPR difference at the top equals many false alarms, which AP feels and AUC barely registers. A team reviewing 500 cases a day operates at the extreme top of the ranking, so the relevant metric is precision at 500 (or the left end of the PR curve). Model Y, with the higher AP, is very likely better there; confirm by computing precision@500 for both on validation data before choosing."
sources: [{"title": "The precision-recall plot is more informative than the ROC plot on imbalanced datasets (Saito and Rehmsmeier)", "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432", "note": "Open-access paper with clear simulations of ROC vs PR under imbalance."}, {"title": "scikit-learn: Precision-recall example", "url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html", "note": "How average precision is computed and why it is not the trapezoidal area."}, {"title": "Google ML Crash Course: ROC and AUC", "url": "https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc", "note": "The ranking interpretation of AUC and choosing a threshold, with interactive figures."}, {"title": "scikit-learn: ROC example", "url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html", "note": "One-vs-rest and micro/macro ROC for multiclass problems."}]
practice: ["auc-roc", "area-under-roc-curve-via-trapezoidal-rule", "precision-recall-curve-and-average-precision", "canary-mann-whitney", "precision-recall-at-k"]
tags: ["ROC AUC", "average precision", "operating point"]
tools: ["scikit-learn", "NumPy"]
---

## The threshold sweep

The previous lesson fixed a threshold and read one confusion matrix. Sort the validation examples by score and lower the threshold one example at a time: each step converts one example from predicted negative to predicted positive, so either TP or FP goes up by one. Recording the rates at each step gives a curve that describes the whole ranking.

~~~dg-flow One sort produces both curves; each threshold is one point on each.
Scores + labels -(sort descending)-> Ranked list -(cut at each score)-> Confusion matrix per cut -(TPR vs FPR)-> *ROC curve
Ranked list -(precision vs recall)-> *PR curve -(step sum)-> Average precision
~~~

## ROC and AUC

The **ROC curve** puts FPR on the x-axis and TPR (recall) on the y-axis. It starts at (0, 0) with the threshold above every score and ends at (1, 1) with everything predicted positive. A random ranking gives the diagonal; a perfect ranking goes straight up to (0, 1) and then across.

The area under it has a clean meaning:

$$\text{AUC} = P\big(s(x^+) > s(x^-)\big) + \tfrac12 P\big(s(x^+) = s(x^-)\big)$$

for a random positive $x^+$ and a random negative $x^-$. That is the Mann-Whitney U statistic divided by $n_+ n_-$, so you can compute AUC by ranks in $O(n \log n)$ without drawing anything. Consequences worth knowing:

- AUC is **invariant to any monotone transform** of the scores. Squaring probabilities or applying a sigmoid does not change it, so AUC says nothing about calibration.
- Because TPR and FPR are each computed within one class, AUC **does not change with prevalence**.
- Ties count half. With heavily tied scores (a coarse model, rounded probabilities), the curve has diagonal segments and the trapezoid rule is the right way to integrate them.

## Precision-recall curves

The **PR curve** puts recall on the x-axis and precision on the y-axis. Its random baseline is a horizontal line at the prevalence $\pi$: a random ranking has precision $\pi$ at every recall. So an AP of 0.3 is poor when $\pi = 0.5$ and strong when $\pi = 0.001$. Always quote the prevalence with it.

The PR curve is not monotone and its points cannot be joined with straight lines: precision between two achievable points is not a linear blend (Davis and Goadrich showed the correct interpolation in ROC space maps to a curve in PR space). The standard summary, **average precision**, avoids interpolation by using a step sum over thresholds:

$$\text{AP} = \sum_n (R_n - R_{n-1})\, P_n$$

scikit-learn's `average_precision_score` computes exactly this; `auc(recall, precision)` uses trapezoids and is optimistic.

## Why ROC can flatter an imbalanced problem

The x-axis of ROC is a rate over negatives. When negatives outnumber positives 100 to 1, an FPR of 0.01 sounds negligible but equals as many false positives as there are positives in total. Precision, which is what a reviewer or user experiences, divides by the count of alarms and feels that directly.

$$\text{precision} = \frac{\pi\,\text{TPR}}{\pi\,\text{TPR} + (1-\pi)\,\text{FPR}}$$

This formula also lets you translate any ROC point into a precision for a given deployment prevalence, which is useful when the test set was rebalanced.

| Question | Better tool
| --- | --- |
| How well does the model rank, independent of class mix? | ROC AUC
| How many alarms are real at the recall I need, on a rare class? | PR curve, AP
| Is the probability itself trustworthy? | neither: reliability diagram, log loss (module 03)
| Which threshold do I ship? | the curve plus a cost or constraint, on validation data

<div class="ex">
<div class="lab">Worked example</div>

**higher AUC, worse product**

100,000 validation cases, 1,000 positive (1%). Model A has ROC AUC 0.95, model B 0.93. The product needs recall 0.80.

At recall 0.80, model A sits at FPR 0.03: 2,970 false positives against 800 true ones, precision 0.21. Model B's curve is steeper in that region and reaches recall 0.80 at FPR 0.01: 990 false positives, precision 0.45.

B wins where the product operates, despite the lower AUC; A's extra area comes from the high-recall corner nobody will deploy. Comparing the PR curves or "precision at recall 0.8" would have shown this immediately.

</div>

## Choosing an operating point

Common ways to pick a threshold from the curve, always on validation data, then confirmed once on test:

1. **Constraint**: "recall at least 0.95" (screening) or "FPR at most 0.1%" (spam that hides real mail). Take the threshold of the first point meeting it.
2. **Cost**: expected cost is $c_{FN}\,P\,(1-\text{TPR}) + c_{FP}\,N\,\text{FPR}$. Lines of equal cost in ROC space have slope $c_{FP}N / (c_{FN}P)$; the best point is where such a line touches the curve.
3. **Capacity**: "the review team handles 300 cases a day" fixes the number of alarms, i.e. a top-$k$ cut. Then precision at $k$ is the metric.

```python
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
print(roc_auc_score(y_val, s_val), average_precision_score(y_val, s_val), y_val.mean())
prec, rec, thr = precision_recall_curve(y_val, s_val)        # len(thr) == len(prec) - 1
ok = rec[:-1] >= 0.80
t = thr[ok][-1]                     # highest threshold that still meets the recall target
```

Two cautions. A curve estimated from few positives is noisy: with 50 positives each one moves recall by 0.02, so bootstrap the curve (ML Maths covers the bootstrap) before arguing about the third decimal of AUC. And if the deployment prevalence differs from validation, the PR curve and the chosen precision move, while the ROC point does not; recompute precision with the formula above.

<div class="take">
<div class="lab">Key takeaway</div>

use ROC AUC to compare rankings, the PR curve to see what a rare-class product will actually feel, and choose the operating point from a cost or constraint rather than from the area.

</div>
