---
id: "06.02"
course: "machine-learning"
module: "06"
slug: "roc-and-precision-recall-curves"
title: "ROC and precision-recall curves"
keyIdeas: ["Lowering the threshold through the sorted scores turns one confusion matrix into a curve. The **ROC curve** plots true positive rate against false positive rate; the **PR curve** plots precision against recall. Both judge the ranking, not one decision.", "**ROC AUC** is the probability that a random positive scores above a random negative (the Mann-Whitney statistic). It ignores calibration, is unchanged by any monotone transform of the scores, and does not depend on prevalence.", "On a rare class a tiny false positive rate still means many false alarms, so ROC can look excellent while precision is poor. The PR curve shows this; its random baseline is the prevalence, not 0.5, so always quote the prevalence with it.", "**Average precision** is the step sum $\\sum_n (R_n - R_{n-1}) P_n$, the mean precision at each positive. Joining PR points with straight lines (trapezoids) can overstate it, especially with tied scores.", "The shipped threshold, the **operating point**, comes from a constraint, a cost line or a review capacity, chosen on validation data. A higher AUC does not guarantee a better model at that point."]
question: "Two fraud models are compared on a test set where 0.2% of transactions are fraud. Model X has ROC AUC 0.97 and AP 0.18; model Y has ROC AUC 0.94 and AP 0.31. Explain how both statements can be true, and which model you would expect to serve a team that can review only the top 500 transactions per day. What would you compute to confirm it?"
answer: "AUC averages ranking quality over the whole false-positive-rate range and weights every (fraud, legitimate) pair equally, while AP is the mean precision measured at each fraud case, so it is dominated by how clean the top of the ranking is. Model X ranks well overall (most fraud lands above most legitimate traffic, giving the higher AUC), but its very top is contaminated with legitimate transactions; model Y puts fraud more cleanly at the very top and loses ground further down the list. With 0.2% prevalence there are about 500 legitimate transactions per fraud case, so a false positive rate difference that is invisible on the ROC plot, say 0.002 versus 0.006, translates into a large difference in how many alarms are false, which precision and AP feel and AUC barely registers (precision = pi*TPR / (pi*TPR + (1 - pi)*FPR)). Both AP values should also be read against the random baseline of 0.002, so both models are far better than chance. A team reviewing 500 cases a day operates at the extreme top of the ranking: a top-k cut. The relevant metric is precision at 500 (equivalently, the left end of the PR curve), and model Y, with the higher AP, is very likely better there. To confirm, compute precision@500 (and recall@500) for both models on the validation set, scaled to the daily volume, ideally with bootstrap intervals because only a few hundred fraud cases may be involved. Choose the model on that number, then check it once on the test set."
sources: [{"title": "The precision-recall plot is more informative than the ROC plot on imbalanced datasets (Saito and Rehmsmeier)", "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432", "note": "Open-access paper with clear simulations of ROC vs PR under imbalance."}, {"title": "scikit-learn: Precision-recall example", "url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html", "note": "How average precision is computed and why it is not the trapezoidal area."}, {"title": "Google ML Crash Course: ROC and AUC", "url": "https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc", "note": "The ranking interpretation of AUC and choosing a threshold, with interactive figures."}, {"title": "scikit-learn: ROC example", "url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html", "note": "One-vs-rest and micro/macro ROC for multiclass problems."}]
practice: ["auc-roc", "area-under-roc-curve-via-trapezoidal-rule", "precision-recall-curve-and-average-precision", "canary-mann-whitney", "precision-recall-at-k"]
tags: ["ROC AUC", "average precision", "operating point"]
tools: ["scikit-learn", "NumPy"]
---

## What you will learn

- How sliding the threshold through a ranked list traces a whole curve, and how to build the ROC and precision-recall curves by hand.
- What the area under the ROC curve (AUC) really measures: the chance that a random positive is scored above a random negative.
- Why ROC can look excellent on a rare-class problem while most alarms are false, and how the precision-recall curve exposes it.
- How average precision is computed, and why joining PR points with straight lines can overstate it.
- How to pick the threshold you actually ship (an operating point) from a constraint, a cost or a review capacity.

*Before you start:* you need the confusion matrix, precision, recall and false positive rate from lesson 06.01, and the idea of a validation split from module 01. Probabilities and calibration are touched on but covered properly in module 03.

## From one threshold to all of them

Lesson 06.01 fixed one threshold $t$ and read one confusion matrix. That answers "how good is this decision rule?" but not "how good is this model?", because the same model gives a different confusion matrix for every $t$. Two models can swap places when you move the threshold.

Sort the examples by score and look at every possible cut at once. Each cut is one confusion matrix; plot a pair of rates for each cut and you get a curve that describes the model's whole **ranking**.

~~~dg-flow One sort produces both curves; each threshold is one point on each.
Scores + labels -(sort descending)-> Ranked list -(cut at each score)-> Confusion matrix per cut -(TPR vs FPR)-> *ROC curve
Ranked list -(precision vs recall)-> *PR curve -(step sum)-> Average precision
~~~

## The idea in one picture

A spam filter scores ten emails. Four are spam (positive), six are not. Sort them from highest score to lowest, then lower the threshold one email at a time. Each step moves one email from "predicted not spam" to "predicted spam", so either the true-positive count TP or the false-positive count FP goes up by one.

| Rank | Score | Spam? | TP | FP | TPR (recall) | FPR | Precision
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.95 | yes | 1 | 0 | 0.25 | 0.00 | 1.00
| 2 | 0.90 | yes | 2 | 0 | 0.50 | 0.00 | 1.00
| 3 | 0.80 | no | 2 | 1 | 0.50 | 0.17 | 0.67
| 4 | 0.70 | yes | 3 | 1 | 0.75 | 0.17 | 0.75
| 5 | 0.60 | no | 3 | 2 | 0.75 | 0.33 | 0.60
| 6 | 0.55 | yes | 4 | 2 | 1.00 | 0.33 | 0.67
| 7 | 0.45 | no | 4 | 3 | 1.00 | 0.50 | 0.57
| 8 | 0.30 | no | 4 | 4 | 1.00 | 0.67 | 0.50
| 9 | 0.20 | no | 4 | 5 | 1.00 | 0.83 | 0.44
| 10 | 0.10 | no | 4 | 6 | 1.00 | 1.00 | 0.40

Row $k$ is the confusion matrix you get with threshold $t$ equal to the $k$-th score. The **true positive rate** is $\text{TPR} = TP/4$ (the share of spam caught), the **false positive rate** is $\text{FPR} = FP/6$ (the share of good mail wrongly flagged), and **precision** is $TP/(TP+FP)$ (the share of flags that are right).

~~~dg-fig 06.02-sweep.svg
The same ten rows plotted two ways. Left: each spam email moves the ROC curve up, each good email moves it right. Right: precision jumps up and down as spam and good mail alternate in the ranking.
~~~

Read the figure in plain words:

- The **ROC curve** (left) climbs steeply at first because the two highest scores are both spam. A perfect ranking would go straight up to the top-left corner; a random ranking hugs the dashed diagonal. The shaded area is 0.875.
- The **precision-recall (PR) curve** (right) starts at precision 1, then saw-tooths downward. It ends at 0.4, which is simply the fraction of spam in the data: flag everything and 40% of your flags are right.

## Step 1: the ROC curve

The ROC curve (the name comes from radar "receiver operating characteristic" analysis) puts FPR on the x-axis and TPR on the y-axis:

$$\text{TPR}(t) = \frac{TP(t)}{P}, \qquad \text{FPR}(t) = \frac{FP(t)}{N},$$

where $P$ is the number of actual positives, $N$ the number of actual negatives, and $TP(t)$, $FP(t)$ the counts at threshold $t$. Three landmarks:

- With $t$ above every score nothing is flagged: the point (0, 0).
- With $t$ below every score everything is flagged: the point (1, 1).
- In between, a positive moves the curve up by $1/P$ and a negative moves it right by $1/N$. In our table, $1/P = 0.25$ and $1/N \approx 0.17$.

So the shape *is* the ranking: positives at the top make tall vertical runs; an early negative makes a low horizontal step.

## Step 2: the area under the ROC curve

The **ROC AUC** squeezes the whole curve into one number between 0 and 1. Its meaning is cleaner than "area" suggests:

$$\text{AUC} = P\big(s(x^+) > s(x^-)\big) + \tfrac12 P\big(s(x^+) = s(x^-)\big),$$

where $x^+$ is a randomly chosen positive, $x^-$ a randomly chosen negative, and $s(\cdot)$ the model's score. In words: pick one spam and one good email at random; AUC is the probability the spam scores higher. Ties count half. A random model scores 0.5, a perfect one 1.0, and a model below 0.5 is ranking backwards (flip its sign).

This is the **Mann-Whitney U statistic** divided by $P \cdot N$, so you can compute AUC by counting pairs or from ranks in $O(n \log n)$ time, without ever drawing the curve.

<div class="ex">
<div class="lab">Worked example</div>

**AUC of the spam filter, two ways**

Pair counting. There are $4 \times 6 = 24$ (spam, good) pairs. For each spam email, count the good emails ranked below it.

The spam at 0.95 beats all 6 good emails. The spam at 0.90 also beats all 6. The spam at 0.70 is below the good email at 0.80, so it beats 5. The spam at 0.55 is below the good emails at 0.80 and 0.60, so it beats 4.

Total wins: $6 + 6 + 5 + 4 = 21$. AUC $= 21/24 = 0.875$.

Area. The curve is a staircase, so the area is a sum of rectangles. Each time FPR moves right by $1/6$, the height is the current TPR: the first good email (rank 3) is passed at TPR 0.50, the second (rank 5) at TPR 0.75, and the last four at TPR 1.00.

Area $= \tfrac16(0.50) + \tfrac16(0.75) + \tfrac46(1.00) = 0.0833 + 0.125 + 0.6667 = 0.875$. Same answer, as it must be.

Ranks. Give the scores ranks 1 (lowest) to 10 (highest). The spam emails have ranks 10, 9, 7 and 5, summing to 31. Subtract the smallest possible sum $4 \cdot 5/2 = 10$ and divide by 24: $(31 - 10)/24 = 0.875$.

</div>

The next figure shows where AUC comes from. If the two classes' score distributions barely overlap, almost every positive beats almost every negative and the curve hugs the top-left corner. If they sit on top of each other, the curve is the diagonal.

~~~dg-fig 06.02-separation.svg
Left: score distributions of negatives and of positives for two models. Right: the more the distributions overlap, the closer the ROC curve gets to the diagonal and the closer AUC gets to 0.5.
~~~

Three consequences follow directly from the pair-counting definition:

- AUC is **invariant to any monotone transform** of the scores. Squaring probabilities, taking logs or applying a sigmoid leaves the ranking, and so the AUC, unchanged. AUC therefore says nothing about whether a score of 0.8 means "80% likely" (that is calibration, module 03).
- TPR is computed only among positives and FPR only among negatives, so AUC **does not change with prevalence** (the fraction of positives).
- With heavily tied scores (a coarse model, rounded probabilities) the ROC curve has diagonal segments; the trapezoid rule integrates them exactly and matches the "ties count half" rule.

## Step 3: the precision-recall curve

The PR curve puts recall (the same quantity as TPR) on the x-axis and precision on the y-axis. Its random baseline is not a diagonal but a horizontal line at the **prevalence** $\pi = P/(P+N)$: if the ranking is random, every slice of it contains positives at the base rate, so precision is $\pi$ at every recall. In the spam example $\pi = 0.4$.

That makes PR numbers meaningless without the prevalence next to them. A PR summary of 0.3 is poor when $\pi = 0.5$ and strong when $\pi = 0.001$.

Unlike ROC, the PR curve is not monotone. Precision rises when a positive is added (rank 4: from 0.67 to 0.75) and falls when a negative is added (rank 5: down to 0.60). This saw-tooth is normal.

## Step 4: why ROC can flatter a rare-class problem

Here is the central reason PR curves exist. Precision can be written with the rates and the prevalence:

$$\text{precision} = \frac{\pi\,\text{TPR}}{\pi\,\text{TPR} + (1-\pi)\,\text{FPR}}.$$

Every symbol is defined above: $\pi$ is the prevalence, TPR and FPR are the coordinates of a point on the ROC curve. Number check with the spam table at rank 4: $\pi = 0.4$, TPR $= 0.75$, FPR $= 1/6$. Then $0.4 \times 0.75 = 0.30$, $0.6 \times 0.1667 = 0.10$, and precision $= 0.30/0.40 = 0.75$, matching the table.

Now set $\pi = 0.01$ (1 positive in 100) and take an ROC point that looks superb, TPR 0.80 at FPR 0.03. Precision is $0.008/(0.008 + 0.0297) = 0.21$: four of every five alarms are false.

~~~dg-fig 06.02-imbalance.svg
The same model scored on a balanced population and on one with 1% positives. The ROC curves lie on top of each other (AUC 0.92 both); the PR curves are worlds apart (AP 0.92 versus 0.27).
~~~

The figure is the whole argument. ROC describes the model; PR describes the experience of whoever reads the alarms. The formula also gives you a practical tool: if your test set was rebalanced (say 50/50) but production has 1% positives, keep the ROC point and recompute the precision with the production $\pi$.

## Step 5: average precision

The standard one-number summary of the PR curve is **average precision (AP)**:

$$\text{AP} = \sum_n (R_n - R_{n-1})\, P_n,$$

where $n$ runs over the thresholds from highest to lowest, $R_n$ and $P_n$ are the recall and precision at threshold $n$, and $R_0 = 0$. Recall only increases when a positive is passed, so AP is the average of the precision values measured *at each positive*.

<div class="ex">
<div class="lab">Worked example</div>

**average precision of the spam filter**

Recall moves at ranks 1, 2, 4 and 6 (the spam emails), each time by $1/4 = 0.25$.

Precision at those ranks: $1/1 = 1.00$, $2/2 = 1.00$, $3/4 = 0.75$, $4/6 = 0.667$.

AP $= 0.25 \times (1.00 + 1.00 + 0.75 + 0.667) = 0.25 \times 3.417 = 0.854$.

</div>

Why not just take the area with trapezoids? Because in PR space a straight line between two achievable points is not achievable. The case where it matters most is a jump: tied scores or sparse thresholds, where recall leaps and precision falls in one step.

~~~dg-fig 06.02-ap.svg
A coarse model gives six emails the same score. The step sum (AP 0.750) only credits precision that a threshold actually reaches; the straight line (0.875) claims precision no threshold achieves; even breaking the tie at random gives a curved path (0.821), not a straight one.
~~~

`average_precision_score` computes the step sum; `auc(recall, precision)` uses trapezoids. On fine-grained curves the two are usually close, but on coarse or tied scores the trapezoid can be clearly optimistic, so report AP.

## ROC and PR in code

On a realistic size problem you never sort by hand. A 2%-positive dataset, a logistic regression, and everything from this lesson:

```python
import numpy as np
from scipy.stats import rankdata
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

X, y = make_classification(n_samples=20_000, n_features=20, n_informative=5,
                           weights=[0.98], flip_y=0.0, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
s_val = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_val)[:, 1]

print(y_val.mean())                              # 0.020   prevalence: quote it with AP
print(roc_auc_score(y_val, s_val))               # 0.965
print(average_precision_score(y_val, s_val))     # 0.779

r = rankdata(s_val)                              # AUC from ranks (Mann-Whitney), no curve
n_pos, n_neg = y_val.sum(), (1 - y_val).sum()
print((r[y_val == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))   # 0.965

prec, rec, thr = precision_recall_curve(y_val, s_val)   # len(thr) == len(prec) - 1
print(auc(rec, prec))                            # 0.778  trapezoids, close here (fine curve)
i = np.where(rec[:-1] >= 0.80)[0][-1]            # highest threshold still meeting recall 0.80
print(thr[i], prec[i], rec[i])                   # 0.095  0.508  0.800

top = np.argsort(-s_val)[:100]                   # capacity: a team reviews 100 cases
print(y_val[top].mean())                         # 0.95   precision@100
```

Notice: AUC 0.965 sounds near-perfect, yet at recall 0.80 only half the alarms are real; the threshold for recall 0.80 is 0.095, far from the default 0.5; and the top 100 are 95% positive. Which number matters depends on how the model is used.

## Choosing an operating point

A curve describes every threshold; a product runs at one. That chosen point is the **operating point**. Pick it on validation data, then confirm it once on the test set. There are three common ways:

1. **Constraint**: "recall at least 0.95" (medical screening) or "FPR at most 0.1%" (a spam filter that must not bury real mail). Walk down the thresholds and take the first one that satisfies it, as in the code above.
2. **Cost**: if a missed positive costs $c_{FN}$ and a false alarm costs $c_{FP}$, the expected total cost is $c_{FN}\,P\,(1-\text{TPR}) + c_{FP}\,N\,\text{FPR}$. Points of equal cost form straight lines in ROC space with slope $c_{FP}N / (c_{FN}P)$; the cheapest operating point is where such a line just touches the curve. Check: 1,000 positives, 99,000 negatives, and a miss costing 50 times a false alarm give slope $99{,}000/50{,}000 = 1.98$.
3. **Capacity**: "the review team handles 300 cases a day" fixes the number of alarms. That is a top-$k$ cut, and **precision at $k$** is the metric to compare models on.

<div class="ex">
<div class="lab">Worked example</div>

**higher AUC, worse product**

100,000 validation cases, 1,000 positive (1%), so 99,000 negatives. Model A has ROC AUC 0.95, model B 0.93. The product needs recall 0.80, which means catching 800 positives.

Model A reaches recall 0.80 at FPR 0.03: $0.03 \times 99{,}000 = 2{,}970$ false positives. Precision $= 800/(800 + 2{,}970) = 800/3{,}770 = 0.21$.

Model B's curve is steeper in that region and reaches recall 0.80 at FPR 0.01: $990$ false positives. Precision $= 800/1{,}790 = 0.45$.

B makes fewer than half as many false alarms where the product operates, despite the lower AUC. A's extra area comes from the high-FPR region (beyond about 0.09, where the two curves cross) that nobody will deploy.

</div>

~~~dg-fig 06.02-operating.svg
Left: B's ROC curve rises faster at small FPR, so it hits recall 0.80 first even though A has more total area. Right: at 1% prevalence that small FPR gap doubles the precision at recall 0.80.
~~~

## Reading and checking the curves

A few habits that catch most problems:

- **Always print the prevalence next to AP.** Compare AP to $\pi$, not to 0.5.
- **Look at the region you will use.** Zoom into small FPR, or read precision at your target recall, instead of relying on the total area.
- **Count the positives.** With 50 positives, each one moves recall by 0.02 and the curve is jagged. Bootstrap the validation set (resample with replacement, recompute, repeat 1,000 times; ML Maths covers it) to get an interval before arguing about the third decimal of AUC.
- **Check the extremes.** AUC near 1.0 on a hard problem usually means leakage (module 01): a feature that encodes the label. AUC well below 0.5 usually means the labels or the scores are flipped.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Reporting only ROC AUC on a rare class | AUC 0.97, yet reviewers complain most alarms are false | Add the PR curve, AP and precision at the operating recall
| Quoting AP without prevalence | "AP 0.3" read as bad when $\pi = 0.002$ | State $\pi$; compare AP with it
| `auc(recall, precision)` instead of AP | Summary higher than `average_precision_score`, especially with ties | Use `average_precision_score`
| Passing hard labels instead of scores | ROC "curve" with one corner, AUC equals balanced accuracy | Pass `predict_proba[:, 1]` or `decision_function`
| Choosing the threshold on the test set | Test metrics look better than production | Pick on validation, confirm once on test
| Picking the model with the higher AUC | Worse precision at the recall the product needs | Compare at the operating point, or use partial AUC

## Going deeper

**ROC convex hull.** If two thresholds give ROC points $a$ and $b$, flipping a biased coin to choose between them for each example achieves any point on the straight segment between $a$ and $b$. So the *achievable* ROC region is the convex hull of the curve, and an optimal operating point under any linear cost always lies on that hull.

**ROC and PR are linked but not equivalent.** For a fixed dataset, each ROC point maps to exactly one PR point through the precision formula above. Davis and Goadrich showed that one curve dominates another in ROC space if and only if it dominates in PR space. But a straight ROC segment maps to a *curve* in PR space, which is why PR points cannot be joined linearly, and why the areas can rank two crossing curves differently.

**PR has an unachievable region.** Even the worst ranking (every negative above every positive) has precision $\pi R/(\pi R + 1 - \pi)$ at recall $R$, which is above zero. The lowest possible AP therefore depends on $\pi$, one more reason to compare AP only between models evaluated on the same prevalence.

**Partial AUC and Gini.** When only small FPRs are acceptable, integrate the ROC curve over $[0, \text{FPR}_{\max}]$: `roc_auc_score(y, s, max_fpr=0.05)` returns a standardised partial AUC. Credit scoring often reports the **Gini coefficient** $2\,\text{AUC} - 1$, the same information rescaled to $[-1, 1]$.

**Multiclass.** ROC is binary. For $K$ classes, compute one-vs-rest curves (each class against the rest) and average them: macro averaging weights classes equally, micro averaging pools all decisions so frequent classes dominate. In scikit-learn: `roc_auc_score(y, P, multi_class="ovr", average="macro")`.

## Key terms

| Term | Meaning
| --- | --- |
| Threshold sweep | Lowering the decision threshold through every score to get one confusion matrix per cut
| ROC curve | True positive rate plotted against false positive rate over all thresholds
| ROC AUC | Area under the ROC curve; the probability a random positive outscores a random negative
| Mann-Whitney U | The pair-counting rank statistic that equals AUC times $P \cdot N$
| Precision-recall curve | Precision plotted against recall over all thresholds; random baseline at the prevalence
| Prevalence | The fraction of examples that are positive, $\pi = P/(P+N)$
| Average precision | Step sum $\sum_n (R_n - R_{n-1}) P_n$: mean precision measured at each positive
| Operating point | The single threshold (point on the curve) a deployed system uses
| Precision at k | Fraction of the top $k$ scored cases that are positive; the metric for a fixed review capacity

<div class="take">
<div class="lab">Key takeaway</div>

sweep the threshold to see the whole ranking; use ROC AUC to compare rankings independently of class mix, the PR curve and AP (always next to the prevalence) to see what a rare-class product will actually feel, and choose the operating point from a constraint, cost or capacity on validation data rather than from the area.

</div>
