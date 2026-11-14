---
id: "06.01"
course: "machine-learning"
module: "06"
slug: "classification-metrics-and-thresholds"
title: "Classification metrics and thresholds"
keyIdeas: ["A classifier usually outputs a **score**; a **threshold** turns it into decisions, and only then is there a confusion matrix. Every count-based metric is a property of the model *and* the threshold, and `predict()` silently uses 0.5.", "**Precision** $= TP/(TP+FP)$ answers \"when it fires, is it right?\"; **recall** $= TP/(TP+FN)$ answers \"of the real positives, how many did it catch?\". Precision depends on prevalence; recall does not.", "Accuracy is dominated by the majority class: with 2% positives, \"never positive\" scores 98%. Compare every model with that baseline and prefer recall, precision, MCC or cost.", "$F_1$ is the harmonic mean of precision and recall; $F_\\beta$ weights recall $\\beta$ times as much. Neither knows what an error costs, and both ignore true negatives.", "With calibrated probabilities and per-error costs, the cost-minimising rule is: predict positive when $p > c_{FP}/(c_{FP}+c_{FN})$. Otherwise sweep the threshold on validation data and report the result on test.", "For multiclass, **macro** averaging weights every class equally, **micro** pools counts (and equals accuracy for single-label problems), **weighted** averages by support. Always show the per-class table too."]
question: "A content-moderation model is evaluated on a balanced test set (50% violating posts) and reaches precision 0.90 and recall 0.80 at its threshold. In production only 1% of posts violate policy. Estimate the production precision at the same threshold, explain why it changes, and say what the team should report instead."
answer: "Recall and the false positive rate are computed within each true class, so they carry over to production as long as the posts themselves look alike. On the balanced set, take N posts: 0.5N violate and 0.5N are clean. Recall 0.8 gives TP = 0.4N. Precision 0.9 means TP/(TP+FP) = 0.9, so FP = 0.4N/0.9 - 0.4N, about 0.044N, out of 0.5N clean posts: **FPR = 0.089**. In production, per 10,000 posts: 100 violate, so TP = 0.8 x 100 = 80; 9,900 are clean, so FP = 0.089 x 9,900, about 880. Precision = 80/(80 + 880) = 80/960, roughly **0.08**. The same formula in one line: precision = R pi / (R pi + FPR (1 - pi)) = 0.008/(0.008 + 0.088) = 0.083. More than 9 in 10 flags will be false alarms, because precision mixes the two classes and the clean posts now outnumber violating ones 99 to 1, so even a small false positive rate produces many false alarms. The team should report recall and FPR (which do not depend on prevalence), state the expected production prevalence, and give precision (or reviewer workload per true catch, here about 12 reviews per real violation) computed at that prevalence. They should then re-tune the threshold on validation data with the real class mix, against the actual cost of a missed violation versus a wasted review, and possibly add a second-stage model or human review queue sized to the expected alert volume."
sources: [{"title": "scikit-learn: Metrics and scoring", "url": "https://scikit-learn.org/stable/modules/model_evaluation.html", "note": "The classification metrics section: confusion matrix, averaging options, MCC, with exact definitions."}, {"title": "scikit-learn: Tuning the decision threshold", "url": "https://scikit-learn.org/stable/modules/classification_threshold.html", "note": "Why 0.5 is arbitrary and how TunedThresholdClassifierCV picks a threshold for a given scorer or cost."}, {"title": "Google ML Crash Course: Accuracy, precision, recall", "url": "https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall", "note": "Short interactive treatment of the trade-off as the threshold moves."}, {"title": "Thresholding classifiers to maximize F1 (Lipton, Elkan, Narayanaswamy)", "url": "https://arxiv.org/abs/1402.1892", "note": "Shows the F1-optimal threshold for calibrated scores is half the achievable F1, and the pitfalls of optimising F1 over many labels."}]
practice: ["cross-tabulation", "conditional-aggregation", "top-k-classification-accuracy", "mean-iou-for-segmentation"]
tags: ["confusion matrix", "precision and recall", "decision threshold"]
tools: ["scikit-learn", "NumPy"]
---

## What you will learn

- Why a classifier really produces a **score**, and why a **threshold** has to be chosen before any yes/no metric exists.
- How to build a confusion matrix and read precision, recall, specificity, $F_1$ and MCC from its four cells.
- Why accuracy misleads on rare classes, and why precision changes when the share of positives changes while recall does not.
- How to pick a threshold from the cost of each kind of mistake, by formula and by sweeping it on validation data.
- How macro, micro and weighted averaging summarise a multiclass problem, and which one hides a failing class.

*Before you start:* you should know what a train/validation/test split is and why you never tune on the test set (module 01), and what a predicted probability from logistic regression means (02.03). Calibration (module 03) is mentioned here and is helpful but not required.

## Where this module starts

Every module so far ended with "check it on held-out data". This module is about what number to compute there, and how to use it to choose between models. A wrong metric quietly picks the wrong model. The order: classification metrics and thresholds (this lesson), ROC and precision-recall curves (06.02), regression metrics (06.03), cross-validation (06.04), hyperparameter search (06.05) and imbalanced data (06.06).

## The idea in one picture

A bank's model scores twelve card transactions. Four are truly fraud. The model does not say "fraud" or "not fraud"; it says how suspicious each transaction looks, as a number between 0 and 1.

| Truth | Scores the model gave
| --- | --- |
| Fraud (4) | 0.92, 0.81, 0.64, 0.38
| Legitimate (8) | 0.71, 0.55, 0.44, 0.33, 0.26, 0.18, 0.11, 0.04

To act, the bank needs a rule: block every transaction scoring at least some cut-off $t$. With $t = 0.5$:

~~~dg-fig 06.01-scores.svg
Twelve transactions placed by score. The threshold line at 0.5 sorts them into four groups: caught fraud (TP), missed fraud (FN), wrongly blocked legitimate payments (FP) and correctly passed ones (TN).
~~~

Three frauds score above 0.5 and are caught; one (0.38) slips through. Two honest payments (0.71 and 0.55) are blocked by mistake; six pass. Four groups:

- **True positive (TP)**: a real positive the model flagged. Here 3.
- **False negative (FN)**: a real positive the model missed. Here 1.
- **False positive (FP)**: a real negative the model flagged, a false alarm. Here 2.
- **True negative (TN)**: a real negative the model let pass. Here 6.

These four counts in a 2 by 2 grid are the **confusion matrix**. Now slide the line. At $t = 0.75$ two frauds are caught and nobody honest is blocked; at $t = 0.3$ every fraud is caught but four honest customers are blocked. Same model, very different behaviour: a metric like precision belongs to a model *plus* a threshold.

## Step 1: from scores to decisions

Almost every classifier outputs a **score** $s(x)$: a probability from logistic regression, a signed distance from an SVM, a vote fraction from a forest. Higher means more likely positive. A **decision threshold** $t$ turns the score into an action:

$$\hat y = \begin{cases} 1 & \text{if } s(x) \ge t \\ 0 & \text{otherwise} \end{cases}$$

Here $\hat y$ is the predicted label, 1 means "positive" (the thing you are looking for: fraud, disease, spam) and 0 means "negative".

~~~dg-flow The model ranks; the threshold decides. Change the threshold and every count-based metric moves.
Features -(model)-> Score s(x) -(compare to t)-> *Decision -(vs label)-> Confusion matrix -(ratios)-> Metrics
~~~

`model.predict(X)` in scikit-learn silently uses $t = 0.5$ on the probability (or 0 on the margin), a convention rather than a decision about your problem. Use `predict_proba` and choose $t$ yourself.

## Step 2: the confusion matrix

Lay the four counts out with the truth in rows and the prediction in columns:

| Actual class | Predicted positive | Predicted negative
| --- | --- | --- |
| Actually positive | TP (true positive) | FN (false negative, a miss)
| Actually negative | FP (false positive, a false alarm) | TN (true negative)

For the twelve transactions at $t = 0.5$ the grid is TP = 3, FN = 1, FP = 2, TN = 6. Two sanity checks always hold: the first row sums to the number of real positives ($3 + 1 = 4$), and the second row to the number of real negatives ($2 + 6 = 8$). Rows are fixed by the data; only the split within each row moves with the threshold.

## Step 3: the rates you can read off it

Each metric is a ratio of cells that answers one question.

**Accuracy**: what fraction of all decisions were right?
$$\text{accuracy} = \frac{TP + TN}{N}, \qquad N = TP+TN+FP+FN.$$
Check: $(3 + 6)/12 = 0.75$.

**Precision** (also *positive predictive value*): when the model raises an alarm, how often is it real? It looks only at the predicted-positive column.
$$\text{precision} = \frac{TP}{TP + FP}.$$
Check: $3/(3+2) = 0.60$. Six in ten blocked payments were fraud.

**Recall** (also *sensitivity* or *true positive rate*, TPR): of all real positives, how many did the model catch? It looks only at the first row.
$$\text{recall} = \frac{TP}{TP + FN}.$$
Check: $3/(3+1) = 0.75$.

**Specificity**: of all real negatives, how many were correctly left alone? Its complement is the **false positive rate** (FPR).
$$\text{specificity} = \frac{TN}{TN+FP}, \qquad \text{FPR} = \frac{FP}{FP+TN} = 1 - \text{specificity}.$$
Check: $6/8 = 0.75$, so FPR $= 0.25$.

**$F_1$**: one number that is high only when precision $P$ and recall $R$ are both high. It is their **harmonic mean**, which sits close to the smaller of the two:
$$F_1 = \frac{2PR}{P + R}.$$
Check: $2 \times 0.6 \times 0.75 / 1.35 = 0.667$. If precision were 1.0 and recall 0.1, the plain average would be a flattering 0.55 but $F_1$ is only 0.18. A generalisation, $F_\beta = (1+\beta^2)PR/(\beta^2 P + R)$, treats recall as $\beta$ times as important as precision; $F_2$ here is $5 \times 0.45 / (2.4 + 0.75) = 0.714$.

**Matthews correlation coefficient (MCC)**: the correlation between the predicted and true labels, using all four cells:
$$\text{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.$$
It runs from $-1$ (always wrong) through 0 (no better than a constant guess) to $+1$ (perfect). Check: $(18 - 2)/\sqrt{5 \cdot 4 \cdot 8 \cdot 7} = 16/33.47 = 0.48$.

## Step 4: why accuracy is not enough, and why precision moves with prevalence

**Prevalence** (or base rate) is the share of real positives in the data. In card fraud it might be 2%. A model that *never* flags anything scores 98% accuracy and catches zero fraud. Accuracy is dominated by whichever class is larger, so on rare-event problems it mostly measures the majority class.

The figure sweeps the threshold for a logistic regression on synthetic data with 3% positives (the data of the code section). The dotted accuracy line sits near 0.97 to 0.98 almost everywhere and reveals none of the trade-off.

~~~dg-fig 06.01-sweep.svg
Raising the threshold makes alarms rarer but more trustworthy: precision climbs while recall falls. F1 peaks where the two balance; accuracy barely moves because 97% of examples are negatives.
~~~

A subtler trap: recall is computed *inside* the row of real positives and specificity inside the row of negatives, so neither depends on the rows' relative sizes, and neither moves when prevalence changes. Precision mixes the two rows, so it does. By Bayes' rule, with prevalence $\pi$:

$$\text{precision} = \frac{R\,\pi}{R\,\pi + \text{FPR}\,(1 - \pi)}.$$

The numerator is the share of items that are caught positives; the denominator adds the false alarms. With recall 0.9 and FPR 0.05, precision is $0.45/(0.45 + 0.025) = 0.95$ on a balanced test set but $0.009/(0.009 + 0.0495) = 0.15$ when only 1% are positive.

~~~dg-fig 06.01-prevalence.svg
The same two models evaluated at different prevalences. Their recall is constant, but precision collapses as positives become rare. Cutting the false positive rate from 5% to 1% helps a lot, and still cannot rescue precision at 0.1%.
~~~

A precision quoted without its base rate is incomplete; a model tested on a rebalanced set looks far more precise than it will be in production.

## Step 5: choosing the threshold from costs

Mistakes rarely cost the same. A missed fraud might lose 500; a false alarm costs about 10 of analyst time. Suppose the model's probability $p$ is **calibrated** (of items scored 0.2, about 20% really are positive; see module 03). Compare the two actions:

- Flag it: you pay $c_{FP}$ only if it is actually legitimate, which has probability $1-p$. Expected cost $(1-p)\,c_{FP}$.
- Pass it: you pay $c_{FN}$ only if it is actually fraud. Expected cost $p\,c_{FN}$.

Flag whenever flagging is cheaper, $(1-p)\,c_{FP} < p\,c_{FN}$. Solving for $p$:

$$p > t^* = \frac{c_{FP}}{c_{FP} + c_{FN}}.$$

Here $c_{FP}$ is the cost of a false alarm and $c_{FN}$ the cost of a miss. With $c_{FP} = 10$ and $c_{FN} = 500$, $t^* = 10/510 \approx 0.02$: flag anything with even a 2% chance of fraud. The usual 0.5 is the special case where both errors cost the same.

When probabilities are not calibrated or costs are approximate, do it empirically: try many thresholds on validation data and keep the cheapest.

~~~dg-fig 06.01-cost.svg
Total validation cost (500 per missed fraud, 10 per review) as the threshold moves. The cheapest threshold is about 0.045, close to the theoretical 0.02 and far from the default 0.5, which costs about 40% more.
~~~

The curve is ragged because the validation set holds only about 150 positives; each caught or missed fraud shifts the cost by 500. That noise is why you choose on validation and report on separate test data.

You can also choose by a **constraint**: "recall at least 0.9, then maximise precision", or "the team can review 200 alerts a day". Lesson 06.02 turns the sweep into curves that make these choices visual.

<div class="ex">
<div class="lab">Worked example</div>

**fraud screening at two thresholds**

Setting: 10,000 transactions, 200 fraudulent (prevalence 2%). A missed fraud costs 500; a manual review costs 10.

At $t = 0.5$ the model gives TP = 120, FN = 80, FP = 180, TN = 9,620. Row checks: $120 + 80 = 200$ positives and $180 + 9{,}620 = 9{,}800$ negatives.

Accuracy $= (120 + 9{,}620)/10{,}000 = 0.974$. Predicting "never fraud" scores $9{,}800/10{,}000 = 0.98$, so by accuracy the model looks worse than doing nothing.

Precision $= 120/(120 + 180) = 0.40$. Recall $= 120/200 = 0.60$. $F_1 = 2 \times 0.40 \times 0.60/(0.40 + 0.60) = 0.48$.

MCC: numerator $120 \times 9{,}620 - 180 \times 80 = 1{,}154{,}400 - 14{,}400 = 1{,}140{,}000$. Denominator $\sqrt{300 \times 200 \times 9{,}800 \times 9{,}700} = \sqrt{5.7036 \times 10^{12}} \approx 2{,}388{,}000$. MCC $\approx 0.477$.

Cost at $t = 0.5$: $80 \times 500 + 180 \times 10 = 40{,}000 + 1{,}800 = 41{,}800$.

The cost-optimal threshold for calibrated scores is $10/510 \approx 0.02$. Suppose at that threshold the model gives TP = 180, FN = 20, FP = 1,400, TN = 8,400.

Precision $= 180/1{,}580 = 0.114$, recall $= 180/200 = 0.90$, $F_1 = 2 \times 0.114 \times 0.90/1.014 = 0.20$.

Cost: $20 \times 500 + 1{,}400 \times 10 = 10{,}000 + 14{,}000 = 24{,}000$. That is $1 - 24{,}000/41{,}800 = 43\%$ cheaper, even though $F_1$ fell from 0.48 to 0.20. The metric you optimise must be the one the business pays for.

</div>

## Step 6: more than two classes

With $K$ classes the confusion matrix is $K \times K$: row = true class, column = predicted class, diagonal = correct. Treating each class in turn as "positive" and the rest as "negative" (**one-vs-rest**) gives $K$ precision/recall pairs, summarised three ways:

- **Macro** average: compute the metric per class, then take the plain mean. Every class counts equally, so a weak rare class pulls the score down. Use it when small classes matter as much as big ones.
- **Micro** average: add up TP, FP and FN across all classes first, then compute one metric. Big classes dominate. For single-label problems micro precision, micro recall and micro $F_1$ all equal accuracy.
- **Weighted** average: per-class metric weighted by **support** (the number of true examples in that class). It looks like macro but behaves almost like accuracy.

A support desk routes 100 tickets: 80 billing, 15 technical, 5 legal. The router gets billing recall 0.95, technical 0.67 and legal only 0.40.

~~~dg-fig 06.01-multiclass.svg
Per-class F1 for three ticket types and three averages. Macro (0.70) exposes the weak legal class; weighted (0.875) and micro (0.88, equal to accuracy) mostly reflect the big billing class.
~~~

If legal tickets carry the most risk, the weighted 0.875 hides the problem. Print the per-class table next to any average.

## Metrics and thresholds in code

The same synthetic problem as in the figures: 20,000 examples, 3% positive, split into train, validation and test.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, f1_score

X, y = make_classification(n_samples=20_000, n_features=10, n_informative=4,
                           weights=[0.97], class_sep=1.5, flip_y=0, random_state=0)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=0)

model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
p_val = model.predict_proba(X_val)[:, 1]       # scores, not decisions
p_te = model.predict_proba(X_te)[:, 1]

print(confusion_matrix(y_val, p_val >= 0.5).ravel())   # [4843 7 84 66] = TN FP FN TP
print(classification_report(y_val, p_val >= 0.5, digits=3))
# class 1: precision 0.904  recall 0.440  f1 0.592 ; accuracy 0.982

def cost(y, p, t, c_fp=10, c_fn=500):
    tn, fp, fn, tp = confusion_matrix(y, p >= t).ravel()
    return fp * c_fp + fn * c_fn

ts = np.linspace(0.005, 0.6, 120)
best_t = ts[np.argmin([cost(y_val, p_val, t) for t in ts])]   # choose on validation
print(round(best_t, 3))                                         # 0.045
print(cost(y_te, p_te, 0.5), cost(y_te, p_te, best_t))         # 44050 34440  (report on test)
print(confusion_matrix(y_te, p_te >= best_t).ravel())           # [4256 594 57 93]

# scikit-learn can do the sweep with cross-validation for any scorer
def neg_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return -(fp * 10 + fn * 500)
tuned = TunedThresholdClassifierCV(LogisticRegression(max_iter=1000),
                                   scoring=make_scorer(neg_cost), cv=5).fit(X_tr, y_tr)
print(round(tuned.best_threshold_, 3), -neg_cost(y_te, tuned.predict(X_te)))   # 0.074 35020

# multiclass averaging for the ticket example (0 billing, 1 technical, 2 legal)
yt = np.array([0]*80 + [1]*15 + [2]*5)
yp = np.array([0]*76 + [1]*3 + [2]*1 + [0]*4 + [1]*10 + [2]*1 + [0]*3 + [2]*2)
for avg in ["macro", "micro", "weighted"]:
    print(avg, round(f1_score(yt, yp, average=avg), 3))   # macro 0.697, micro 0.88, weighted 0.875
```

At the default threshold the model looks excellent by accuracy (0.982) and precision (0.904) but misses 56% of positives. Moving to 0.045 makes most alerts false (precision about 0.14) yet cuts test cost by 22%. The cross-validated tuner picks 0.074 with a similar cost: the exact optimum is noisy, the region is not.

## Checking the result

- **Is the matrix the right way round?** scikit-learn puts truth in rows, and `.ravel()` returns TN, FP, FN, TP. Swapping them swaps precision and recall.
- **Which class is positive?** Binary metrics default to `pos_label=1`; set it explicitly for other codings.
- **Compare against trivial baselines**: the accuracy of "always the majority class" and the precision of "flag everything" (equal to the prevalence).
- **Is the operating point stable?** A sharp, isolated optimum on a small validation set is probably noise; prefer the middle of a flat region.
- **How uncertain is the number?** With 150 positives, recall of 0.62 has a standard error of about $\sqrt{0.62 \times 0.38/150} \approx 0.04$. Two models differing by 0.02 in recall are not distinguishable.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Reporting accuracy on rare classes | 98% accuracy, users complain nothing is caught | Report recall, precision, MCC or cost; show the majority baseline
| Using `predict()` and its hidden 0.5 | Low recall on rare classes despite good scores | Use `predict_proba` and choose $t$ from costs or constraints
| Tuning the threshold on the test set | Production cost much worse than reported | Tune on validation or with CV; touch test once
| Quoting precision from a rebalanced test set | Precision 0.9 offline, 0.1 in production | State prevalence; recompute precision at the real base rate
| Maximising $F_1$ when costs are known | "Best" model by $F_1$ loses more money | Optimise expected cost directly
| Reporting only a weighted average | Rare class failing unnoticed | Add the per-class table and the macro average

## Going deeper

**The general cost rule.** If correct decisions also have costs (or negative costs, i.e. benefits) $c_{TP}$ and $c_{TN}$, the same expected-cost comparison gives
$$t^* = \frac{c_{FP} - c_{TN}}{(c_{FP} - c_{TN}) + (c_{FN} - c_{TP})}.$$
Only cost *differences* matter. This is Bayes decision theory: with true class probabilities, thresholding at $t^*$ is the best possible rule, which is why calibration matters. If deployment prevalence differs from training prevalence, calibrated probabilities shift too; one correction multiplies the odds $p/(1-p)$ by $\frac{\pi_{new}/(1-\pi_{new})}{\pi_{old}/(1-\pi_{old})}$ before thresholding.

**$F_1$ is not decomposable.** Accuracy and expected cost average a per-example loss; $F_1$ is a ratio of counts over the whole set, so the best decision for one item depends on the others. For calibrated scores, Lipton, Elkan and Narayanaswamy show that the $F_1$-optimal threshold equals half of the best achievable $F_1$. In the sweep figure the best $F_1$ was 0.63; half of that is about 0.32, and the empirical optimum (0.40) sits on the same flat plateau. So a weak model maximises $F_1$ with a low threshold, sometimes flagging nearly everything.

**Why MCC behaves well.** MCC is the Pearson correlation between two 0/1 variables (prediction and label). It is symmetric in the two classes, 0 for any constant predictor, and high only when all four cells are good. $F_1$ ignores TN, so it can be large for a chance-level model on data with many positives.

**Micro equals accuracy, in one line.** In a single-label problem every mistake is a false negative for the true class and a false positive for the predicted class. Summed over classes, total FP equals total FN equals the number of errors, so micro precision, micro recall and micro $F_1$ all reduce to (correct)/(total).

**Thresholded metrics versus scoring rules.** This lesson judges hard decisions. To judge the probabilities themselves, use a **proper scoring rule** such as log loss or the Brier score, minimised only by honest probabilities (module 03). A sensible workflow: a scoring rule or ranking metric chooses the model, a cost or constraint chooses its threshold.

## Key terms

| Term | Meaning
| --- | --- |
| Score | The model's continuous output; higher means more likely positive
| Threshold | The cut-off that turns a score into a yes/no decision
| Confusion matrix | The table of TP, FN, FP, TN (or $K \times K$ counts) for one threshold
| Precision | Of the items flagged, the fraction that are truly positive
| Recall (sensitivity, TPR) | Of the truly positive items, the fraction flagged
| False positive rate | Of the truly negative items, the fraction wrongly flagged
| $F_1$ / $F_\beta$ | Harmonic mean of precision and recall; $\beta$ weights recall more
| MCC | Correlation between predictions and labels using all four cells
| Prevalence | The share of truly positive items; precision depends on it, recall does not
| Macro / micro / weighted | Average per class equally / pool counts / weight by class size

<div class="take">
<div class="lab">Key takeaway</div>

a confusion matrix belongs to a model plus a threshold; choose the threshold from costs or constraints on validation data, and report precision with its base rate.

</div>
