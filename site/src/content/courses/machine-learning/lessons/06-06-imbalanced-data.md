---
id: "06.06"
course: "machine-learning"
module: "06"
slug: "imbalanced-data"
title: "Imbalanced data"
keyIdeas: ["Imbalance is mostly a problem of **evaluation and decision**, not of learning: a well-specified probabilistic model trained on the true class mix produces good rankings and calibrated probabilities. What breaks is accuracy as a metric and 0.5 as a threshold.", "Start with stratified splits, a threshold-free metric (PR AUC, recall at fixed precision, expected cost) and a tuned threshold. Often that is all you need.", "Class weights and resampling change the effective prior, so predicted probabilities become inflated. Undersampling negatives with keep-rate $\\beta$ can be undone exactly: $p = \\beta p_s / (\\beta p_s - p_s + 1)$.", "SMOTE interpolates synthetic minority points; it can help weak or unregularised models but rarely helps strong learners after threshold tuning, and it must only ever run inside the training fold.", "The real constraint is the **absolute number of positives**: with 200 positives in validation, recall is uncertain by about $\\pm 3$ points. Collecting more positives beats any rebalancing trick."]
question: "A medical team trains a gradient-boosted model on data with 3% positives, oversamples positives with SMOTE to 50%, and reports that cross-validated recall jumped from 0.41 to 0.83 and precision is \"still acceptable\" at 0.6. The oversampling was done on the full dataset before running 5-fold CV. List the problems with this result and describe how you would redo the evaluation."
answer: "First, SMOTE before splitting leaks: synthetic positives are interpolations of real positives, so validation folds contain points nearly identical to training points (and possibly derived from validation positives themselves), inflating recall and precision. Second, the validation folds now have about 50% positives, so the precision of 0.6 is measured at the wrong prevalence; at the real 3% it would be far lower (precision depends on prevalence). Third, the jump in recall at the default 0.5 threshold is mostly a threshold shift: rebalancing inflates scores, which a tuned threshold on the original model could reproduce. Redo it with a pipeline that resamples only inside each training fold, stratified (and grouped by patient) folds evaluated on the true class mix, a threshold-free metric (PR AUC) plus recall at the clinically required precision, a threshold-tuned no-resampling baseline for comparison, and a calibration check if the probabilities will be shown to clinicians."
sources: [{"title": "The harm of class imbalance corrections for risk prediction models (van den Goorbergh et al.)", "url": "https://arxiv.org/abs/2202.09101", "note": "Shows resampling and weighting wreck calibration in clinical models without improving AUC."}, {"title": "To SMOTE, or not to SMOTE? (Elor and Averbuch-Elor)", "url": "https://arxiv.org/abs/2201.08528", "note": "Large benchmark: balancing helps weak learners; strong learners plus threshold tuning do as well."}, {"title": "SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al.)", "url": "https://arxiv.org/abs/1106.1813", "note": "The original algorithm and its motivation."}, {"title": "imbalanced-learn: Over-sampling", "url": "https://imbalanced-learn.org/stable/over_sampling.html", "note": "SMOTE variants and how the imblearn Pipeline keeps sampling inside training folds."}]
practice: ["balanced-dataloader", "log-loss-binary-cross-entropy", "sigmoid-focal-loss", "precision-recall-curve-and-average-precision", "softmax-cross-entropy-gradient"]
tags: ["class imbalance", "resampling", "class weights"]
tools: ["scikit-learn", "imbalanced-learn"]
---

## What actually goes wrong

A dataset with 1% positives breaks three things, and only one of them is the model.

- **The metric**: accuracy rewards predicting the majority class, as lessons 01.05 and 06.01 showed. Use PR curves, recall at a fixed precision, or expected cost.
- **The threshold**: a calibrated model rarely outputs probabilities above 0.5 for a rare event, so the default threshold predicts almost nothing positive. Tune the threshold (06.01, 06.02).
- **The amount of signal**: 1,000 positives is a small dataset, however many negatives you have. Models with many parameters can overfit the minority class, and every metric about it is noisy.

The first two have cheap fixes. A common misconception is that the learner itself "ignores" the minority class; a logistic regression or gradient-boosted model trained with log loss on the true class mix estimates $P(y=1 \mid x)$ fine, with good ranking. Many "imbalance fixes" in practice are just indirect ways of moving the threshold.

~~~dg-flow Try the fixes in this order; stop when the cost metric stops improving on validation.
Stratified split -(threshold-free metric)-> Tune threshold -(still poor?)-> Class weights -(still poor?)-> Resampling -(still poor?)-> *More positive data or better features
~~~

## Class weights

Multiplying the loss of each positive by $w$ is equivalent, in expectation, to duplicating positives $w$ times. scikit-learn's `class_weight="balanced"` sets each class's weight to $n / (K \cdot n_k)$, so with 1% positives the positive weight is 50 and the negative weight about 0.5. Effects:

- The model focuses more capacity on the minority region, which can help models whose loss is dominated by easy negatives.
- The predicted probabilities now correspond to a world where classes are balanced: they are systematically too high. If anyone consumes the probabilities (pricing, risk scores, expected-value calculations), recalibrate on data with the true class mix (module 03).
- Ranking (AUC, AP) often changes little, which tells you the weight mainly moved the threshold.

## Resampling

**Random undersampling** keeps every positive and a fraction $\beta$ of negatives. It is mostly a compute tool: training on 10% of the negatives is 10 times faster and usually loses little ranking quality when negatives are plentiful. Because sampling negatives at rate $\beta$ multiplies the odds by $1/\beta$, the probabilities can be corrected exactly:

$$p = \frac{\beta\, p_s}{\beta\, p_s - p_s + 1}$$

where $p_s$ is the model's output trained on the sampled data. With $\beta = 0.1$, a sampled-model output of 0.5 means a true probability of $0.05 / 0.55 = 0.091$.

**Random oversampling** duplicates positives, which is equivalent to weighting and encourages memorising them. **SMOTE** creates synthetic positives by interpolating between a positive and one of its $k$ nearest positive neighbours: $x_{new} = x_i + u\,(x_{nn} - x_i)$ with $u \sim U(0, 1)$. It fills in the minority region more smoothly but assumes straight lines between positives stay positive, which fails for categorical features and for overlapping classes. Benchmarks (Elor and Averbuch-Elor, 2022) found it helps weak learners such as unregularised trees, but for strong learners like gradient boosting, threshold tuning alone gives the same or better results. Van den Goorbergh et al. (2022) showed, for clinical risk models, that rebalancing produced badly miscalibrated risks with no gain in discrimination.

Hard rule: resample **only the training fold**. Oversampling before splitting puts copies or interpolations of validation positives into training, which inflates every score. `imblearn.pipeline.Pipeline` applies samplers during `fit` only, which makes this automatic inside CV.

```python
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(scaler, SMOTE(k_neighbors=5, random_state=0), LogisticRegression(max_iter=1000))
# cross_val_score(pipe, X, y, cv=StratifiedKFold(5), scoring="average_precision") resamples inside each fold
```

## Loss-level fixes in deep learning

In dense object detection, a one-stage detector sees about 100,000 candidate boxes per image, nearly all easy background. **Focal loss** multiplies cross-entropy by $(1 - p_t)^\gamma$, where $p_t$ is the probability assigned to the true class, so confidently correct examples contribute almost nothing and training concentrates on the hard ones. With $\gamma = 2$, an example classified at $p_t = 0.9$ is down-weighted 100 times. For classification with mini-batches, class-balanced batch sampling serves a similar purpose: every batch contains minority examples, so their gradient is not drowned out. Both distort probabilities in the same way as class weights.

## Metrics and splits for rare classes

- Use stratified (and, where needed, grouped) splits so each fold has enough positives.
- Report PR AUC with the prevalence, recall at the precision the business needs, and expected cost; MCC and balanced accuracy are reasonable single-number summaries.
- Compute uncertainty. Recall on $m$ positives has standard error about $\sqrt{r(1-r)/m}$; with $m = 200$ and $r = 0.7$ that is 0.032, so two models 3 points apart are not distinguishable.

<div class="ex">
<div class="lab">Worked example</div>

**credit default at 1%**

100,000 loans, 1,000 defaults; stratified split leaves 200 defaults in validation. Three logistic regressions with the threshold tuned on validation for minimum cost:

Plain (true class mix): AP 0.42, mean predicted probability 0.010 (matches the base rate). Balanced class weights: AP 0.42, mean predicted probability 0.19, so on average the probabilities are 19 times too high. SMOTE inside the training fold: AP 0.41.

After threshold tuning all three have nearly the same cost, within the $\pm 3$-point noise on recall. The team ships the plain model because its probabilities feed the interest-rate calculation, and spends the effort on two new features and on labelling six more months of loans.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

fix the metric and the threshold first; use class weights or resampling only when they measurably help on validation, recalibrate probabilities afterwards, resample inside training folds only, and remember that more positive examples beat any rebalancing.

</div>
