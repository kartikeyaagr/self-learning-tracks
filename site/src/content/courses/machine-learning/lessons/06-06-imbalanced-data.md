---
id: "06.06"
course: "machine-learning"
module: "06"
slug: "imbalanced-data"
title: "Imbalanced data"
keyIdeas: ["Imbalance is mostly a problem of **evaluation and decision**, not of learning: a well-specified model trained on the true class mix ranks well and gives calibrated probabilities. What breaks is accuracy as a metric and 0.5 as a threshold.", "Start with stratified splits, a metric that focuses on the rare class (PR AUC, recall at a fixed precision, expected cost) and a threshold tuned on validation. Often that is all you need.", "Class weights and resampling multiply the model's odds, so predicted probabilities come out too high. Undersampling negatives with keep-rate $\\beta$ can be undone exactly: $p = \\beta p_s / (\\beta p_s - p_s + 1)$.", "SMOTE interpolates synthetic minority points. It can help weak or unregularised models but rarely helps strong learners after threshold tuning, and it must only ever run inside the training fold.", "The real constraint is the **absolute number of positives**: with 200 positives in validation, recall is uncertain by about $\\pm 3$ points. Collecting more positives beats any rebalancing trick."]
question: "A medical team trains a gradient-boosted model on data with 3% positives, oversamples positives with SMOTE to 50%, and reports that cross-validated recall jumped from 0.41 to 0.83 and precision is \"still acceptable\" at 0.6. The oversampling was done on the full dataset before running 5-fold CV. List the problems with this result and describe how you would redo the evaluation."
answer: "There are three separate problems. First, SMOTE before splitting leaks: synthetic positives are interpolations of real positives, so each validation fold contains points nearly identical to training points (and some synthetic training points were built from the validation positives themselves). The model is partly being tested on data it has seen, which inflates both recall and precision; with a flexible model like gradient boosting the inflation can be enormous. Second, the validation folds now contain about 50% positives instead of 3%, so the precision of 0.6 is measured at the wrong prevalence. Precision depends on prevalence: at the real 3% the same false positive rate produces many more false alarms per true positive, so real precision would be far lower. Third, the jump in recall at the default 0.5 threshold is mostly a threshold shift. Rebalancing to 50% multiplies the model's odds by about 30, so scores are inflated and more cases cross 0.5; the original model with a lower, tuned threshold would likely reach a similar recall. To redo it: build an imblearn pipeline so SMOTE runs only on each training fold; use stratified folds, grouped by patient if patients have several rows; evaluate every fold on the untouched, true 3% class mix; report PR AUC alongside the 3% prevalence and recall at the clinically required precision (or expected cost if costs can be agreed); compare against a no-resampling baseline whose threshold is tuned the same way; add confidence intervals, since with few positives per fold differences of a few points are noise; and check calibration (recalibrating if needed) if the probabilities will be shown to clinicians. Finally confirm on a held-out test set that was never used for any choice."
sources: [{"title": "The harm of class imbalance corrections for risk prediction models (van den Goorbergh et al.)", "url": "https://arxiv.org/abs/2202.09101", "note": "Shows resampling and weighting wreck calibration in clinical models without improving AUC."}, {"title": "To SMOTE, or not to SMOTE? (Elor and Averbuch-Elor)", "url": "https://arxiv.org/abs/2201.08528", "note": "Large benchmark: balancing helps weak learners; strong learners plus threshold tuning do as well."}, {"title": "SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al.)", "url": "https://arxiv.org/abs/1106.1813", "note": "The original algorithm and its motivation."}, {"title": "imbalanced-learn: Over-sampling", "url": "https://imbalanced-learn.org/stable/over_sampling.html", "note": "SMOTE variants and how the imblearn Pipeline keeps sampling inside training folds."}]
practice: ["balanced-dataloader", "log-loss-binary-cross-entropy", "sigmoid-focal-loss", "precision-recall-curve-and-average-precision", "softmax-cross-entropy-gradient"]
tags: ["class imbalance", "resampling", "class weights"]
tools: ["scikit-learn", "imbalanced-learn"]
---

## What you will learn

- Why a dataset with 1% positives breaks the metric and the threshold far more often than it breaks the model.
- The order in which to try fixes: stratified splits, a cost-aware metric, a tuned threshold, then (only if needed) class weights or resampling.
- What class weights, undersampling, oversampling and SMOTE actually do to a model's probabilities, and how to undo the distortion.
- The one rule that keeps resampling honest: it happens inside the training fold, never before the split.
- How focal loss and balanced batches handle extreme imbalance in deep learning.
- Why the absolute number of positives, not the ratio, limits what you can learn and measure.

*Before you start:* you should know precision, recall and choosing a threshold by cost (06.01), PR curves and average precision (06.02), and stratified or grouped cross-validation (06.04). The accuracy paradox from 01.05 and probability calibration from module 03 come back here too.

The previous lesson (06.05) tuned hyperparameters against a validation score. This last lesson of the module covers the setting where that score is easiest to get wrong: one class is rare, as in fraud, loan default, disease screening or machine failure. That is the normal case in industry, not an edge case.

## The idea in one picture

A lender has 100,000 past loans, and 988 of them defaulted: a **prevalence** (share of positives) of about 1%. A logistic regression is trained on 60% of the loans, with no tricks at all, and scored on 20,000 held-out loans (197 defaults).

| Rule | Defaults caught (recall) | Accuracy | Cost (missed default = 10, false alarm = 1)
| --- | --- | --- | --- |
| Approve everyone | 0% | 99.0% | 1,970
| Model, flag if $p \ge 0.5$ | 6% | 99.0% | 1,858
| Model, flag if $p \ge 0.07$ | 49% | 97.2% | 1,476

~~~dg-fig 06.06-idea.svg
Predicted default probabilities for 20,000 held-out loans (log count scale). Almost every score is below 0.1, so the usual 0.5 cut-off flags almost nobody; a cut-off at 0.07 catches about half the defaults.
~~~

Read the table from top to bottom. The do-nothing rule already has 99% accuracy, so accuracy cannot tell a useful model from a useless one. The model at the default threshold of 0.5 barely beats doing nothing, which is where people conclude "the model ignores the minority class". But the same model, with the same weights, cuts cost by a quarter once the threshold drops to 0.07. Nothing about learning changed; only the decision did.

That is the main message of this lesson. With **class imbalance** (one class much rarer than the other), three things can go wrong, and only the third is really about the model:

- **The metric.** Accuracy rewards predicting the majority class. Use PR curves, recall at a required precision, or expected cost instead.
- **The threshold.** A well-calibrated model rarely outputs more than 0.5 for an event that happens 1% of the time, so the default threshold predicts almost nothing positive.
- **The amount of signal.** 1,000 positives is a small dataset, however many negatives sit next to them. Flexible models can memorise the few positives, and every metric computed on them is noisy.

A common belief is that the learner itself "ignores" the rare class. In fact a logistic regression or gradient-boosted model trained with log loss on the true class mix estimates $P(y=1 \mid x)$ and ranks examples well. Many "imbalance fixes" are indirect ways of moving the threshold.

~~~dg-flow Try the fixes in this order; stop as soon as validation cost stops improving.
Stratified split -(threshold-free metric)-> Tune threshold -(still poor?)-> Class weights -(still poor?)-> Resampling -(still poor?)-> *More positive data or better features
~~~

## Step 1: split and score so the minority is visible

Before touching the model, make sure the evaluation can see the rare class at all.

**Stratify every split.** A **stratified split** keeps the prevalence the same in every fold. With only 50 positives, a plain random 5-fold split might give one fold 4 and another 16, and scores swing wildly. `StratifiedKFold` (or `StratifiedGroupKFold` when the same customer or patient appears in several rows, 06.04) removes that source of noise.

**Pick a metric that ignores the easy majority.** Good choices for rare positives:

- **PR AUC / average precision** (06.02), always reported together with the prevalence, since a random model's AP equals the prevalence (0.01 here, so 0.23 is 23 times better than chance).
- **Recall at a fixed precision** (or the reverse), when the business states a constraint like "at most 4 false alarms per real fraud".
- **Expected cost**, $\text{cost} = C_{FN} \cdot FN + C_{FP} \cdot FP$, where $C_{FN}$ is the cost of a missed positive and $C_{FP}$ the cost of a false alarm. This is the most honest number when the costs are known.
- **Balanced accuracy** (the average of the recall on each class) and the **Matthews correlation coefficient (MCC)** are reasonable single-number summaries when costs are unknown.

ROC AUC can look excellent at 1% prevalence (0.925 for our loans) while precision stays low, because its false positive rate divides by a huge number of negatives (06.02).

## Step 2: move the threshold, not the data

A classifier outputs a score; a **threshold** $t$ turns it into a decision: flag if $p \ge t$. Lesson 06.01 showed that for a calibrated probability the cost-minimising threshold is

$$t^\star = \frac{C_{FP}}{C_{FP} + C_{FN}}.$$

Here $C_{FP} = 1$ and $C_{FN} = 10$, so $t^\star = 1/11 \approx 0.091$. The reason is simple: flag a loan when the expected cost of approving it, $C_{FN} \cdot p$, exceeds the expected cost of flagging it, $C_{FP} \cdot (1-p)$.

In practice you sweep $t$ on a validation set and pick the lowest cost. The sweep below chose 0.07, close to the theoretical 0.091.

~~~dg-fig 06.06-threshold.svg
Validation cost against threshold. The plain model's best threshold is 0.07; the class-weighted model's is 0.88. Their lowest costs are almost identical: the weights moved the curve sideways, not down.
~~~

The green curve is the same model trained with balanced class weights (Step 3). Its cost curve has slid to the right, with a minimum of the same height. If a rebalancing trick only shifts the curve, tuning the threshold gets the same result for free.

## Step 3: class weights

If threshold tuning is not enough, for example because the model's capacity is being spent on easy negatives, the next lever is the loss itself.

A **class weight** multiplies every example's loss by a number that depends on its class. Weighting positives by $w$ is equivalent, in expectation, to duplicating every positive $w$ times. scikit-learn's `class_weight="balanced"` sets

$$w_k = \frac{n}{K \cdot n_k},$$

where $n$ is the number of training examples, $K$ the number of classes and $n_k$ the number of examples in class $k$. Each class then contributes the same total weight to the loss.

What changes when you do this:

- The model pays more attention to the minority region, which can help when many easy negatives dominate the loss.
- The predicted probabilities now describe a world in which the classes are balanced, so they are systematically too high. On our loans, the average predicted probability jumped from 0.010 (the true default rate) to 0.226.
- Ranking quality (AUC, AP) usually changes very little: 0.229 vs 0.226 here. When ranking does not move, the weight has mostly just moved the threshold.

If anyone downstream uses the probabilities themselves (pricing, risk scores, expected profit), recalibrate on data with the true class mix (module 03), or use the exact correction in the Going deeper section.

<div class="ex">
<div class="lab">Worked example</div>

**what "balanced" does, by hand**

Training set: 10,000 card transactions, 100 frauds, 9,900 legitimate. So $n = 10{,}000$, $K = 2$.

Fraud weight: $w_1 = 10{,}000 / (2 \times 100) = 50$. Legitimate weight: $w_0 = 10{,}000 / (2 \times 9{,}900) = 0.505$.

Total weight per class: $50 \times 100 = 5{,}000$ and $0.505 \times 9{,}900 = 5{,}000$. The two classes now count equally in the loss.

The weight ratio is $50 / 0.505 = 99$. Minimising weighted log loss multiplies the model's odds $p/(1-p)$ by this ratio (Going deeper). Take a transaction whose true fraud probability is 0.0909, which is exactly the cost threshold $1/11$ for $C_{FN} = 10$, $C_{FP} = 1$. Its odds are $0.0909 / 0.9091 = 0.1$.

Weighted model's odds: $0.1 \times 99 = 9.9$, so its output is $9.9 / (1 + 9.9) = 0.908$.

So the right threshold for the weighted model is about 0.91, not 0.09. That is why the green curve in the threshold figure bottoms out near 0.88. Flag with $t = 0.5$ and you flag far too many transactions; tune the threshold and you are back where the plain model started.

</div>

## Step 4: resampling

**Resampling** changes the training data instead of the loss. Three common forms:

**Random undersampling** keeps every positive and a random fraction $\beta$ (the **keep-rate**) of negatives. It is mostly a compute tool: training on 10% of the negatives is about 10 times faster and usually loses little ranking quality when negatives are plentiful. Because dropping negatives at rate $\beta$ multiplies the odds by exactly $1/\beta$, the probabilities can be corrected exactly:

$$p = \frac{\beta\, p_s}{\beta\, p_s - p_s + 1},$$

where $p_s$ is the output of the model trained on the sampled data and $p$ is the corrected probability for the real world. With $\beta = 0.1$, a sampled-model output of 0.5 means a true probability of $0.05 / 0.55 = 0.091$.

**Random oversampling** duplicates positives. It is equivalent to class weighting, and with flexible models it encourages memorising the few positives you have.

**SMOTE** (Synthetic Minority Over-sampling Technique) creates new, synthetic positives instead of copies. For a positive $x_i$, it picks one of its $k$ nearest positive neighbours $x_{nn}$ and places a new point at a random spot on the segment between them:

$$x_{new} = x_i + u\,(x_{nn} - x_i), \qquad u \sim U(0, 1),$$

where $u$ is a random number between 0 and 1 ($u = 0$ gives $x_i$ itself, $u = 1$ gives the neighbour).

~~~dg-fig 06.06-smote.svg
Left: when the minority class forms one compact region, SMOTE's synthetic points (crosses) fill it in. Right: when the minority sits in two separate clumps, the straight segments between them run through the majority, and synthetic "positives" land among real negatives.
~~~

SMOTE fills in the minority region more smoothly than duplication, but it assumes that a straight line between two positives stays positive. That fails for categorical or one-hot features (halfway between "red" and "blue" is not a colour), for multimodal minorities as in the right panel, and when classes overlap. A large benchmark (Elor and Averbuch-Elor, 2022) found that balancing helps weak learners such as unregularised trees, but for strong learners like gradient boosting, threshold tuning alone gives the same or better results. Van den Goorbergh et al. (2022) showed, for clinical risk models, that rebalancing produced badly miscalibrated risks with no gain in discrimination.

## Step 5: resample only inside the training fold

This is the hard rule. If you oversample (or SMOTE) the whole dataset and then split, copies or interpolations of validation positives end up in the training data. The model has effectively seen the answers, and every score is inflated.

The figure shows how large the effect is. A random forest on 6,000 patients with 3.7% positives was evaluated three ways.

~~~dg-fig 06.06-leak.svg
SMOTE applied before 5-fold CV reports near-perfect scores. Doing SMOTE inside each training fold gives honest numbers that match an untouched test set.
~~~

The leaky version reports recall 0.98 and PR AUC 0.99; the honest version reports 0.33 and 0.17, and a held-out test set confirms the honest one (0.27 and 0.14). Note also that the leaky folds contain 50% positives, so even its precision is measured at the wrong prevalence.

The fix is mechanical: put the sampler inside a pipeline, as in `make_pipeline(SMOTE(), model)` passed to `cross_val_score`. `imblearn.pipeline.Pipeline` runs samplers during `fit` only, never during `predict` or scoring, so each training fold is resampled and each validation fold stays untouched.

## Imbalanced data in code

Here is the full comparison on the loan data: a stratified three-way split, four ways of training, and the threshold tuned on validation for each.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

rng = np.random.default_rng(0)                     # 100,000 loans, about 1% default
X = rng.normal(size=(100_000, 6))
logit = -6.6 + X @ np.array([1.4, -1.1, 0.9, 0.7, 0, 0])
y = (rng.random(100_000) < 1 / (1 + np.exp(-logit))).astype(int)

X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0)
print(y_tr.sum(), y_va.sum(), y_te.sum())          # 593 198 197

def cost(p, y, t, c_fn=10, c_fp=1):                # a missed default costs 10x a false alarm
    flag = p >= t
    return c_fn * np.sum(~flag & (y == 1)) + c_fp * np.sum(flag & (y == 0))

grid = np.linspace(0.001, 0.99, 991)
models = {
    "plain":    LogisticRegression(),
    "balanced": LogisticRegression(class_weight="balanced"),
    "smote":    make_pipeline(SMOTE(random_state=0), LogisticRegression()),
    "under":    make_pipeline(RandomUnderSampler(sampling_strategy=0.1, random_state=0),
                              LogisticRegression()),
}
for name, m in models.items():
    m.fit(X_tr, y_tr)                              # samplers run inside fit only
    p_va, p_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
    t = grid[np.argmin([cost(p_va, y_va, g) for g in grid])]   # tune on validation
    print(f"{name:8s} AP={average_precision_score(y_te, p_te):.3f} mean p={p_te.mean():.3f} "
          f"cost@0.5={cost(p_te, y_te, 0.5)} t*={t:.2f} cost@t*={cost(p_te, y_te, t)}")
# plain    AP=0.229 mean p=0.010 cost@0.5=1858 t*=0.07 cost@t*=1476
# balanced AP=0.226 mean p=0.226 cost@0.5=3461 t*=0.88 cost@t*=1499
# smote    AP=0.226 mean p=0.211 cost@0.5=3303 t*=0.90 cost@t*=1488
# under    AP=0.228 mean p=0.059 cost@0.5=1524 t*=0.43 cost@t*=1509

beta = (y_tr.sum() / 0.1) / (y_tr == 0).sum()      # kept 5,930 of 59,407 negatives: 0.0998
p_s = models["under"].predict_proba(X_te)[:, 1]
p = beta * p_s / (beta * p_s - p_s + 1)            # undo the undersampling
print(p_s.mean().round(3), p.mean().round(4))      # 0.059 0.0096
```

`sampling_strategy=0.1` asks for a positive-to-negative ratio of 0.1 after sampling, so 593 positives keep 5,930 negatives. All four models rank equally well (AP 0.226 to 0.229, within noise). At the default threshold they differ a lot. After threshold tuning their costs agree to within about 2%. Only the plain model's probabilities match the real default rate without extra work.

## Reading the result

How to tell whether an imbalance fix actually helped:

- **Compare against a threshold-tuned plain baseline**, never against the plain model at 0.5, or you credit the fix for a threshold shift.
- **Check whether ranking moved.** If AP and AUC are unchanged, the fix did nothing a threshold could not do.
- **Check calibration** whenever probabilities are used. On a reliability plot (module 03), bin the predictions and compare the average prediction to the observed rate in each bin.

~~~dg-fig 06.06-calibration.svg
Reliability plot on log axes. The plain model and the corrected undersampled model sit on the diagonal. Balanced weights and raw undersampling predict probabilities 10 to 100 times higher than the observed default rate.
~~~

- **Put error bars on everything about positives.** Recall measured on $m$ positives has standard error about $\sqrt{r(1-r)/m}$. With $m = 200$ and $r = 0.7$ that is $\sqrt{0.21/200} = 0.032$, so two models 3 points apart are not distinguishable.

~~~dg-fig 06.06-noise.svg
How much measured recall wobbles around a true 70%, depending on how many positives are in the evaluation set. With 200 positives the 95% range is about plus or minus 6 points.
~~~

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Judging by accuracy | 99% accuracy, near-zero recall | PR AUC with prevalence, recall at fixed precision, expected cost
| Keeping the 0.5 threshold | Model "never predicts positive" | Tune the threshold on validation for cost or a precision target
| Resampling before the split | CV scores near 1, test scores far lower | Sampler inside an `imblearn` pipeline; resample training folds only
| Evaluating on rebalanced data | Precision looks fine, collapses in production | Validation and test always keep the true class mix
| Shipping weighted or resampled probabilities | Risk scores 10 to 20 times too high | Recalibrate on true-mix data, or apply the odds correction
| SMOTE on one-hot or categorical columns | Synthetic rows with impossible values | SMOTENC, or class weights instead
| Declaring a winner from a 2-point gap | Result does not replicate | Standard errors or bootstrap; more positives

## Deep learning: losses and batches

A one-stage object detector scores about 100,000 candidate boxes per image, nearly all easy background, whose many small losses swamp the few objects. **Focal loss** fixes this by multiplying cross-entropy by $(1 - p_t)^\gamma$:

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log p_t,$$

where $p_t$ is the probability the model assigns to the true class and $\gamma \ge 0$ controls how strongly easy examples are discounted ($\gamma = 0$ is plain cross-entropy). With $\gamma = 2$, an example already classified at $p_t = 0.9$ is multiplied by $0.1^2 = 0.01$, a 100-fold down-weighting, while a hard example at $p_t = 0.5$ keeps a factor of 0.25. Training concentrates on the hard cases.

For classification with mini-batches, **class-balanced batch sampling** serves a similar purpose: a sampler draws minority examples more often, so every batch contains some and their gradient is not drowned out. Both tricks distort probabilities in the same way as class weights, so recalibrate afterwards if you need probabilities.

## Going deeper

**Why weights multiply the odds.** At a fixed input $x$ with true probability $q = P(y=1 \mid x)$, a model outputting $p$ pays expected weighted log loss $-w_1 q \log p - w_0 (1-q)\log(1-p)$. Setting the derivative to zero gives

$$\frac{p}{1-p} = \frac{w_1}{w_0} \cdot \frac{q}{1-q}.$$

So a flexible enough model learns the true odds multiplied by the weight ratio. To recover $q$, divide the odds by $w_1/w_0$. Undersampling negatives at keep-rate $\beta$ has the same effect with ratio $1/\beta$, which is where the correction formula comes from. Duplicating positives $c$ times gives ratio $c$. The same algebra explains why the tuned threshold moved from about 0.09 (odds 0.1) to about 0.9 (odds 0.1 times roughly 100) in the code above.

**Why it is only approximate for real models.** The derivation assumes the model can represent any function of $x$. For a misspecified model (a linear boundary on curved data), weights also change *where* the boundary sits, trading accuracy in the majority region for the minority region. That is when weights or resampling can genuinely improve ranking, so measure rather than assume.

**The real constraint: positives.** Every estimate about the minority class is driven by $m$, the number of positives. With 200 validation positives, recall is uncertain by about $\pm 3$ points (one standard error), and model selection over many configurations (06.05) on so few positives overfits the validation set. Labelling more positives, active learning, or better features almost always beat any rebalancing trick.

## Key terms

| Term | Meaning
| --- | --- |
| Class imbalance | One class is much rarer than the other(s) in the data
| Prevalence | Share of examples that are positive; the AP of a random model
| Stratified split | A split that keeps the class proportions equal in every fold
| Class weight | A per-class multiplier on the loss; "balanced" gives each class equal total weight
| Undersampling / keep-rate $\beta$ | Dropping a fraction of majority examples; $\beta$ is the fraction kept
| SMOTE | Making synthetic minority points by interpolating between minority neighbours
| Focal loss | Cross-entropy scaled by $(1-p_t)^\gamma$ so easy examples count less
| Expected cost | $C_{FN} \cdot FN + C_{FP} \cdot FP$, the metric to minimise when costs are known

<div class="take">
<div class="lab">Key takeaway</div>

fix the metric and the threshold first; use class weights or resampling only when they measurably help on validation, recalibrate probabilities afterwards, resample inside training folds only, and remember that more positive examples beat any rebalancing.

</div>
