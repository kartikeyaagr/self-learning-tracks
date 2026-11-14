---
id: "03.04"
course: "machine-learning"
module: "03"
slug: "calibrated-probabilities"
title: "Calibrated probabilities"
keyIdeas: ["A classifier is **calibrated** if, among cases where it predicts $p$, the event happens a fraction $p$ of the time. This is separate from discrimination: AUC measures ranking only, and any order-preserving transform of the scores leaves AUC unchanged while changing every probability.", "Check calibration with a **reliability diagram** (observed rate against mean prediction per bin) and summarise it with ECE, always alongside a proper scoring rule such as log loss or the Brier score, which cannot be gamed by constant predictions.", "**Platt scaling** fits $\\sigma(a s + b)$ to held-out scores: two parameters, little data needed, AUC unchanged, but only sigmoid-shaped corrections. **Temperature scaling** divides all logits by one fitted $T$ and is its multiclass cousin.", "**Isotonic regression** fits any non-decreasing step function with pool-adjacent-violators: more flexible, but it needs roughly a thousand or more calibration points and overfits on small sets.", "Always calibrate on data the model was not trained on, and correct or recalibrate after resampling, class weighting or a change in the base rate (a shift of the log-odds by the change in prior log-odds)."]
question: "Your gradient-boosted model has validation AUC 0.91 but a reliability diagram shows it is overconfident at both ends (ECE 0.12). You have 300 labelled examples not used in training. Do you choose Platt scaling or isotonic regression, what happens to AUC and log loss after calibration, and what mistake would make the calibration itself unreliable? Finally, the model was trained on a sample with 50% positives while the live rate is 5%: what else must you do?"
answer: "Choose Platt scaling. With 300 examples, isotonic regression's step function has too many free values: each step would average only a handful of labels and the map would follow noise. Platt's two parameters are well determined from 300 points, and a symmetric overconfidence pattern is roughly sigmoid-shaped in log-odds, which $\\sigma(a s + b)$ with $a < 1$ corrects by cooling the scores. Platt's map is strictly increasing, so the ranking is unchanged and AUC stays exactly 0.91. Log loss and Brier score should fall, and ECE should drop substantially (though with 300 points any ECE estimate is noisy, so compare with a bootstrap or cross-validation). Isotonic could also shift AUC slightly through ties. The mistake to avoid is fitting the calibrator on the training data, or on scores from a model that has seen those rows. Training-set scores are optimistically confident, so the calibrator would learn too small a correction and the deployed model would stay overconfident. Fit on the 300 untouched examples, or use cross-validated calibration (`CalibratedClassifierCV`), and evaluate on yet another split. For the base rate: if the 300 calibration examples were drawn from the same 50/50 sample, the calibrated probabilities are still honest only for a 50% world. Shift the log-odds by $\\ln(0.05/0.95) - \\ln(0.5/0.5) \\approx -2.94$ (so a calibrated 0.5 becomes about 0.05), or better, draw the calibration set from live data with the true 5% rate so the calibrator learns the shift itself. Then monitor calibration after launch, because the rate can drift again."
sources: [{"title": "Predicting Good Probabilities With Supervised Learning", "url": "https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf", "note": "Niculescu-Mizil and Caruana (ICML 2005): how each model family miscalibrates, and Platt vs isotonic by data size."}, {"title": "On Calibration of Modern Neural Networks", "url": "https://arxiv.org/abs/1706.04599", "note": "Guo et al. (2017): overconfidence in deep nets, reliability diagrams, ECE and temperature scaling."}, {"title": "scikit-learn: Probability calibration", "url": "https://scikit-learn.org/stable/modules/calibration.html", "note": "Calibration curves, CalibratedClassifierCV and the sigmoid vs isotonic choice."}, {"title": "scikit-learn: Isotonic regression", "url": "https://scikit-learn.org/stable/modules/isotonic.html", "note": "The isotonic model and the pool-adjacent-violators fit behind it."}]
practice: ["expected-calibration-error", "log-loss-binary-cross-entropy", "logistic-regression", "auc-roc"]
tags: ["calibration", "reliability diagram"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- What a **calibrated** probability is, and why it differs from ranking cases well.
- How to read a **reliability diagram** and summarise it with ECE, Brier score and log loss.
- How the course's classifiers typically miscalibrate, and why.
- Two fixes, **Platt scaling** (and temperature scaling) and **isotonic regression**, and how to choose.
- How to calibrate without leakage, and correct for resampling or a changed base rate.

*Before you start:* logistic regression and log loss (02.03), softmax (02.04), and Bayes' rule as used in naive Bayes (03.01) and discriminant analysis (03.03). For AUC (module 06) it is enough to know it measures how well a model *ranks* positives above negatives.

Lesson 03.03 turned class densities into posterior probabilities with Bayes' rule. Every classifier in this course hands you a number between 0 and 1 and calls it a probability. This last lesson of the module asks: should you believe that number?

## The idea in one picture

A spam filter scores 1,000 emails. You group them by the probability the filter gave, then count how many in each group really were spam:

| Bin of predicted probability | 0 to 0.2 | 0.2 to 0.4 | 0.4 to 0.6 | 0.6 to 0.8 | 0.8 to 1.0
| --- | --- | --- | --- | --- | --- |
| Emails in the bin | 300 | 100 | 100 | 100 | 400
| Average predicted probability | 0.05 | 0.30 | 0.50 | 0.70 | 0.95
| Fraction actually spam | 0.12 | 0.33 | 0.50 | 0.62 | 0.80

In the middle the filter is honest. At the ends it is not: when it says 0.95, only 80% are spam, and when it says 0.05, 12% are.

~~~dg-fig 03.04-spam.svg
The table as a reliability diagram. Points on the diagonal are honest; the gaps at both ends show overconfidence. The bars show how many emails each gap affects.
~~~

A model is **calibrated** when stated probabilities match observed frequencies: of the cases where it says 0.8, about 80% are positive. The plot is a **reliability diagram**. This filter is **overconfident** (predictions more extreme than reality); predictions bunched too close to the middle are **underconfident**.

Why care, if spam still scores above non-spam? Because many decisions use the probability as a number: an expected cost ($p \times$ cost of spam versus $(1-p) \times$ cost of losing a real email), an ad's value $p_{\text{click}} \times \text{bid}$, a threshold reused after retraining, or averaging several models. All of them assume 0.95 really means 95%.

## Step 1: calibration is not the same as ranking

In symbols, a model producing probabilities $\hat p(X)$ is calibrated if

$$\Pr\big(Y = 1 \mid \hat p(X) = p\big) = p \quad \text{for every value } p.$$

Read it as: "among all cases where the model said $p$, a fraction $p$ are positive." $Y$ is the true label, $X$ the input.

The other quality is **discrimination**: positives should score higher than negatives. AUC measures only this, and depends only on the *order* of scores, so any **monotone transform** (one that never reverses the order, such as a sigmoid) leaves AUC unchanged while changing every probability.

~~~dg-fig 03.04-same-auc.svg
One set of scores pushed through three order-preserving maps. Ranking is identical, so AUC is 0.82 for all three, but only the calibrated version sits on the diagonal and has the best log loss.
~~~

Two extremes make the difference concrete. Predicting the base rate (say 0.3) for every email is perfectly calibrated and useless (AUC 0.5). Reporting the *square* of the true probability ranks exactly as well as the truth, yet a true 0.5 is reported as 0.25. You want both qualities, and calibration can usually be repaired after training without touching the ranking.

## Step 2: measure it

**Build a reliability diagram.**

1. Sort held-out predictions into bins: ten equal-width bins by default, or **equal-count bins** when predictions cluster near 0 or 1.
2. In each bin compute the mean prediction $\bar p_b$ and the observed positive rate $\bar y_b$.
3. Plot $\bar y_b$ against $\bar p_b$ with the diagonal, and the bin counts underneath.

A curve flatter than the diagonal (below it on the right, above on the left) means overconfident; a steeper S-shape means underconfident.

**Summarise the gaps with ECE.** The **expected calibration error** is the average gap, weighted by how many predictions fall in each bin:

$$\mathrm{ECE} = \sum_{b} \frac{n_b}{n}\,\big|\bar y_b - \bar p_b\big|,$$

where $n_b$ is the number of predictions in bin $b$ and $n$ is the total.

<div class="ex">
<div class="lab">Worked example</div>

**ECE for the spam filter**

Weights $n_b/n$: $(0.3, 0.1, 0.1, 0.1, 0.4)$. Gaps $|\bar y_b - \bar p_b|$: $(0.07, 0.03, 0, 0.08, 0.15)$.

Contributions: $0.3 \times 0.07 = 0.021$, $0.1 \times 0.03 = 0.003$, $0$, $0.1 \times 0.08 = 0.008$, $0.4 \times 0.15 = 0.060$.

Total: $\mathrm{ECE} = 0.021 + 0.003 + 0 + 0.008 + 0.060 = 0.092$: about 9 percentage points off on average, two thirds of it from the big top bin.

</div>

ECE depends on the binning and can be gamed: the constant base-rate model scores near zero. So report it next to a **proper scoring rule**, a score whose expected value is best when you report the true probability:

$$\text{Brier} = \frac{1}{n}\sum_i (\hat p_i - y_i)^2, \qquad \text{log loss} = -\frac{1}{n}\sum_i \big[y_i \ln \hat p_i + (1-y_i)\ln(1-\hat p_i)\big].$$

Brier is the squared error against the 0/1 label; log loss is the cross-entropy of 02.03. Both reward calibration *and* discrimination.

<div class="ex">
<div class="lab">Worked example</div>

**Brier score and log loss on four emails**

Predictions $(0.9, 0.8, 0.3, 0.1)$, labels $(1, 0, 0, 0)$. The second email is a confident mistake.

Brier: squared errors $(0.01, 0.64, 0.09, 0.01)$, mean $0.75/4 = 0.1875$.

Log loss: $-\ln 0.9 = 0.105$, $-\ln 0.2 = 1.609$, $-\ln 0.7 = 0.357$, $-\ln 0.9 = 0.105$; mean $2.177/4 = 0.544$.

The one confident mistake supplies 85% of the Brier score and 74% of the log loss: proper scores punish overconfidence hardest.

</div>

## Why models miscalibrate

Each model family distorts probabilities in its own way, because of what it optimises.

~~~dg-fig 03.04-families.svg
Four classifiers on the same data (with redundant features). Logistic regression hugs the diagonal; naive Bayes is far too sure; the random forest is S-shaped (underconfident); a rescaled SVM margin is not a probability.
~~~

- **Naive Bayes** pushes predictions toward 0 and 1: correlated features are counted as independent evidence, inflating the log-odds (03.01).
- **SVMs** output a margin (02.05); rescaling it into $[0, 1]$ keeps the order but has no probabilistic meaning.
- **Random forests** rarely predict near 0 or 1, because averaging trees that disagree pulls votes toward the middle.
- **Boosted trees** often show a sigmoid-shaped distortion, pushed away from 0 and 1.
- **Logistic regression** is usually nearly calibrated when well specified and lightly regularised, since it minimises log loss.
- **Deep networks** are often overconfident; Guo et al. (2017) found one temperature parameter repairs much of it.

The fix is always the same idea: learn a small **calibration map** from score to probability, on data the model did not see.

## Step 3: Platt scaling

Treat the model's score as a single feature and fit a tiny logistic regression on a held-out **calibration set**:

$$\hat p = \sigma(a\,s + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}},$$

where $s$ is the raw score (an SVM margin, or log-odds $\ln\frac{p}{1-p}$) and $a, b$ minimise log loss on the calibration set. Slope $a < 1$ softens overconfident scores, $a > 1$ sharpens underconfident ones; $b$ shifts everything.

Two parameters need little data (a few hundred examples) and rarely overfit, and with $a > 0$ the map is increasing, so AUC never changes. The limitation is shape: only sigmoid curves. Platt's paper also replaces the 0/1 targets with $(N_+ + 1)/(N_+ + 2)$ and $1/(N_- + 2)$ ($N_\pm$ = class counts) to avoid extreme fits on small sets.

<div class="ex">
<div class="lab">Worked example</div>

**applying a fitted Platt map**

On the naive Bayes model in the figures, Platt on log-odds learned $a = 0.29$, $b = 0.24$: it divides the log-odds by about 3.4.

A raw prediction of 0.99 has log-odds $\ln(0.99/0.01) = 4.595$. Then $a s + b = 0.29 \times 4.595 + 0.24 = 1.573$, and $\sigma(1.573) = 0.83$.

A raw 0.10 has log-odds $-2.197$, so $a s + b = -0.637 + 0.24 = -0.397$, and $\sigma(-0.397) = 0.40$.

Both moves go where naive Bayes's reliability diagram says they should: its "99%" cases are positive far less often than 99%, its "10%" cases far more often than 10%.

</div>

## Step 4: temperature scaling for many classes

With $K$ classes and logits $z_1, \dots, z_K$, **temperature scaling** divides every logit by one number $T > 0$ before the softmax:

$$\hat p_k = \frac{e^{z_k / T}}{\sum_j e^{z_j / T}}.$$

$T$ is fitted on validation log loss. $T > 1$ flattens overconfident outputs; $T < 1$ sharpens. Dividing by a positive constant never changes the largest logit, so accuracy is untouched. It is Platt scaling with $a = 1/T$, $b = 0$, shared across classes.

<div class="ex">
<div class="lab">Worked example</div>

**one temperature, three classes**

Logits $(4, 2, 0)$. At $T = 1$: $e^4 = 54.60$, $e^2 = 7.39$, $e^0 = 1$, total $62.99$, so probabilities $(0.867, 0.117, 0.016)$.

At $T = 2$ the logits become $(2, 1, 0)$: $7.39, 2.72, 1$, total $11.11$, so probabilities $(0.665, 0.245, 0.090)$.

The predicted class is still the first one; only its confidence dropped from 87% to 67%.

</div>

## Step 5: isotonic regression

If the distortion is not sigmoid-shaped, assume no shape: fit the best **non-decreasing step function** from score to probability. Sort calibration examples by score and walk through their labels, keeping blocks with their mean label. Whenever a block's mean is lower than the previous one (a violation), merge them into one block with the weighted mean, and repeat. This is **pool-adjacent-violators** (PAV), linear time after sorting:

```python
def pav(y_sorted):                     # labels ordered by increasing score
    blocks = []                        # each block: [mean, weight]
    for v in y_sorted:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:   # violation: merge
            (m2, w2), (m1, w1) = blocks.pop(), blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    return [m for m, w in blocks for _ in range(w)]

```

<div class="ex">
<div class="lab">Worked example</div>

**PAV by hand**

Labels sorted by score: $(0, 1, 0, 0, 1, 1)$.

Read 0, then 1: fine so far. Read the next 0: the block "1" is above it, a violation, so pool them into one block of mean $(1 + 0)/2 = 0.5$ and weight 2.

Read the next 0: the block 0.5 is above it, pool again: mean $(0.5 \times 2 + 0)/3 = 1/3$, weight 3. The block before it is 0, which is not above $1/3$, so stop.

Read 1, then 1: no violations. Result: $(0, \tfrac13, \tfrac13, \tfrac13, 1, 1)$. A new score landing in the middle block gets probability $1/3$.

</div>

~~~dg-fig 03.04-maps.svg
Calibrating naive Bayes. Left: Platt's sigmoid cannot follow the steep rise and plateau in the observed rates; the isotonic steps can. Right: on a fresh split, isotonic sits on the diagonal with the lower log loss.
~~~

The price of flexibility: one free value per step, and with few points each step averages a handful of labels. Niculescu-Mizil and Caruana (2005) found Platt better below roughly a thousand calibration points and isotonic better above; the crossover depends on how far from a sigmoid the distortion is.

~~~dg-fig 03.04-datasize.svg
Simulated non-sigmoid distortion. With 50 to 200 examples isotonic overfits; Platt levels off early because its shape is wrong. Here the curves cross at a few hundred examples.
~~~

Its flat steps create ties, which usually lower AUC slightly (or raise it, where the raw order was locally wrong).

## Step 6: calibrate without leakage

~~~dg-flow Calibration is a second model fitted on data the first model never saw, then applied to its scores at serving time.
Training split -(fit)-> Classifier -(score)-> Calibration split -(fit Platt or isotonic)-> *Calibrator -(apply at serving)-> Probabilities
~~~

- **Hold-out data.** Scores on the model's own training data are overconfident, so a calibrator fitted there learns too small a correction. Use a separate split, or `CalibratedClassifierCV(method="sigmoid" or "isotonic", cv=5)`, which fits on out-of-fold predictions.
- **Resampling breaks calibration.** Undersampling, oversampling or class weights move probabilities toward the minority class. By Bayes' rule a changed prior adds a constant to the log-odds, so shift by $\ln\frac{\pi}{1-\pi} - \ln\frac{\pi_s}{1-\pi_s}$ ($\pi$ = true base rate, $\pi_s$ = training rate), or recalibrate on data with the true rate.
- **Base rates move.** If prevalence changes after deployment (**prior shift**), apply the same shift with the new rate, or recalibrate on recent labels.

~~~dg-fig 03.04-prior.svg
A model trained on a 50/50 sample, corrected for the real base rate by shifting its log-odds. With 1% fraud, a raw 0.5 means 1% and even a raw 0.9 means only about 8%.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**undoing balanced sampling**

A fraud model is trained on a 1:1 sample; the true rate is 1%. Shift: $\ln(0.01/0.99) - \ln(0.5/0.5) = -4.60$.

Raw 0.5 (logit 0) becomes $\sigma(-4.60) = 0.010$. Raw 0.9 (logit $\ln 9 = 2.20$) becomes $\sigma(-2.40) = 0.083$.

Uncorrected, the cost model would overstate fraud risk by a factor of about 10 to 50.

</div>

## Calibration in code

Naive Bayes on data with redundant (correlated) features, calibrated both ways with cross-validation and scored on a test split:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

X, y = make_classification(n_samples=20000, n_features=20, n_informative=4,
                           n_redundant=8, random_state=0)       # redundant = correlated features
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)

def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)           # equal-width bin of each prediction
    return sum(np.mean(idx == b) * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if np.any(idx == b))

models = {
    "raw NB":   GaussianNB().fit(X_tr, y_tr),
    "sigmoid":  CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5, ensemble=False).fit(X_tr, y_tr),
    "isotonic": CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5, ensemble=False).fit(X_tr, y_tr),
}
for name, m in models.items():
    p = m.predict_proba(X_te)[:, 1]
    print(f"{name:9s} AUC {roc_auc_score(y_te, p):.3f}  log loss {log_loss(y_te, p):.3f}  "
          f"Brier {brier_score_loss(y_te, p):.3f}  ECE {ece(y_te, p):.3f}")
# raw NB    AUC 0.786  log loss 0.912  Brier 0.207  ECE 0.174
# sigmoid   AUC 0.786  log loss 0.539  Brier 0.179  ECE 0.083
# isotonic  AUC 0.802  log loss 0.484  Brier 0.161  ECE 0.012

```

Raw naive Bayes ranks reasonably but its probabilities are poor (`calibration_curve` shows that when it says 0.03, 29% are positive). The sigmoid map keeps AUC exactly and cuts log loss from 0.91 to 0.54, yet ECE stays 0.08: this distortion is not sigmoid-shaped. Isotonic, with 10,000 out-of-fold points, nearly removes the error (ECE 0.01) and wins on log loss and Brier; its AUC even rose, because pooling flattened regions where the raw order was noisy.

`ensemble=False` fits one classifier plus one calibrator on its out-of-fold scores (the default averages five calibrated copies). scikit-learn's sigmoid is applied to `decision_function` if the model has one, else to `predict_proba`; for saturating models like naive Bayes, Platt on log-odds (as in the figures) behaves better.

## Reading and checking the result

Check on a split neither the model nor the calibrator saw:

1. **Reliability diagram on fresh data**, with bin counts. Ignore wiggles in tiny bins; look for systematic tilt.
2. **Log loss or Brier should fall.** If they rise, the calibrator overfitted or the calibration data differ from the test data.
3. **AUC should not move** for Platt or temperature scaling, and only slightly for isotonic.
4. **Check subgroups.** Calibrated overall can still mean overconfident for one region or customer segment.
5. **Mind the noise.** With 300 test points, ECE 0.05 versus 0.08 is not a real difference; bootstrap it.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Calibrating on the training data | calibrated model still overconfident on new data | use a held-out split or `CalibratedClassifierCV`
| Isotonic on a few hundred points | jagged steps, test log loss worse than Platt | use Platt below roughly a thousand points
| Platt on a non-sigmoid distortion | log loss improves but reliability curve still bent | isotonic with more data, or Platt on log-odds
| Trained on a rebalanced sample | probabilities far above the real base rate | logit shift by the prior ratio, or recalibrate
| Base rate drifts after launch | predictions systematically too high or too low over time | monitor calibration (module 09) and recalibrate
| Judging by ECE alone | constant or lazy predictions look "perfect" | report log loss or Brier as well

## Going deeper

**Why proper scores reward honesty.** If the true probability is $p$ and you report $q$, the expected Brier score is $p(1-q)^2 + (1-p)q^2$, with derivative $2(q - p)$: zero only at $q = p$. Shading the truth never helps. Log loss behaves the same way.

**The Brier decomposition.** Group predictions into bins as in the reliability diagram, and let $\bar y$ be the overall positive rate. Then, up to a small within-bin term,

$$\text{Brier} = \underbrace{\sum_b \tfrac{n_b}{n}(\bar p_b - \bar y_b)^2}_{\text{reliability (lower is better)}} - \underbrace{\sum_b \tfrac{n_b}{n}(\bar y_b - \bar y)^2}_{\text{resolution (higher is better)}} + \underbrace{\bar y(1 - \bar y)}_{\text{uncertainty}}.$$

Reliability is a squared ECE; resolution is discrimination; uncertainty is fixed by the data. Calibration shrinks the first term; only a better model raises the second.

**Why logistic regression is nearly calibrated.** At the unregularised log-loss optimum the intercept's gradient gives $\sum_i (\hat p_i - y_i) = 0$ (average prediction equals observed rate), and each feature's gives $\sum_i (\hat p_i - y_i)x_{ij} = 0$. These balance conditions are calibration "in the large" and within groups the features describe. Strong regularisation, misspecification or class weights break them.

**The prior-shift formula from Bayes' rule.** Posterior log-odds equal the log likelihood ratio plus the prior log-odds: $\ln\frac{p(1 \mid x)}{p(0 \mid x)} = \ln\frac{p(x \mid 1)}{p(x \mid 0)} + \ln\frac{\pi}{1-\pi}$. If only the prior changes, the likelihood ratio stays and the log-odds move by the change in prior log-odds: the Step 6 correction, and the same algebra naive Bayes and discriminant analysis use.

**Multiclass calibration has several meanings.** *Confidence calibration* asks only that the top-class probability be honest (what most reported ECE numbers measure); *class-wise* asks it per class; *full* calibration asks it of the whole probability vector, which is far harder to check. Say which one you measured.

## Key terms

| Term | Meaning
| --- | --- |
| Calibration | Of cases given probability $p$, a fraction $p$ are positive
| Discrimination | How well scores rank positives above negatives (AUC)
| Reliability diagram | Observed rate against mean prediction, per bin
| ECE | Bin-size-weighted average gap between predicted and observed rates
| Proper scoring rule | A score (log loss, Brier) optimised in expectation by the true probability
| Platt scaling | $\sigma(a s + b)$ fitted to held-out scores
| Temperature scaling | Divide all logits by one fitted $T$ before the softmax; accuracy unchanged
| Isotonic regression | Best non-decreasing step map from score to probability (PAV)
| Prior shift | A change in the class base rate; corrected by shifting the log-odds

<div class="take">
<div class="lab">Key takeaway</div>

ranking quality and probability quality are different things. Check calibration with a reliability diagram plus a proper score such as log loss, fix it on held-out data with Platt or temperature scaling (little data, sigmoid-shaped distortion) or isotonic regression (plenty of data, any monotone distortion), and recalibrate whenever the sampling or the base rate changes.

</div>
