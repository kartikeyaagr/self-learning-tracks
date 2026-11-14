---
id: "03.04"
course: "machine-learning"
module: "03"
slug: "calibrated-probabilities"
title: "Calibrated probabilities"
keyIdeas: ["A classifier is **calibrated** if, among cases where it predicts $p$, the event happens a fraction $p$ of the time; this is separate from discrimination, which AUC measures and any monotone transform leaves unchanged.", "Check it with a **reliability diagram** (mean prediction against observed frequency per bin), and summarise it with ECE alongside a proper scoring rule such as log loss or the Brier score.", "**Platt scaling** fits $\\sigma(a s + b)$ to held-out scores: two parameters, little data needed, good for sigmoid-shaped distortion. Temperature scaling is its one-parameter multiclass cousin.", "**Isotonic regression** fits any non-decreasing map with pool-adjacent-violators: more flexible, but it needs roughly a thousand or more calibration points and can overfit.", "Always calibrate on data the model was not trained on, and recalibrate after resampling, class reweighting or a change in base rate."]
question: "Your gradient-boosted model has validation AUC 0.91 but a reliability diagram shows it is overconfident at both ends (ECE 0.12). You have 300 labelled examples not used in training. Do you choose Platt scaling or isotonic regression, what happens to AUC and log loss after calibration, and what mistake would make the calibration itself unreliable?"
answer: "Choose Platt scaling. With 300 examples, isotonic regression's step function has too many degrees of freedom and will overfit the noise in each step. Platt's two parameters are well determined, and a symmetric over-confidence pattern is roughly sigmoid-shaped in logit space, which $\\sigma(a s + b)$ with $a < 1$ corrects. Platt's map is strictly increasing, so the ranking is unchanged and AUC stays 0.91. Log loss and Brier score should improve, and ECE should drop substantially. Isotonic could lower AUC slightly through ties. The mistake to avoid is fitting the calibrator on the training data or on the same split used to tune the model. Those scores are optimistically confident, so the calibrator learns too small a correction. Fit on the 300 untouched examples, or use cross-validated calibration. Then evaluate on yet another split, or with cross-validation, because 300 points give a noisy ECE."
sources: [{"title": "Predicting Good Probabilities With Supervised Learning", "url": "https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf", "note": "Niculescu-Mizil and Caruana (ICML 2005): how each model family miscalibrates, and Platt vs isotonic by data size."}, {"title": "On Calibration of Modern Neural Networks", "url": "https://arxiv.org/abs/1706.04599", "note": "Guo et al. (2017): overconfidence in deep nets, reliability diagrams, ECE and temperature scaling."}, {"title": "scikit-learn: Probability calibration", "url": "https://scikit-learn.org/stable/modules/calibration.html", "note": "Calibration curves, CalibratedClassifierCV and the sigmoid vs isotonic choice."}, {"title": "scikit-learn: Isotonic regression", "url": "https://scikit-learn.org/stable/modules/isotonic.html", "note": "The isotonic model and the pool-adjacent-violators fit behind it."}]
practice: ["expected-calibration-error", "log-loss-binary-cross-entropy", "logistic-regression", "auc-roc"]
tags: ["calibration", "reliability diagram"]
tools: ["NumPy", "scikit-learn"]
---

## What calibration means

Many decisions use a probability as a number, not just to rank: expected cost $p \cdot \text{loss}$, an ad's expected value $p_{\text{click}} \times \text{bid}$, a threshold chosen on last month's data, or combining several models' outputs. These need **calibration**:

$$\Pr(Y = 1 \mid \hat p(X) = p) = p \quad \text{for all } p.$$

Of 1,000 emails given spam probability 0.8, about 800 should be spam. Calibration is separate from **discrimination**, how well scores rank positives above negatives. AUC measures ranking only, so any monotone transform of the scores keeps AUC and can wreck or repair calibration. A model can have excellent AUC with useless probabilities; one that predicts the base rate for everyone is perfectly calibrated and useless for ranking.

Models from earlier lessons miscalibrate in characteristic ways:

- **Naive Bayes**: pushed toward 0 and 1, because correlated evidence is double-counted (03.01).
- **SVMs**: scores are margins, not probabilities (02.05).
- **Boosted trees**: often pushed away from 0 and 1, a sigmoid-shaped distortion; **random forests** rarely predict near 0 or 1 because averaging trees pulls them in.
- **Logistic regression**: usually close to calibrated when well specified and regularised lightly, since it directly optimises log loss.
- **Modern deep networks**: often overconfident; Guo et al. found temperature scaling fixes much of it.

## Measuring it

A **reliability diagram** sorts predictions into bins (10 equal-width bins, or equal-count bins when predictions cluster), and plots each bin's mean predicted probability against the observed positive rate. Points on the diagonal mean calibrated; below it means overconfident on the high side.

**Expected calibration error** summarises the gaps, weighted by bin size: $\mathrm{ECE} = \sum_b \frac{n_b}{n}\,|\bar y_b - \bar p_b|$. It is easy to read but depends on the binning and can be gamed (constant predictions score near zero). Report it next to a **proper scoring rule**: log loss or the Brier score $\frac1n\sum_i(\hat p_i - y_i)^2$. Proper scores are minimised by the true probabilities and reward calibration and discrimination together. Module 06 covers the threshold-free ranking metrics; the ML in production module (09) covers monitoring calibration drift after launch.

## Platt scaling and temperature scaling

Platt scaling fits a one-feature logistic regression on held-out scores $s$:

$$\hat p = \sigma(a\,s + b),$$

with $a, b$ chosen to minimise log loss on a calibration set. Two parameters means it needs little data (a few hundred examples) and rarely overfits. It assumes the distortion is sigmoid-shaped, which suits SVM margins and boosted trees well. Platt's paper also replaces 0/1 targets with $(N_+ + 1)/(N_+ + 2)$ and $1/(N_- + 2)$ to avoid overfitting on small sets.

**Temperature scaling** is the multiclass version with one parameter: divide all logits by $T$ before the softmax and fit $T$ on validation log loss. $T > 1$ softens overconfident predictions. Because dividing by a positive constant never changes the argmax, accuracy is untouched.

## Isotonic regression

When the distortion is not sigmoid-shaped, fit an arbitrary non-decreasing step function from score to probability. The **pool-adjacent-violators** algorithm (PAV) does it in linear time after sorting:

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

Isotonic regression can fix any monotone distortion, but it has as many degrees of freedom as there are steps. With small calibration sets it overfits, and Niculescu-Mizil and Caruana found Platt better below roughly a thousand calibration points, isotonic better above. Its flat steps also create ties, which can lower AUC slightly.

## Doing it without leakage

~~~dg-flow Calibration is a second model fitted on data the first model never saw, then applied to its scores at serving time.
Training split -(fit)-> Classifier -(score)-> Calibration split -(fit Platt or isotonic)-> *Calibrator -(apply at serving)-> Probabilities
~~~

- **Hold-out data.** A model's scores on its own training data are overconfident, so calibrating on them learns the wrong map. Use a separate split, or cross-validation: scikit-learn's `CalibratedClassifierCV(method="sigmoid" or "isotonic", cv=5)` fits the calibrator on out-of-fold predictions.
- **Resampling breaks calibration.** Undersampling negatives or using class weights shifts probabilities toward the minority class. Correct by shifting the logit by $\log\frac{\pi}{1-\pi} - \log\frac{\pi_s}{1-\pi_s}$, where $\pi$ is the true base rate and $\pi_s$ the training one, or recalibrate on data with the true rate.
- **Base rates move.** If prevalence changes between training and deployment (prior shift), the same logit correction applies with the new rate. Recalibrate on recent labelled data.

<div class="ex">
<div class="lab">Worked example</div>

**undoing balanced sampling, then PAV**

A fraud model is trained on a 1:1 sample, but the true fraud rate is 1%. The logit shift is $\ln(0.01/0.99) - \ln(1) = -4.60$. A raw output of 0.5 (logit 0) becomes $\sigma(-4.60) = 0.01$; a raw 0.9 (logit 2.20) becomes $\sigma(-2.40) = 0.083$. Without the correction, the cost model would overstate fraud risk by a factor of 10 to 50.

PAV on labels sorted by score, $(0, 1, 0, 0, 1, 1)$: the 1 then 0 violate order and pool to 0.5; the next 0 violates again and pools to $1/3$. The result is $(0, \tfrac13, \tfrac13, \tfrac13, 1, 1)$: a monotone step function.

ECE contribution of one bin: 200 of 1,000 predictions, mean prediction 0.80, observed rate 0.60, so $0.2 \times 0.20 = 0.04$.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

ranking quality and probability quality are different. Check calibration with a reliability diagram and a proper score, fix it on held-out data with Platt scaling (little data) or isotonic regression (lots), and recalibrate whenever sampling or base rates change.

</div>
