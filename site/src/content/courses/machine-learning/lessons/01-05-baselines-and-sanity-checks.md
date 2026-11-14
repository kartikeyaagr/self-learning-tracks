---
id: "01.05"
course: "machine-learning"
module: "01"
slug: "baselines-and-sanity-checks"
title: "Baselines and sanity checks"
keyIdeas: ["Before any real model, compute a **trivial baseline**: majority class or class frequencies for classification, mean or median for regression, last value or same-season value for time series. A score means little until it sits next to one.", "Climb the ladder in order: trivial predictor, a simple expert rule, a **linear baseline** (logistic or ridge on standardised features), then a complex model, all on the same folds and metric.", "A gain over a baseline is only convincing when it exceeds the fold-to-fold noise and is worth the extra cost of the bigger model.", "**Learning curves** (score versus training-set size) tell you whether more data will help: curves that meet early signal high bias, a large gap with validation still rising signals high variance.", "Cheap sanity checks catch pipeline bugs: the first loss should be about $\\ln K$, a working loop can memorise 20 examples, and with shuffled labels the score must fall to the trivial baseline."]
question: "Your new recommender beats a random-item baseline by 40% on hit rate@10, and your manager is pleased. What baseline is missing, why could it change the conclusion, and what would a learning curve add to the decision about whether to keep investing? Name one sanity check you would run before presenting the numbers."
answer: "The missing baseline is **most-popular items** (optionally per segment or over a recent time window). Random items are a straw man: popularity alone usually captures a large share of hits, because a few items account for most interactions, so a model that beats random by 40% might be no better than, or even worse than, simply recommending the top sellers. Compare against popularity on the same split, ideally a time-ordered leave-last-out split per user so that the evaluation mimics recommending the next item, and with the same metric (hit rate@10). Report the fold or bootstrap spread too, so a small gap over popularity is not mistaken for a real one.\n\nA learning curve (hit rate versus the number of training interactions, with validation measured on the same held-out interactions) then shows the trend. If validation hit rate is still climbing at full data and the train/validation gap is large, the model is variance-limited: more interaction data should help and further investment is justified. If both curves have flattened close together, the model is bias-limited and you need new signals (content features, context, recency) or a different model class, not more of the same data.\n\nA useful sanity check before presenting: shuffle the user-item interactions relative to the users (or recompute with labels permuted) and confirm the hit rate falls to the popularity or random level; if it stays high, information is leaking, for example through items that appear in both the training history and the evaluation target. Also check that no test interaction predates the training data in a time-ordered split."
sources: [{"title": "A Recipe for Training Neural Networks (Karpathy)", "url": "https://karpathy.github.io/2019/04/25/recipe/", "note": "The \"set up the end-to-end skeleton and get dumb baselines\" section: init-loss checks, overfitting one batch, input-independent baselines."}, {"title": "scikit-learn: Validation curves and learning curves", "url": "https://scikit-learn.org/stable/modules/learning_curve.html", "note": "How to compute and read both plots."}, {"title": "scikit-learn: DummyClassifier", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html", "note": "The trivial strategies (most_frequent, prior, stratified, uniform) as ready-made baselines."}, {"title": "Rules of Machine Learning (Google)", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Rules 1-4 and 13-17: launch a simple model and heuristic first, then iterate."}]
practice: ["pos-tagging", "popularity-baseline-leave-last-out", "mini-batch-training-loop", "regression-metrics"]
tags: ["baselines", "learning curves"]
tools: ["scikit-learn"]
---

## What you will learn

- Why a score means nothing until you put it next to a **baseline**, and which trivial baseline fits each kind of task.
- How to climb the **ladder of baselines**: trivial predictor, simple rule, linear model, and only then a complex model.
- How to read a **learning curve** to decide whether more data, a bigger model or better features is the next investment.
- Five cheap **sanity checks** that catch pipeline bugs before they cost you days: loss at initialisation, overfitting a tiny subset, shuffled labels, reading the worst errors, and slice comparisons.
- How to tell whether a gain over a baseline is real or just fold-to-fold noise.

*Before you start:* you should know train/validation/test splits and cross-validation (01.03), the bias-variance trade-off and what over- and underfitting look like (01.02), and basic metrics such as accuracy and log loss (01.01). Familiarity with logistic regression helps but is not required; module 02 covers it in full.

## Closing the loop on module 01

Lesson 01.04 finished the machinery: losses (01.01), generalisation (01.02), honest splits (01.03) and penalties (01.04). Each produces a *number*, and a number alone cannot tell you whether the model is good, whether the pipeline is broken, or whether next week should go into data or modelling. Baselines and sanity checks answer those questions cheaply.

## The idea in one picture

A hospital wants to flag emergency-room visits that need urgent attention. You have 8,000 past visits with five vital signs (age, heart rate, blood oxygen saturation SpO2, temperature, blood pressure) and 25 other recorded columns. About 6% of visits (5.8% in this simulated data) were urgent. You train four models on the same five cross-validation folds:

| Model | Accuracy | Recall on urgent | ROC AUC
| --- | --- | --- | --- |
| Always predict "not urgent" | 94.2% | 0% | 0.50
| Rule: SpO2 < 92 or heart rate > 120 | 92.1% | 32% | 0.64
| Logistic regression | 94.3% | 3% | 0.79 ± 0.015
| Gradient boosting | 94.4% | 9% | 0.81 ± 0.013

~~~dg-fig 01.05-ladder.svg
Four rungs of the ladder on the same folds. Accuracy (left) is about 94% for everything, including the model that never flags anyone. Recall and ROC AUC (right, chance level dashed) are what separate the rungs.
~~~

A model that has learned *nothing* and never flags a patient scores 94.2% accuracy; boosting beats it by 0.2 points. Without the first row, "94% accurate" sounds impressive. With it, you see that accuracy is the wrong lens here and that the ranking metric (AUC) is where progress shows.

A **baseline** is a reference model whose score you compute on the same data, split and metric as your real model, so that the real model's score has something to be compared with. A **trivial baseline** uses no features at all (it ignores the input). Baselines do three jobs:

- **Learnability**: if nothing beats the trivial row, the signal is not in these features, or the pipeline is broken.
- **Bug detection**: a fancy model that barely beats the majority class points at the data before the model.
- **Cost bar**: a model that beats logistic regression by a hair may not be worth its serving and maintenance cost.

## Step 1: the trivial baseline for your task

The trivial baseline is the best constant prediction under your metric, so it depends on the metric.

| Task | Trivial baseline | Score it gives
| --- | --- | --- |
| Binary classification, accuracy | predict the majority class | accuracy = majority rate; AUC = 0.5
| Classification, probabilistic | predict the class frequencies | log loss = entropy of the label distribution
| Regression scored by MSE | predict the training mean | $R^2 = 0$ on that data
| Regression scored by MAE | predict the training median | MAE of the median
| Time series | repeat the last value, or the same day last week | "naive" or "seasonal naive" error
| Ranking / recommendation | most popular items | popularity hit rate

**Log loss** is the average of $-\ln p$, where $p$ is the probability given to the label that actually happened. The **entropy** of the labels is the log loss of predicting each class's overall frequency; no input-free predictor does better.

<div class="ex">
<div class="lab">Worked example</div>

**trivial baselines for the triage data, by hand**

Suppose 6% of visits are urgent. On 1,000 validation visits that is 60 urgent and 940 not urgent.

Majority class: predict "not urgent" for everyone. Correct on 940 of 1,000, so accuracy is $940/1000 = 94\%$. Urgent cases caught: $0/60$, so recall is 0.

Prior probabilities: predict $p = 0.06$ for every visit. For an urgent visit the loss is $-\ln 0.06 = 2.813$. For a non-urgent visit it is $-\ln 0.94 = 0.0619$.

Average log loss: $0.06 \times 2.813 + 0.94 \times 0.0619 = 0.1688 + 0.0582 = 0.227$. That is the entropy of a 6/94 split, in nats.

For comparison, predicting $p = 0.5$ for everyone gives $-\ln 0.5 = 0.693$, three times worse. So a model with log loss 0.30 *sounds* good next to 0.693 but is actually worse than knowing nothing except the base rate.

A regression version: delivery times of 20, 22, 25, 28 and 55 minutes. The mean is $150/5 = 30$, the median is 25. Predicting the mean gives errors $-10, -8, -5, -2, 25$, so MSE $= (100+64+25+4+625)/5 = 163.6$ (RMSE 12.8) and MAE $= 50/5 = 10$. Predicting the median gives absolute errors $5, 3, 0, 3, 30$, so MAE $= 41/5 = 8.2$ but MSE $= 943/5 = 188.6$. The mean wins on squared error, the median on absolute error: pick the constant that matches your metric.

</div>

On the simulated data the prior predictor scores log loss 0.221 (entropy of a 5.8% rate) and logistic regression 0.186; only the gap between them is evidence of learning.

## Step 2: a simple heuristic

The next rung is the rule a domain expert would write in five minutes: "flag the visit if oxygen saturation is below 92% or heart rate is above 120". It uses one or two features and no training at all.

Heuristics are often what the business already does, so they are the real competitor: a model that cannot beat the current rule should not replace it. In the figure the rule catches 32% of urgent cases (more than either trained model at the default 0.5 threshold) but has only 0.64 AUC, because a yes/no rule cannot rank patients by risk. That is itself a finding: the trained models need a lower threshold before deployment (module 06).

## Step 3: a linear baseline

Then fit a **linear baseline**: logistic regression for classification, ridge for regression, on standardised features (module 02 covers both). They train in seconds, have one best answer, are hard to overfit with a little L2 penalty (01.04), and their coefficients are a free bug detector: a huge weight on "row id" or "discharge code" means leakage (01.03).

Record every rung in one table with the **same folds and metric**; scores from different splits are not comparable.

~~~dg-flow Each rung must beat the previous one by more than its fold-to-fold noise before you climb further.
Trivial predictor -(beats?)-> Simple heuristic -(beats?)-> Linear model -(beats?)-> *Complex model
~~~

## Step 4: is the gain over the baseline real?

Cross-validation gives each model a mean and a fold standard deviation. A quick, conservative rule: trust a gap only when it exceeds about two fold standard deviations, and only act on it if it is worth the extra cost.

<div class="ex">
<div class="lab">Worked example</div>

**should the triage team switch to gradient boosting?**

Logistic regression: AUC 0.789, fold standard deviation 0.015. Gradient boosting: AUC 0.814, standard deviation 0.013, on the same five folds.

Gap: $0.814 - 0.789 = 0.025$. Two standard deviations of the noisier model: $2 \times 0.015 = 0.030$. The gap is smaller than the noise bar, so by the quick rule it is not yet convincing.

Next, the learning curve (figure below). Boosting's validation AUC is 0.824 at 4,000 rows, 0.838 at 8,000 and 0.845 at 16,000, and its training AUC is still well above validation: it is variance-limited and more data helps. Logistic regression is flat at about 0.79 from 4,000 rows on: more data will not help it.

Decision: label more visits (a better investment than more tuning), keep logistic regression in production for now, and switch once boosting's lead over it clearly exceeds the noise bar on the larger dataset.

</div>

## Step 5: learning curves

A **learning curve** plots training and validation score against the number of training examples for one fixed model: train on 250, 500, 1,000 ... rows, score each on the same validation set. Lesson 01.02 used it to illustrate bias and variance; here it answers "will more data help?" before you pay for data.

~~~dg-fig 01.05-learning-curves.svg
Learning curves on a larger simulated triage pool. Logistic regression (left) converges by about 4,000 rows near 0.79, so it is bias-limited. Gradient boosting (right) starts by memorising its training set, and its validation score is still climbing at 16,000 rows.
~~~

| Shape | Reading | Action
| --- | --- | --- |
| Both curves close together, flat, at a poor score | high bias; more data will not help | richer model or better features
| Large gap, validation still improving at full size | high variance; more data will help | collect data, or regularise
| Large gap, validation flat | variance that data alone will not fix | regularise, simplify, check for leakage or label noise
| Validation better than training | usually a split or augmentation artefact | check the split

The shaded area is the train/validation gap. For logistic regression it closes and the curves level off together: **high bias**, a model too simple for the data (it cannot express "heart rate too low *or* too high"). For boosting the gap is large but shrinking and validation still rises: **high variance** that more rows keep reducing.

A related plot, the **validation curve**, sweeps a hyperparameter (tree depth, penalty strength) instead of data size; that is the capacity sweep from lesson 01.02. In scikit-learn the two are `learning_curve` and `validation_curve` in `sklearn.model_selection`.

## Step 6: sanity checks that catch bugs

A baseline tells you whether the model is *good*; a sanity check tells you whether the pipeline is *working*. Each takes minutes.

1. **Loss at initialisation.** Before training, a $K$-class softmax with small random weights predicts roughly $1/K$ for every class, so its log loss should be about $\ln K$: 2.30 for 10 classes, 0.693 for 2. A much larger first loss means badly scaled logits; a much smaller one means the model already "knows" the answer, which points to leakage. For an imbalanced binary task, initialise the output bias to the log-odds of the base rate, $\ln(0.06/0.94) = -2.75$ for 6% positives, so that the first loss is already the entropy baseline.
2. **Overfit a tiny subset.** Take 20 examples and train until training loss is near zero. Any model with enough capacity can memorise 20 points. If it cannot, the training loop, the labels or the gradients are broken.
3. **Shuffle the labels.** Retrain with the labels randomly permuted. The validation score should fall to the trivial baseline. If it does not, information is leaking through the features or the split.
4. **Look at predictions by eye.** Read the 20 most confident errors; mislabels, encoding bugs and leaked IDs show up immediately.
5. **Compare to the baseline on slices.** A model can win overall but lose on an important segment (children, a new hospital site).

~~~dg-fig 01.05-sanity.svg
Two checks on softmax regression for 10 handwritten-digit classes. Left: with small weights the first loss is 2.30 = ln 10, exactly as theory predicts; with weights 100 times too large it starts at 6.2. Right: a healthy loop memorises 20 examples; a loop that feeds labels out of step with their inputs hovers around 2.3 and never comes down.
~~~

The right panel shows a common silent bug: inputs and labels drifting out of alignment (shuffling one array but not the other, an off-by-one in batching). Nothing crashes; the loss just stays at the "guess uniformly" level $\ln K$, as if the labels were noise.

The shuffle test catches leakage. Below, labels are coin flips, so no honest pipeline can beat AUC 0.5. One pipeline picks the 20 "best" of 2,000 noise columns using *all* rows, then cross-validates; the other selects inside each training fold.

~~~dg-fig 01.05-shuffle.svg
Forty runs with labels that carry no information. Feature selection done inside each fold scores around chance (0.50). Selecting features on all rows before splitting scores around 0.86 on pure noise: the leak, not the data, is producing the score.
~~~

The selection step had already seen the validation labels. With real labels this looks like skill; with noise labels it is unmistakable. scikit-learn automates the test as `permutation_test_score`. Karpathy's "A Recipe for Training Neural Networks" lists these checks as first steps for deep models; they apply to every model in this course.

## Baselines and sanity checks in code

The whole ladder for the triage problem, on one set of folds. `DummyClassifier` provides the trivial rungs; the rule is a five-line class.

```python
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, log_loss

# Simulated triage data: 5 vital signs + 25 irrelevant columns, ~6% urgent
rng = np.random.default_rng(0); n = 8000
age = rng.uniform(18, 90, n); hr = rng.normal(85, 15, n)
spo2 = np.clip(rng.normal(96, 2.5, n), 80, 100); temp = rng.normal(37, .7, n); sbp = rng.normal(125, 18, n)
logit = (-4.5 + .7*(hr-85)/15 + .5*abs(hr-85)/15 + np.maximum(0, 94-spo2)
         + .02*(age-50) + 1.5*(temp > 38.3) + 1.5*(sbp < 95))
y = (rng.random(n) < 1/(1+np.exp(-logit))).astype(int)
X = np.c_[age, hr, spo2, temp, sbp, rng.normal(size=(n, 25))]

class Rule:                      # the five-minute expert heuristic
    def fit(self, X, y): return self
    def predict_proba(self, X):
        p = ((X[:, 2] < 92) | (X[:, 1] > 120)).astype(float); return np.c_[1-p, p]

ladder = {"majority": DummyClassifier(strategy="most_frequent"),
          "prior":    DummyClassifier(strategy="prior"),
          "rule":     Rule(),
          "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
          "boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, random_state=0)}
cv = StratifiedKFold(5, shuffle=True, random_state=0)       # same folds for every rung
for name, model in ladder.items():
    s = []
    for tr, va in cv.split(X, y):
        p = model.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        s.append([accuracy_score(y[va], p > .5), recall_score(y[va], p > .5),
                  roc_auc_score(y[va], p), log_loss(y[va], np.clip(p, .001, .999))])
    m, sd = np.mean(s, 0), np.std(s, 0)
    print(f"{name:9s} acc {m[0]:.3f}  recall {m[1]:.2f}  AUC {m[2]:.3f}±{sd[2]:.3f}  logloss {m[3]:.3f}")
# majority  acc 0.942  recall 0.00  AUC 0.500±0.000  logloss 0.400
# prior     acc 0.942  recall 0.00  AUC 0.500±0.000  logloss 0.221
# rule      acc 0.921  recall 0.32  AUC 0.640±0.015  logloss 0.547
# logistic  acc 0.943  recall 0.03  AUC 0.789±0.015  logloss 0.186
# boosting  acc 0.944  recall 0.09  AUC 0.814±0.013  logloss 0.179
```

In the log-loss column, the majority and rule rows output hard 0/1 predictions (clipped to 0.001/0.999), so their confident mistakes are punished heavily. The prior row's 0.221 is the honest "knows nothing" bar. For regression, `DummyRegressor(strategy="mean")` or `"median"` plays the same role.

## Time series: the naive forecasts

For data ordered in time, the trivial baselines repeat the past. The **naive forecast** predicts tomorrow equals today. The **seasonal naive forecast** predicts the value from one season ago (the same weekday last week, the same month last year). They are embarrassingly hard to beat on many real series.

~~~dg-fig 01.05-naive-forecast.svg
Daily orders with a weekend peak, forecast one day ahead over two validation weeks. The training mean (MAE 11.0) and "same as yesterday" (MAE 10.4) both miss the weekly rhythm; "same day last week" follows it with MAE 4.6.
~~~

The mean and the naive forecast both ignore the weekly cycle; the seasonal naive forecast halves their error with zero learning. A model that does not beat 4.6 has learned nothing useful, however far it beats the mean. Split time series by time (01.03).

## Reading and checking the result

Before reporting a model, check: it sits next to trivial and linear baselines on the same folds; its gain exceeds fold noise and justifies its cost; a learning curve says whether data would help; the first loss was near theory and a tiny subset was memorised; shuffled labels fell to the trivial score; and you have read the most confident errors and checked key slices.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| No trivial baseline | "94% accurate" on a 94/6 problem impresses everyone | always report the majority or prior row
| Straw-man baseline | beating "random items" by 40% in a recommender | compare with the strongest cheap baseline (most popular items)
| Metric that hides failure | accuracy flat across models, recall near 0 | use AUC, log loss or recall, matched to the decision
| Ignoring noise | switching models for a gap of 0.005 AUC | compare the gap with the fold standard deviation
| Buying data blindly | more data, no improvement | read the learning curve first; flat and close means bias
| Leaky preprocessing | strong scores even with shuffled labels | fit every step inside the training fold (a `Pipeline`)

## Going deeper

**Why the trivial baselines are the right constants.** For squared error, the constant $c$ minimising $\frac1n\sum_i (y_i - c)^2$ has derivative $-\frac2n \sum_i (y_i - c) = 0$, so $c = \bar y$. For absolute error the derivative of $\sum_i |y_i - c|$ is (number of points below $c$) minus (number above), which is zero at the median. For log loss, the best constant probability for class $j$ is its frequency $\pi_j$, and the resulting loss $-\sum_j \pi_j \ln \pi_j$ is the entropy. Any model's improvement in log loss over this entropy is an estimate of how many nats of information the features carry about the label.

**$R^2$ is already a baseline comparison.** $R^2 = 1 - \text{MSE}_{\text{model}} / \text{MSE}_{\text{mean}}$ is literally "how much better than the mean baseline". A negative $R^2$ on test data means worse than predicting the mean. The forecasting analogue is MASE (mean absolute scaled error), which divides by the naive forecast's error, so MASE above 1 means worse than naive.

**Paired comparisons.** The two-standard-deviation rule ignores that both models used the same folds. Per-fold differences are sharper: in the triage example they are 0.027, 0.025, 0.023, 0.036 and 0.016 (mean 0.025, standard deviation 0.007), all positive. But fold scores are not independent (training sets overlap by 75%), so a plain t-test is overconfident; the Nadeau-Bengio corrected t-test or repeated cross-validation fixes this. A consistent sign is encouraging, not proof.

**Learning curves and the power law.** Validation error often falls roughly as $\text{err}(n) \approx a + b\,n^{-c}$, where $a$ is the floor set by noise and bias. Fitting it to a few points gives a rough estimate of what doubling the data would buy.

## Key terms

| Term | Meaning
| --- | --- |
| Baseline | A reference model scored on the same data, split and metric, so another score can be judged against it
| Trivial baseline | A predictor that ignores the inputs: majority class, class frequencies, mean, median, last value
| Heuristic baseline | A hand-written rule from domain knowledge, often what is already in use
| Linear baseline | Logistic or ridge regression on standardised features; the bar a complex model must clear
| Learning curve | Training and validation score plotted against training-set size for a fixed model
| Validation curve | Training and validation score plotted against one hyperparameter
| Sanity check | A cheap test with a known expected outcome that reveals pipeline bugs
| Seasonal naive forecast | Predicting the value from one season earlier, such as the same weekday last week

<div class="take">
<div class="lab">Key takeaway</div>

always report a model next to a trivial and a linear baseline on the same split, judge gaps against fold-to-fold noise, read learning curves before buying data, and run the cheap sanity checks (initial loss, tiny-subset overfit, shuffled labels) before trusting any number.

</div>
