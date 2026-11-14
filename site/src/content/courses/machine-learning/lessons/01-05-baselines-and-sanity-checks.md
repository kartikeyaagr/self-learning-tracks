---
id: "01.05"
course: "machine-learning"
module: "01"
slug: "baselines-and-sanity-checks"
title: "Baselines and sanity checks"
keyIdeas: ["Before any real model, compute a **trivial baseline**: majority class for classification, mean or median for regression, last value for time series.", "A **linear baseline** (logistic or ridge on standardised features) sets the bar a complex model must clear to justify its cost.", "**Learning curves** (score vs training-set size) tell you whether more data will help, separating bias from variance problems.", "Cheap sanity checks catch pipeline bugs: overfit a tiny subset, shuffle labels and expect chance, and check that the loss at initialisation matches theory."]
question: "Your new recommender beats a random-item baseline by 40% on hit rate@10, and your manager is pleased. What baseline is missing, why could it change the conclusion, and what would a learning curve add to the decision about whether to keep investing?"
answer: "The missing baseline is **most-popular items** (optionally per segment or recent window). Random items are a straw man: popularity alone usually captures a large share of hits, so a model that beats random by 40% might be no better than, or even worse than, simply recommending the top sellers. Compare against popularity on the same split (ideally time-ordered, leave-last-out per user). A learning curve (hit rate vs number of training interactions) then shows the trend: if validation hit rate is still climbing at full data and the train/validation gap is large, more interaction data should help and investment is justified; if both curves have flattened close together, the model is bias-limited and you need new signals or a different model, not more of the same data."
sources: [{"title": "A Recipe for Training Neural Networks (Karpathy)", "url": "https://karpathy.github.io/2019/04/25/recipe/", "note": "The \"set up the end-to-end skeleton and get dumb baselines\" section: init-loss checks, overfitting one batch, input-independent baselines."}, {"title": "scikit-learn: Validation curves and learning curves", "url": "https://scikit-learn.org/stable/modules/learning_curve.html", "note": "How to compute and read both plots."}, {"title": "scikit-learn: DummyClassifier", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html", "note": "The trivial strategies (most_frequent, prior, stratified, uniform) as ready-made baselines."}, {"title": "Rules of Machine Learning (Google)", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Rules 1-4 and 13-17: launch a simple model and heuristic first, then iterate."}]
practice: ["pos-tagging", "popularity-baseline-leave-last-out", "mini-batch-training-loop", "regression-metrics"]
tags: ["baselines", "learning curves"]
tools: ["scikit-learn"]
---

## Why baselines first

A model with 95% accuracy sounds good until you learn that 95% of the labels are the same class. Without a baseline, a number has no meaning. Baselines do three jobs:

- They tell you whether the problem is **learnable at all** from these features.
- They catch **pipeline bugs**: if your fancy model barely beats the majority class, suspect the data before the model.
- They set a **cost bar**: a gradient-boosted model that beats logistic regression by 0.2 points may not be worth its training and serving cost.

## The ladder of baselines

Climb it in order, and record each number in the same table with the same split and metric.

~~~dg-flow Each rung must beat the previous one by a margin larger than its noise before you climb further.
Trivial predictor -(beats?)-> Simple heuristic -(beats?)-> Linear model -(beats?)-> *Complex model
~~~

| Task | Trivial baseline | Score it gives |
| --- | --- | --- |
| Binary classification | predict the majority class | accuracy = majority rate; AUC = 0.5 |
| Classification, probabilistic | predict class frequencies | log loss = entropy of the label distribution |
| Regression (MSE) | predict the training mean | $R^2 = 0$ on that data |
| Regression (MAE) | predict the training median | MAE of the median |
| Time series | repeat the last value or the same day last week | "naive" or "seasonal naive" error |
| Ranking / recommendation | most popular items | popularity hit rate |

A **simple heuristic** is the rule a domain expert would write in five minutes: "flag transactions over 5,000 dollars from new accounts". Then the **linear baseline**: logistic regression or ridge on standardised features with basic encoding. Linear models are fast, stable, hard to overfit at moderate dimensionality, and their coefficients are a free bug detector (a huge weight on "row id" means leakage).

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
for name, m in [("majority", DummyClassifier(strategy="most_frequent")),
                ("prior", DummyClassifier(strategy="prior")),
                ("logistic", LogisticRegression(max_iter=1000))]:
    m.fit(X_tr, y_tr); print(name, m.score(X_va, y_va))
```

## Learning curves

A learning curve plots training and validation score against the number of training examples, for a fixed model. Train on 10%, 20%, ... 100% of the training set and evaluate each on the same validation set. The shape answers "will more data help?" before you pay for it:

| Shape | Reading | Action |
| --- | --- | --- |
| Both curves high error and close together, flat | high bias; more data won't help | richer model or better features |
| Large gap, validation still improving at full size | high variance; more data will help | collect data, or regularise |
| Large gap, validation flat | variance that data alone won't fix | regularise, simplify, check for leakage or label noise |
| Validation better than training | usually a split or augmentation artefact | check the split |

A related plot, the **validation curve**, sweeps a hyperparameter instead of data size; that is the capacity sweep from lesson 01.02.

## Sanity checks that catch bugs

These take minutes and catch many of the bugs that waste days:

1. **Loss at initialisation.** A $K$-class softmax with small random weights should start at log loss of about $\ln K$ (2.30 for 10 classes). A much larger value means badly scaled logits; a much smaller one means leakage.
2. **Overfit a tiny subset.** Take 20 examples and train to near-zero training loss. If a model with enough capacity cannot memorise 20 points, the training loop, labels or gradients are broken.
3. **Shuffle the labels.** Retrain with labels permuted: validation score should fall to the trivial baseline. If it does not, information is leaking through the features or the split.
4. **Check predictions by eye.** Look at the 20 most confident errors. Mislabelled examples, encoding bugs and leaked IDs show up immediately.
5. **Compare to the baseline on slices.** A model can beat the overall baseline while losing to it on an important segment.

Karpathy's "A Recipe for Training Neural Networks" lists these as the first steps for deep models, but they apply unchanged to any model in this course.

<div class="ex">
<div class="lab">Worked example</div>

**a medical triage classifier**

Data: 8,000 visits, 6% urgent. The majority baseline scores 94% accuracy and 0 recall on urgent cases. A logistic regression on 30 features reaches AUC 0.81. A gradient-boosted model reaches AUC 0.83 on the same 5 folds, standard deviation 0.012 across folds.

The learning curve for the boosted model still rises at 8,000 rows (validation AUC 0.79 at 4,000, 0.83 at 8,000) with a large train/validation gap: variance-limited. The team decides that labelling 8,000 more visits is a better investment than more tuning, and keeps logistic regression in production until the gap over it exceeds two standard deviations.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

always report a model next to a trivial and a linear baseline on the same split, read learning curves before buying data, and run the cheap sanity checks before trusting any number.

</div>
