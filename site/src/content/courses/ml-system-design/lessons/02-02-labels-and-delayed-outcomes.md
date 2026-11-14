---
id: "02.02"
course: "ml-system-design"
module: "02"
slug: "labels-and-delayed-outcomes"
title: "Labels and delayed outcomes"
keyIdeas: ["Labels come from three places: **natural outcomes** (clicks, purchases, chargebacks), **human annotation**, and **heuristics or weak supervision**. Each has its own cost, delay and bias.", "Many outcomes arrive late. Chargebacks can take 60 days, and churn is only known at the end of a period, so the most recent examples have incomplete labels.", "Training on recent data with immature labels teaches the model that recent examples are negative. Wait for labels to mature, or model the delay.", "Annotation needs clear guidelines, quality checks such as agreement rates and gold examples, and a plan for ambiguous cases."]
question: "A card-fraud model is retrained weekly on the last 90 days of transactions, labelled fraud if a chargeback has been filed. Chargebacks arrive up to 60 days after the transaction. What goes wrong, and give two ways to fix it."
answer: "For transactions in the last 60 days, many frauds have not been reported yet, so they are labelled legitimate. The most recent third of the training data has a fraud rate that is too low, and those false negatives look exactly like the newest fraud patterns, so the model learns to under-score them. Calibration also drifts downwards. Fix 1: only train on transactions old enough for labels to have matured (for example older than 60 days, or older than the 95th percentile of chargeback delay), accepting staler data. Fix 2: use faster partial signals as additional labels, such as customer dispute calls, analyst decisions or issuer fraud flags, which arrive in days. Fix 3: model the delay explicitly by weighting recent negatives by the probability that a fraud would already have been reported by now, using the empirical delay distribution."
sources: [{"title": "Yoshikawa and Imai, A Nonparametric Delayed Feedback Model for Conversion Rate Prediction (2018)", "url": "https://arxiv.org/abs/1802.00255", "note": "Modelling the delay between click and conversion, building on Chapelle (2014)."}, {"title": "Ratner et al., Snorkel: Rapid Training Data Creation with Weak Supervision (2017)", "url": "https://arxiv.org/abs/1711.10160", "note": "Combining noisy labelling functions into training labels."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Rules on objectives and labels."}]
practice: ["delayed-label-monitoring", "point-in-time-feature-join", "expected-calibration-error"]
tags: ["labels", "delayed feedback"]
---

## Where labels come from

| Source | Examples | Delay | Typical bias
| --- | --- | --- | --- |
| Natural outcomes | click, purchase, chargeback, churn | seconds to months | only observed for what the system showed
| Human annotation | content review, relevance ratings | days | annotator disagreement and guideline drift
| Heuristics, weak supervision | keyword rules, user reports | instant | systematic errors of the rule
| Implicit signals | dwell time, returns, skips | minutes to weeks | noisy proxy for satisfaction

Natural labels are cheap and plentiful but are tied to what the system did: you only learn whether users click on items you showed them. Human labels are expensive but can cover cases the system never acted on. Most systems combine both.

## The delay problem

Some outcomes take a long time. A chargeback can take 60 days; a loan default months; a subscription churn is only known when the renewal date passes. At any moment the newest examples have incomplete labels.

~~~dg-flow Label maturity. Examples to the right of the cut-off still have many positives missing.
Transaction -(day 0)-> Early signals: disputes, analyst flags -(days 1-14)-> Most chargebacks -(days 15-60)-> *Mature label
~~~

If you train on everything up to yesterday and treat "no chargeback yet" as negative, recent fraud is mislabelled. The model learns that recent patterns are safe, which is the opposite of what you want when fraudsters change tactics.

Options:

- **Wait**: only use examples older than the label maturity window. Simple and unbiased, but the model is always weeks behind.
- **Use faster proxies**: early signals such as disputes, analyst decisions or user reports arrive sooner and correlate with the final label.
- **Model the delay**: estimate the delay distribution $P(\text{reported by } t)$ and weight or reinterpret recent negatives accordingly, as in the delayed-feedback models of Chapelle (2014) and later work.
- **Two models**: a fast model on early signals and a slower, more accurate one on mature labels.

## Evaluating with delayed labels

Evaluation suffers too. A model's precision measured on last week's data looks great because few frauds have been reported yet. Always evaluate on periods whose labels have matured, and report metrics as of a fixed label age. The Machine Learning course's monitoring lesson (mlcore 09.03) covers proxy metrics while labels mature.

## Human annotation

When you need labels the system can't produce, such as "is this post hateful?" or "is this result relevant?", you need annotators:

- write guidelines with clear definitions and worked edge cases;
- measure agreement between annotators (for example Cohen's $\kappa$) and treat low agreement as a sign the task or guideline is unclear;
- seed queues with gold examples of known answers to monitor quality;
- route ambiguous items to experts, and feed their decisions back into the guideline.

Sample items for annotation deliberately: random samples estimate prevalence, while samples near the model's decision boundary (active learning) improve the model fastest.

<div class="ex">
<div class="lab">Worked example</div>

**labels that drifted**

A content-moderation team updated its hate-speech policy to include a new category of slurs. Old labels were not revisited, so training data contained the same kind of post labelled both ways depending on its date. Precision on the new category was poor until the team relabelled a sample of old data under the new guideline and versioned labels by policy version.

</div>

## Weak supervision

When annotation is too slow, write labelling functions (keyword rules, heuristics, outputs of other models) and combine them. Tools like Snorkel learn how accurate each function is and produce probabilistic labels. This gets a first model off the ground and can be refined with a small set of human labels.

## Label noise

All labels are noisy. A mislabelled 1% of examples rarely matters for a large model, but systematic noise does: every chargeback filed by a forgetful customer is a "fraud" label on a real purchase. Estimate noise by having experts re-review a sample, and report how often their labels disagree with the source.

## Choosing among label sources

In practice you pick a primary label for training and keep others for evaluation and guardrails. A shopping recommender might train on purchases, evaluate against purchases and returns, and use annotator relevance ratings on a fixed query set as a regression test. Writing down which label serves which purpose avoids a common confusion: a model that improves the training label while the evaluation label, which is closer to the goal, gets worse.

<div class="take">
<div class="lab">Key takeaway</div>

know where each label comes from, how late it arrives and how it is biased; train and evaluate only on matured labels or model the delay, and run annotation with guidelines, agreement checks and gold examples.

</div>
