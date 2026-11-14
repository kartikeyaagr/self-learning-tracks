---
id: "04.02"
course: "ml-system-design"
module: "04"
slug: "choosing-a-model-and-loss"
title: "Choosing a model and loss"
keyIdeas: ["Choose the model family from the data type, the amount of data, the latency budget and the need for explanation, not from fashion.", "The loss should match the decision: log loss for calibrated probabilities, pairwise or listwise losses for ranking, quantile losses when you need intervals, Huber for heavy-tailed targets.", "Ranking systems often use a **cascade**: a cheap model over many candidates, then a richer model over few.", "**Multi-task** models share a representation across related labels (click, purchase, return) and combine their outputs in a final score."]
question: "You need to predict delivery time for food orders and show customers a range (\"25-35 minutes\"). Which model family and loss would you choose, and how would you produce the range? What would you do if late deliveries hurt far more than early ones?"
answer: "Use gradient-boosted trees (or a small neural network if you have rich sequential inputs like courier GPS traces) on features such as restaurant prep-time history, current kitchen load, courier distance and availability, traffic, time of day and weather. For the point estimate, a squared or Huber loss on minutes, possibly on log minutes since errors scale with duration. For the range, train quantile regressors for the 10th and 90th percentiles (pinball loss) or one model with multiple quantile outputs, and check coverage: about 80% of actual times should fall inside the interval. If lateness is worse than earliness, show a higher quantile (for example the 70th percentile) as the headline estimate, or use an asymmetric loss that penalises under-prediction more, and track the share of orders arriving after the promised upper bound as a guardrail."
sources: [{"title": "Burges, From RankNet to LambdaRank to LambdaMART: An Overview (2010)", "url": "https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/", "note": "Pairwise and listwise learning-to-rank losses."}, {"title": "Ma et al., Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (KDD 2018)", "url": "https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/", "note": "Multi-task models for recommendation."}, {"title": "scikit-learn, Choosing the right estimator", "url": "https://scikit-learn.org/stable/machine_learning_map.html", "note": "A map of model families by data size and task."}]
practice: ["log-loss-binary-cross-entropy", "sigmoid-focal-loss", "ndcg-at-k", "regression-metrics", "gradient-boosted-regressor"]
tags: ["model selection", "loss functions"]
---

## Start from the constraints

Model choice follows from a few questions:

| Question | Points towards
| --- | --- |
| Tabular features, up to tens of millions of rows? | gradient-boosted trees
| Raw text, images, audio? | pretrained deep encoders, fine-tuned
| Huge sparse IDs (users, items, ads)? | neural models with embeddings, factorisation machines
| Sequences of events? | sequence models (RNN, transformer) or engineered sequence features
| Latency under 1 ms per item at high volume? | linear models, small trees, or distilled networks
| Must explain each decision? | linear models, shallow trees, or monotonic constraints
| Little labelled data? | pretrained models, transfer learning, simple models

Often the answer is a combination: trees on tabular features plus embeddings from a pretrained encoder fed in as features.

## Matching the loss to the decision

| Decision | Loss | Notes
| --- | --- | --- |
| Probability used in expected value (ads, risk pricing) | log loss | rewards calibration
| Classify rare events | weighted log loss, focal loss | then tune the threshold
| Order items | pairwise (RankNet), listwise (LambdaRank, softmax) | optimises ranking metrics more directly
| Predict a quantity | squared error, Huber, log-scale | Huber for outliers, log for multiplicative errors
| Predict a range | quantile (pinball) loss | one output per quantile
| Retrieve from a large set | sampled softmax, contrastive | with in-batch or sampled negatives

The pinball loss for quantile $\tau$ is $L_\tau(y, \hat y) = \max(\tau (y - \hat y), (\tau - 1)(y - \hat y))$. Training at $\tau = 0.9$ gives a value that the true outcome falls below about 90% of the time.

## Cascades

When there are many candidates, a single expensive model is too slow. A cascade uses cheaper models first:

~~~dg-flow A ranking cascade. Each stage trades recall for precision and spends more compute per item.
Millions of items -(ANN retrieval)-> 1,000 candidates -(light ranker: trees or small net)-> 200 -(heavy ranker: deep multi-task)-> *50 -(re-rank: diversity, rules)-> 10 shown
~~~

Each stage should have high recall for what the next stage would rank highly. Evaluate stages both alone and in combination: a better heavy ranker cannot fix candidates the retrieval stage never returned.

## Multi-task models

Products care about several outcomes: click, long view, purchase, share, report. A multi-task model shares lower layers and has one head per task, which helps sparse tasks borrow strength from dense ones. The final score combines heads with weights chosen by the business, for example $\text{score} = P(\text{click}) \cdot (1 + 2 P(\text{purchase} \mid \text{click})) - 5 P(\text{report})$. Mixture-of-experts architectures such as MMoE let tasks that conflict use different experts.

<div class="ex">
<div class="lab">Worked example</div>

**a loss that matched the product**

A marketplace ranked search results by predicted click-through, trained with log loss on clicks. Sellers complained that cheap, eye-catching listings got all the traffic while buyers rarely purchased them. Switching to a multi-task model with click and purchase heads, ranking by expected purchases, lowered click-through by 2% and raised purchases by 4%.

</div>

## Calibration and thresholds

If downstream code treats the score as a probability, check calibration after training (reliability diagram, expected calibration error) and recalibrate with Platt scaling or isotonic regression on recent data if needed. If the decision uses a threshold, choose it from business costs on a validation set, not 0.5.

## Model size versus the latency budget

A model that wins offline by 1% but needs 30 ms per item is useless in a stage with 10 ms for 500 items. Measure inference cost early. Distillation, quantisation and smaller architectures often recover most of the quality within budget.

## Hyperparameters and search

Tune what matters most first: learning rate and number of trees or epochs, regularisation, and model size. Use a time-based validation set and a modest random or Bayesian search; a few dozen trials is usually enough for trees. Record every trial with its data version so results can be compared later, and be suspicious of a configuration that wins by a tiny margin on one validation window only.

<div class="take">
<div class="lab">Key takeaway</div>

pick the model family from data type, data size, latency and explainability needs; pick the loss to match the decision; and use cascades and multi-task heads when there are many candidates or several outcomes to balance.

</div>
