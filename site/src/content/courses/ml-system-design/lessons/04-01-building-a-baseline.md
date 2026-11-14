---
id: "04.01"
course: "ml-system-design"
module: "04"
slug: "building-a-baseline"
title: "Building a baseline"
keyIdeas: ["A baseline is the simplest system that solves the task end to end. It sets the bar, exposes data and pipeline bugs, and often ships first.", "Build baselines in layers: a constant or popularity rule, then a heuristic, then a simple model (logistic regression or gradient-boosted trees) on a few strong features.", "Getting the pipeline right with a simple model first separates infrastructure bugs from modelling problems.", "Every later model must beat the baseline on the business metric by enough to pay for its extra cost and complexity."]
question: "You are asked to build a model that predicts next week's sales for each of 5,000 products in 300 stores. Describe three baselines of increasing complexity, what each would reveal, and how you would decide when to stop adding complexity."
answer: "Baseline 1: last week's sales for the same product and store (a naive forecast), or the same week last year to capture seasonality. It reveals how predictable the series are and gives an error floor that any model must beat; check its error by product category and store size. Baseline 2: a seasonal moving average, for example the average of the last four weeks scaled by a yearly seasonal index, plus known promotions as multipliers. It shows how much simple seasonality and promotions explain. Baseline 3: a gradient-boosted tree model on lagged sales, rolling means, calendar features, price, promotion flags and store features, trained across all series. It shows whether cross-series learning and interactions help. Stop when the improvement in forecast error no longer translates into meaningful business value (fewer stock-outs or less waste) or when it costs more to run and maintain than it saves; check this per segment, because complex models often help only the high-volume products."
sources: [{"title": "Google, Rules of Machine Learning, Rule 4", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "\"Keep the first model simple and get the infrastructure right.\""}, {"title": "Andrej Karpathy, A Recipe for Training Neural Networks (2019)", "url": "https://karpathy.github.io/2019/04/25/recipe/", "note": "Start simple, verify the pipeline, then add complexity."}, {"title": "Chip Huyen, Machine Learning Systems Design (free notes)", "url": "https://huyenchip.com/machine-learning-systems-design/toc.html", "note": "Baselines and iterative development."}]
practice: ["popularity-baseline-leave-last-out", "logistic-regression", "linear-regression", "gradient-boosted-regressor"]
tags: ["baselines", "iteration"]
---

## Why start simple

A first model has two jobs: to deliver value and to prove the pipeline works. If you start with a complex deep model and get poor results, you cannot tell whether the problem is the data, the features, the training code, the evaluation or the model. A simple model on a working pipeline isolates those questions. Most production ML gains come from better data and features, which a simple model exploits as well as a complex one.

## Layers of baselines

| Layer | Example | What it tells you
| --- | --- | --- |
| Constant | predict the mean, or the majority class | the floor any model must beat
| Popularity or recency | most popular items, last value | how much signal is in aggregate behaviour
| Heuristic | business rules from domain experts | what experts already know
| Simple model | logistic regression, gradient-boosted trees | how much the features can explain
| Complex model | deep networks, sequence models | the extra gain from capacity

Each layer is cheap to build once the previous one exists, and each comparison is informative. If gradient-boosted trees barely beat popularity, the features are weak, and a bigger model will not help much.

~~~dg-flow Iterating from a baseline. Each step should beat the last on the metric that matters.
Constant -(+ aggregate behaviour)-> Popularity -(+ domain rules)-> Heuristic -(+ features)-> *Simple model -(+ capacity)-> Complex model
~~~

## Gradient-boosted trees as a default

For tabular data (fraud, pricing, churn, click-through with engineered features), gradient-boosted trees are a strong default. They handle mixed feature types, missing values and nonlinear interactions; they train quickly on a single machine; and they are easy to inspect with feature importances. Deep models win when inputs are raw text, images, sequences or very high-cardinality IDs that benefit from learned embeddings.

## Testing the pipeline

A baseline is a test harness. Before improving the model, check:

- the model beats a constant on held-out data (if not, there is a bug or no signal);
- it can overfit a tiny subset to near-perfect accuracy (if not, there is a bug in training);
- offline predictions on logged requests match what the serving path returns for the same inputs;
- the metric computed in your evaluation code matches a hand-computed value on a few examples.

<div class="ex">
<div class="lab">Worked example</div>

**a baseline that found a bug**

A team spent two weeks tuning a neural ranker that scored below a popularity baseline. Training logistic regression on the same features showed the same problem. The cause was a join that attached each click to the wrong impression, shifting labels by one row. The baseline would have found it on day one.

</div>

## Deciding whether complexity pays

Compare each candidate with the baseline on:

- the offline metric, with confidence intervals across several time windows;
- the business metric in an online test;
- training and serving cost, latency and memory;
- operational burden: retraining time, debugging difficulty, explainability.

A model that improves AUC by 0.3% but doubles serving cost and takes a day to retrain may not be worth it. Often the right move is to keep the simple model and spend effort on new features or better labels.

## Baselines in an interview

State the baseline before the fancy model: "First, a popularity baseline per region; then logistic regression on user and item features; then a two-tower retrieval with a gradient-boosted or neural ranker." It shows you can ship something quickly and that you will measure each step. Interviewers often ask, "how would you know the complex model is better?" The baseline is the answer.

## Keep the baseline running

Do not delete the baseline once a better model ships. Keep it as the fallback when the main model or its features are unavailable, as a reference in every evaluation report, and as a canary: if the production model's margin over the baseline shrinks, something has degraded. Running both on a small slice of traffic costs little and gives a steady comparison over time.

<div class="take">
<div class="lab">Key takeaway</div>

build baselines in layers from a constant to a simple model, use them to prove the pipeline and measure each added piece of complexity, and keep the baseline as fallback and reference after better models ship.

</div>
