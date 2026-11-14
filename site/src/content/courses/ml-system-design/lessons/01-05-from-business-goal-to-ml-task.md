---
id: "01.05"
course: "ml-system-design"
module: "01"
slug: "from-business-goal-to-ml-task"
title: "From business goal to ML task"
keyIdeas: ["A business goal (\"increase bookings\") must be turned into an **ML task**: a precise prediction with an input, an output, a label and a loss.", "Labels are usually **proxies** for the real goal. Choose proxies that are available quickly, correlate with the goal, and are hard to game.", "Common framings are classification, regression, ranking, retrieval, forecasting and generation. The same goal can often be framed several ways with different data needs.", "Track an **online business metric**, an **offline ML metric** that predicts it, and **guardrail metrics** that must not get worse."]
question: "A video platform wants to \"increase long-term user satisfaction\". Propose two possible ML tasks with labels, say what each proxy gets wrong, and describe how you would choose between them."
answer: "Task A: predict the probability a user clicks a recommended video (binary label from impressions). Clicks are fast and plentiful but reward clickbait thumbnails and say nothing about whether the video was good. Task B: predict expected watch time of the video if shown (regression on seconds watched, or weighted logistic regression). Watch time is closer to satisfaction but favours long videos and can reward compulsive viewing. Possible additions are survey responses (\"was this worth your time?\"), likes, shares and return visits, which are sparser but closer to the goal. Choose by running online experiments: train models on each objective and measure the business metric (for example 7-day retention or survey satisfaction) in an A/B test; keep the proxy whose offline gains best predict online gains, and add guardrails such as report rate and diversity."
sources: [{"title": "Google, Problem framing course", "url": "https://developers.google.com/machine-learning/problem-framing", "note": "Translating a goal into a model output and label."}, {"title": "Covington, Adams and Sargin, Deep Neural Networks for YouTube Recommendations (2016)", "url": "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/", "note": "Predicting watch time instead of clicks to avoid clickbait."}]
practice: ["log-loss-binary-cross-entropy", "ndcg-at-k", "regression-metrics"]
tags: ["ML objective", "proxy metrics"]
---

## The translation step

"Increase bookings", "reduce fraud losses" and "keep users engaged" are goals, not tasks. A model needs a precise target: given this input at this moment, predict this quantity. Getting the translation right matters more than the choice of model, because a model optimises exactly what you give it.

~~~dg-flow Goals become tasks through a chain of choices. Each arrow is a decision you should state.
Business goal -(which decision?)-> Prediction target -(which data?)-> Label -(which loss?)-> *ML task -(which metric?)-> Offline evaluation
~~~

## Common framings

| Framing | Output | Example
| --- | --- | --- |
| Binary classification | probability of an event | will this transaction be fraud?
| Multiclass | one of K labels | which queue should this ticket go to?
| Regression | a number | minutes until delivery
| Ranking | an ordering of items | order search results for this query
| Retrieval | a candidate set | 500 videos this user might like
| Forecasting | future values of a series | units sold per store next week
| Generation | text, image or code | a reply suggestion

The same goal can often be framed several ways. Recommending products can be a ranking problem trained on clicks, a classification problem trained on purchases, or a regression on predicted revenue. Each needs different labels and gives different behaviour.

## Labels are proxies

The thing you care about is often slow, rare or unmeasurable. Long-term satisfaction might take months to show; a purchase might happen days later on another device. So you choose a proxy label that is available soon and moves with the goal.

Good proxies are:

- **available quickly**, so models can be trained and evaluated without long delays;
- **correlated with the goal**, checked by seeing whether proxy gains predict business gains in experiments;
- **hard to game**, so the model can't improve the proxy while hurting the goal.

<div class="ex">
<div class="lab">Worked example</div>

**clicks versus watch time**

A video ranking model trained on clicks learned to promote misleading thumbnails: click-through rose and session time fell. Switching the objective to expected watch time, using weighted logistic regression where positive impressions are weighted by seconds watched, aligned the model better with what users valued. Covington et al. (2016) describe this shift.

</div>

## Combining several objectives

Real products care about more than one thing: clicks, purchases, returns, report rate. Two common patterns:

- **Multi-task models** predict several labels from shared features, then combine scores with weights: $\text{score} = w_1 P(\text{click}) + w_2 P(\text{purchase}) - w_3 P(\text{report})$. The weights are business decisions, tuned by experiment.
- **Constraints**: optimise one objective subject to limits on others (a minimum share of fresh content, a maximum ad load).

## Three layers of metrics

- **Business (online) metric**: what the company cares about, measured in A/B tests. Revenue, retention, fraud losses.
- **ML (offline) metric**: what you compute on a held-out set. AUC, log loss, NDCG, RMSE. It should predict the online metric.
- **Guardrails**: things that must not get worse. Latency, complaint rate, fairness across groups, diversity.

When offline gains stop predicting online gains, the offline metric or the proxy label is the first suspect. Keep a log of past experiments with both numbers side by side; over time it shows which offline metric you can trust, and by how much an offline gain usually shrinks once it meets real users. That ratio is useful when deciding whether a small offline improvement is worth an online test.

## Checking the framing

Before building, sanity-check the framing with three questions. Can a person do the task from the same inputs? If not, the model probably cannot either. Does the label exist for enough examples, at the right time? Would a perfect model on this task actually move the business metric? A perfect click predictor does not help if the bottleneck is that the catalogue lacks what users want. A framing that fails any of these checks is worth rethinking before you spend weeks on data pipelines.

<div class="take">
<div class="lab">Key takeaway</div>

turn a business goal into a precise prediction with a label and loss, treat the label as a proxy that must be fast, correlated and hard to game, and track business, offline and guardrail metrics together.

</div>
