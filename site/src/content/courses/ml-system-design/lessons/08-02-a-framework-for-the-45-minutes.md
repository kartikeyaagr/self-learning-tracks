---
id: "08.02"
course: "ml-system-design"
module: "08"
slug: "a-framework-for-the-45-minutes"
title: "A framework for the 45 minutes"
keyIdeas: ["A repeatable structure: clarify (5 min), frame the ML task and metrics (5), data and labels (5), features (5), models (8), evaluation (5), serving (7), monitoring and iteration (5).", "Spend time in proportion to what is hard about the specific problem: labels for fraud, serving for feeds, evaluation for generative systems.", "At each step, state a choice, a reason and the main alternative. That pattern makes trade-offs visible without long digressions.", "End with risks and next steps; it shows you see the system as something that will run and change."]
question: "Using the framework, give a one-sentence answer for each of the eight steps for \"design a system that predicts whether a customer will cancel a subscription in the next 30 days\"."
answer: "Clarify: the score drives a retention offer sent by email or in-app to at most 5% of 10M subscribers each week; offers cost money; batch scoring weekly is acceptable. Frame: binary classification of cancellation within 30 days of a weekly prediction date; offline metric precision and recall in the top 5%, online metric retained revenue net of offer cost (ideally uplift, since we care who responds to offers). Data and labels: subscription events, usage logs, billing, support tickets; label from cancellations in the 30 days after each prediction date, with point-in-time features. Features: usage trend over 1, 4 and 12 weeks, payment failures, plan and tenure, support contacts, feature adoption, price changes. Models: baseline rules (usage dropped 50%); gradient-boosted trees; later an uplift model to target customers whose decision the offer changes. Evaluation: time-based backtest, precision at 5%, calibration, slices by plan and region; A/B test of the targeting policy against random or rule-based targeting. Serving: weekly batch job writes scores to the CRM, with a fallback to the previous week's scores. Monitoring and iteration: drift in usage features, score distribution, offer acceptance and retention by decile; retrain monthly, watch for feedback because treated customers' labels change."
sources: [{"title": "Google, Machine Learning Crash Course: production ML systems", "url": "https://developers.google.com/machine-learning/crash-course/production-ml-systems", "note": "Components of a production ML system."}, {"title": "Chip Huyen, Machine Learning Systems Design (free notes)", "url": "https://huyenchip.com/machine-learning-systems-design/toc.html", "note": "Project flow from objectives to deployment."}]
tags: ["interview framework", "time management"]
---

## Why a framework helps

Under time pressure it is easy to spend 30 minutes on the model and never reach serving or monitoring. A fixed structure keeps you moving, makes sure you cover what interviewers look for, and gives the interviewer an obvious place to steer you. Adapt the timing to the problem, but keep the order.

## The eight steps

| Step | Minutes | Key outputs
| --- | --- | --- |
| 1. Clarify requirements | 5 | decision, users, scale, latency, error costs, constraints
| 2. Frame the ML task | 5 | input, output, label, offline and online metrics, guardrails
| 3. Data and labels | 5 | sources, logging, label definition and delay, sampling, splits
| 4. Features | 5 | entity × freshness grid, key aggregates, leakage checks
| 5. Models | 8 | baseline, main model and loss, cascade if needed
| 6. Evaluation | 5 | offline metrics with slices, A/B test design, rollout
| 7. Serving | 7 | batch or online, pipeline stages, latency budget, fallbacks
| 8. Monitoring and iteration | 5 | drift, quality, retraining, feedback loops, risks

~~~dg-flow The 45-minute arc. The arrows are where you check in with the interviewer.
Clarify -(confirm)-> Frame -(confirm)-> Data + labels -> Features -> Models -(check depth)-> Evaluation -> Serving -(check time)-> *Monitoring + risks
~~~

## Step by step

**Clarify.** Ask the questions from 01.04 and write the answers as numbers. "10M users, 5% targeted weekly, batch is fine."

**Frame.** One sentence: "Binary classifier predicting X within Y days of a prediction date; optimise log loss; choose a threshold by capacity." Name the business metric and one or two guardrails.

**Data and labels.** Where examples come from, how labels are defined and when they arrive, how negatives are sampled, how you split by time.

**Features.** Group them, give three to five concrete strong features, and point out one leakage risk.

**Models.** Baseline first, then the main model with a reason, then what you would try if it is not enough. Mention the loss and calibration if probabilities matter.

**Evaluation.** Offline metric at the operating point, slices, and how you would run the online test.

**Serving.** Batch, online or hybrid; stages and their sizes; latency budget split; fallbacks.

**Monitoring and iteration.** What you watch, what triggers retraining, which feedback loops exist, and the top risks.

## The choice-reason-alternative pattern

At each step, say three things: what you choose, why, and what you would consider instead. "I'd start with gradient-boosted trees, because the features are tabular and we need fast iteration; if we add user histories as sequences, a neural model with a sequence encoder would be the next step." This shows judgement in one breath.

<div class="ex">
<div class="lab">Worked example</div>

**adapting the time split**

For a fraud prompt, a strong candidate spent ten minutes on labels and adversaries (delayed chargebacks, blocked transactions with no outcome, adaptive attackers) and only three on the model. For a feed-ranking prompt, the same candidate spent ten minutes on the serving pipeline and retrieval. The framework stayed the same; the emphasis followed the hard part of each problem.

</div>

## Going deep

Interviewers usually pick one or two areas to probe. Typical deep dives:

- the loss and training data for a ranking model;
- how to handle delayed or biased labels;
- latency and cost of a multi-stage pipeline;
- evaluating a generative system;
- dealing with cold start or feedback loops.

Have a ready answer with numbers for each of these.

## Finishing well

Use the last minutes to summarise the design in a few sentences, list the top two or three risks (label delay, drift, latency) and say what you would build first. Ending with a clear plan leaves a better impression than running out of time mid-sentence.

## Practising the framework

Time yourself on a few prompts and note where the minutes actually went. Most people overrun on models and run short on serving and monitoring. Practise saying each step's key outputs in two or three sentences, so that you can cover the whole arc quickly and still have time to go deep where the interviewer wants. Recording a mock and listening back is uncomfortable but shows filler and repetition quickly.

<div class="take">
<div class="lab">Key takeaway</div>

follow a fixed eight-step arc from clarifying to monitoring, weight time towards what is hardest in the specific problem, state a choice, a reason and an alternative at each step, and finish with risks and a build plan.

</div>
