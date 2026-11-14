---
id: "08.05"
course: "ml-system-design"
module: "08"
slug: "common-mistakes-and-what-each-level-shows"
title: "Common mistakes and what each level shows"
keyIdeas: ["The most common mistakes are jumping to a model, ignoring labels and their delays, skipping evaluation design, forgetting serving constraints, and never mentioning monitoring.", "Junior candidates are expected to produce a sound end-to-end design; senior candidates add depth on trade-offs, scale and failure modes; staff-level candidates also reason about organisation, platforms, long-term effects and risk.", "Depth means numbers and mechanisms: how a feature is computed, what the latency budget is, which bias affects the metric and how you would correct it.", "Recovering well from a mistake or a hint from the interviewer is a positive signal."]
question: "A candidate answering \"design a fraud detection system\" spends 20 minutes comparing neural architectures and finishes without discussing labels, thresholds or monitoring. List the five most important things they missed and how a senior candidate would cover each in one or two sentences."
answer: "(1) Labels: chargebacks arrive weeks late and blocked transactions never get outcomes, so a senior candidate would train on matured labels, use early proxies and keep a small holdout for outcome data. (2) Operating point: the business cares about losses and customer friction, so they would set thresholds for block, step-up and allow from costs and analyst capacity, reporting precision and recall there. (3) Latency and serving: scoring happens during authorisation, so they would give a budget (for example 50 ms), name the real-time velocity features and a fallback when features are missing. (4) Adversaries and drift: fraudsters adapt, so they would monitor recall on new attack types, retrain frequently and keep fast rules for active campaigns. (5) Evaluation: a time-based split with point-in-time features, slices by merchant type and region, and a staged rollout with automatic rollback on block-rate deviations."
sources: [{"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Many rules correspond to common interview gaps."}, {"title": "Sculley et al., Hidden Technical Debt in Machine Learning Systems (2015)", "url": "https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html", "note": "The system concerns senior candidates are expected to raise."}]
tags: ["interview levels", "common mistakes"]
---

## Mistakes that cost the most

| Mistake | Why it hurts | Fix
| --- | --- | --- |
| Jumping to a model in minute two | shows no understanding of the problem | clarify first; state assumptions
| No label discussion | most real difficulty lives in labels | say where labels come from, their delay and bias
| Accuracy or AUC as the only metric | ignores operating point and business value | metric at the threshold, business metric, guardrails
| No baseline | cannot show the model is worth it | baseline first, then improvements
| Ignoring latency and scale | design cannot run | numbers: QPS, items per request, p99 budget
| Leakage in features | offline results would be fake | point-in-time features, time splits
| No serving story | model never reaches users | batch or online, stages, fallbacks
| No monitoring or retraining | system decays silently | drift, quality, triggers, owners
| Monologue | interviewer cannot steer | check in every few minutes
| Name-dropping | sounds shallow | explain mechanisms and trade-offs

## What each level shows

Expectations scale with level. The same prompt can be passed at junior level and failed at senior level with the same design, if the senior candidate lacks depth.

~~~dg-layers What a strong answer adds at each level. Each layer includes everything below it.
*Staff and above | platform and organisation: shared feature stores, reuse across teams, long-term effects, risk and governance
Senior | depth on trade-offs: numbers, failure modes, bias corrections, cost, rollout and monitoring designed in
Mid-level | a complete, sensible pipeline with metrics tied to the business and a clear serving plan
Junior | a correct framing, reasonable features and model, basic offline evaluation
~~~

**Junior or early career.** A correct end-to-end design: clear task framing, sensible features, a baseline and a reasonable model, offline evaluation with the right metric, and an awareness that serving and monitoring exist.

**Mid-level.** Everything above, plus concrete serving design (batch or online, latency budget), A/B testing, label sources and delays, leakage awareness and basic monitoring.

**Senior.** Depth and trade-offs: numbers for scale and cost, multi-stage pipelines, bias and feedback loops with corrections, calibration and thresholds, rollout and fallbacks, and what to build first. They anticipate failure modes before being asked.

**Staff and above.** System and organisational scope: which components should be shared platforms, how multiple teams' models interact (for example several rankers competing for one feed), long-term effects on users and the ecosystem, governance and risk, and a phased roadmap.

## What depth sounds like

Shallow: "We'd use a feature store for consistency."
Deep: "The 7-day order count would be computed by a streaming job and written to both the online store and the offline history with availability timestamps, so training uses point-in-time joins and serving reads the same definition. We'd log the served value with each prediction to measure any residual skew."

Shallow: "We'd monitor for drift."
Deep: "We'd alert on PSI above 0.2 sustained for three days on the top 10 features by importance, on approval rate by segment, and on first-payment default as an early proxy, since 90-day default labels arrive too late."

<div class="ex">
<div class="lab">Worked example</div>

**recovering from a hint**

A candidate proposed random train-test splits for a churn model. The interviewer asked, "Would that reflect how the model is used?" The candidate paused, explained that it would leak future information through user overlap and time, switched to a time-based split with point-in-time features, and noted that offline metrics would probably drop. The interviewer later cited this as a strong moment: the candidate listened, reasoned and corrected course.

</div>

## A final checklist

Before you finish any practice answer, check that you covered: the decision and error costs, the label and its delay, the metric at the operating point, a baseline, leakage, the latency budget, the serving mode, fallbacks, the online test, monitoring and retraining, and one feedback-loop or fairness risk. Eleven items; most candidates miss three or four.

## Using the case studies

The 21 case studies in modules 09 to 11 each end with a rubric listing what a strong design covers. Practise by answering a prompt aloud for 35 to 45 minutes before reading the model design, then score yourself against the rubric. Note which items you missed; the same few gaps tend to recur, and the final checklist above will catch them once you know what they are.

<div class="take">
<div class="lab">Key takeaway</div>

avoid the common gaps (labels, operating point, baseline, leakage, latency, serving, monitoring), match depth to the level you are interviewing for by giving numbers and mechanisms, and treat a hint as a chance to show reasoning and recovery.

</div>
