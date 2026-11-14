---
id: "05.05"
course: "ml-system-design"
module: "05"
slug: "a-b-testing-and-safe-rollout"
title: "A/B testing and safe rollout"
keyIdeas: ["Offline metrics screen candidates; **online experiments** decide. Randomise users (not requests) into control and treatment, and compare the business metric with a pre-registered analysis.", "Plan the test: primary metric, guardrails, minimum detectable effect, sample size and duration covering weekly cycles.", "Roll out in stages: shadow mode, canary, A/B test, then gradual ramp, with automatic rollback when guardrails break.", "Beware interference between users (marketplaces, social networks), novelty effects, and peeking at results. Each needs a specific design fix."]
question: "A ride-hailing company wants to test a new pricing model. Why might a standard user-level A/B test give a misleading answer, and what design would you use instead?"
answer: "Riders and drivers share one marketplace. If treatment riders see lower prices, they request more rides, using up drivers that control riders would otherwise have got, so control's experience gets worse and the measured difference is exaggerated (interference, a violation of the no-interference assumption). Driver behaviour also responds to aggregate demand. Use a design that randomises at a level where the market is mostly self-contained: switchback experiments that alternate the whole city between treatment and control in time blocks (for example 30 to 60 minutes, with washout periods), or geographic randomisation across comparable cities or zones. Analyse with block-level outcomes and account for carry-over between blocks. Keep guardrails on driver earnings, wait times and cancellation rates, and run long enough to cover weekly patterns."
sources: [{"title": "Kohavi, Tang and Xu, Trustworthy Online Controlled Experiments (book site)", "url": "https://experimentguide.com/", "note": "The standard reference on online experiments."}, {"title": "Johari et al., Always Valid Inference: Bringing Sequential Analysis to A/B Testing (2015)", "url": "https://arxiv.org/abs/1512.04922", "note": "Sequential tests that allow peeking."}, {"title": "Google SRE Workbook, Canarying Releases", "url": "https://sre.google/workbook/canarying-releases/", "note": "Staged rollout and automatic rollback."}]
practice: ["b-test-significance", "canary-mann-whitney", "msprt-always-valid-pvalues", "z-test-one-sample", "bootstrap-confidence-interval"]
tags: ["A/B testing", "rollout"]
tools: ["NumPy", "SciPy"]
---

## Why online tests decide

Offline metrics are estimates built on logged data and assumptions (05.02). The only way to know whether a model improves the business metric is to show it to real users and compare. A randomised controlled experiment does that while controlling for everything else that changes over time. The Machine Learning course covers the statistics (mlcore 09.04, mlmaths 13.06); this lesson covers design choices for ML systems.

## Designing the test

- **Unit of randomisation**: usually the user, so each person has a consistent experience. Request-level randomisation inflates sample size but breaks consistency and can bias results.
- **Primary metric**: one metric that decides the launch, such as bookings per user.
- **Guardrails**: metrics that must not get worse: latency, crash rate, complaints, revenue for other surfaces.
- **Minimum detectable effect and sample size**: the smallest effect worth detecting and the users needed to detect it with, say, 80% power. For a binary metric with base rate $p$, roughly $n \approx 16 p(1-p)/\delta^2$ per arm for a two-sided test at 5% significance.
- **Duration**: at least one or two full weeks to cover weekly cycles and let novelty effects fade.

Write this down before starting. Changing the metric after seeing results is how false wins get shipped.

## Staged rollout

~~~dg-flow A rollout ladder. Each rung can stop the release.
Offline gates -(pass)-> Shadow mode -(no errors, sane outputs)-> Canary 1-5% -(guardrails healthy)-> *A/B test -(primary metric wins)-> Ramp to 100% -(holdback)-> Long-term check
~~~

- **Shadow**: the new model scores live traffic but its outputs are not used; compare predictions and latency with production.
- **Canary**: a small share of traffic gets the new model; automatic rollback if error rates or guardrails break.
- **A/B test**: a properly powered experiment on the primary metric.
- **Ramp**: increase exposure in steps while monitoring.
- **Holdback**: keep a small control group for weeks to measure long-term effects.

## Pitfalls

| Pitfall | What happens | Fix
| --- | --- | --- |
| Peeking | stopping when p < 0.05 first appears inflates false positives | fixed horizon, or sequential tests designed for monitoring
| Novelty effect | users try anything new, then revert | run longer, look at effects over time
| Interference | treatment affects control (shared marketplace, social graph) | switchback, geo or cluster randomisation
| Sample ratio mismatch | arms differ in size more than chance allows | check assignment and logging before trusting results
| Too many metrics | some will be "significant" by chance | one primary metric, correction for others
| Feedback into training | treatment data trains the next model | tag data by arm, analyse separately

<div class="ex">
<div class="lab">Worked example</div>

**a test that measured the wrong thing**

A news app's new ranker raised clicks by 3% in a one-week test and was launched. Over the next month, daily active users fell slightly. A long-running 1% holdback showed the ranker increased clicks on sensational headlines but reduced return visits. The team made 28-day retention a guardrail for future ranking tests and kept a permanent holdback.

</div>

## Interleaving for rankers

For ranking changes, interleaving merges results from two rankers into one list and credits clicks to the ranker that contributed each item. It needs far fewer users than an A/B test to detect which ranker is better, making it a good screening step before a full test.

## Rollback

Every deploy needs a fast, tested rollback: the previous model version stays loaded or can be redeployed from the registry in minutes, and feature or schema changes are backward compatible. Automate rollback on guardrail breaches during the canary.

## Reading results

Look at the confidence interval, not just the p-value. A 0.5% lift with an interval of -0.1% to 1.1% is not a win yet. Check the result across major segments; a win driven entirely by one platform may hide a loss elsewhere. Then write a short decision record: what was tested, the result with its interval, the guardrails, and why the change was launched or not. These records become the evidence for how well offline metrics predict online ones.

<div class="take">
<div class="lab">Key takeaway</div>

decide launches with randomised online experiments planned in advance, roll out through shadow, canary, A/B and ramp stages with automatic rollback, and design around peeking, novelty, interference and sample-ratio problems.

</div>
