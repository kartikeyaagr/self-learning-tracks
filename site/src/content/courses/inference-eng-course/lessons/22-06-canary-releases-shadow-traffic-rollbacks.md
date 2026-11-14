---
id: "22.06"
course: "inference-eng-course"
module: "22"
slug: "canary-releases-shadow-traffic-rollbacks"
title: "Canary Releases, Shadow Traffic & Rollbacks"
keyIdeas: ["Offline evals first, then shadow, then canary, then full rollout.", "Shadow: duplicate traffic, compare outputs and performance, no user impact.", "Canary: small real traffic share, monitored, with automatic rollback.", "LLM-specific signals: output length, refusal rate, format errors, user feedback.", "Keep the old version warm until the new one is proven."]
question: "Compare shadow traffic and a canary for testing a new quantized model, and what you would watch in each."
answer: "Shadow traffic sends a copy of real requests to the new version without returning its responses to users; it is safe and shows performance and output differences, but not user reaction. A canary sends a small share of real users to the new version and returns its responses; it shows real quality signals (feedback, retries, task success) but carries risk. Watch latency, errors, output length, and quality metrics in both; in the canary, also user-facing signals and a fast rollback trigger."
sources: [{"title": "Google SRE Workbook: Canarying releases", "url": "https://sre.google/workbook/canarying-releases/", "note": "Canary practice."}, {"title": "Argo Rollouts", "url": "https://argoproj.github.io/argo-rollouts/", "note": "Progressive delivery on Kubernetes."}]
tags: ["releases"]
---

## Release ladder

~~~dg-flow Safe rollout for a model or engine change.
Offline evals -> Shadow traffic -> *Canary 1% -> 10% -> 50% -> 100% -> Retire old version
~~~

- **Offline evals:** quality benchmarks and your own task suites. Catches large regressions cheaply.
- **Shadow traffic:** copy a share of real requests to the new version and compare, without showing its output to users.
- **Canary:** send a small share of real users to the new version and watch signals before widening.
- **Retire:** only after the new version has run at 100% long enough to trust.

## What to watch

| Signal | Why |
| --- | --- |
| TTFT, TPOT, error rate | performance regressions |
| Output length distribution | cost and behavior shifts |
| Format errors (JSON, tool calls) | integration breakage |
| Refusal or empty response rate | shifts in safety tuning |
| User feedback, retries, task success | real quality |

## How long to hold each stage

A canary needs enough requests to see a real change. To notice a format-error rate going from 0.5% to 1%, you need a few thousand requests on the canary. At 1% of 100 requests/s, that is about 1 request/s, so plan for about an hour at that stage.

<div class="ex">
<div class="lab">Worked example</div>

**a regression only the canary caught**

A new quantized build passes offline evals within 0.3 points of the old one.

At 1% canary, tool-call JSON errors rise from 0.4% to 1.6%. A specific tool schema with nested arrays is affected, and the eval suite had no such schema.

An automatic trigger on format errors rolls back in minutes. Without the canary, every user of that tool would have seen failures.

</div>

## Rollback

- **Automatic triggers** on error rate, format errors or latency SLO breaches.
- **Keep the old version's replicas warm** during the canary, so rollback is a routing change that takes seconds, not a cold start that takes minutes.

## Shadow traffic caveats

- Shadow costs extra GPU time for the share you mirror.
- If responses can trigger actions (tool calls, emails, writes), strip tools or side effects from shadow calls.

## Deep dive: how long a canary must run

To detect an error-rate rise from ⟦m1021⟧ to ⟦m1022⟧ with false-alarm rate ⟦m1023⟧ and power ⟦m1024⟧, the canary needs about

⟦M1020⟧

requests. At canary traffic share ⟦m1025⟧ and total rate ⟦m1026⟧, that takes ⟦m1027⟧ seconds.

<div class="ex">
<div class="lab">Worked example</div>

**catching a doubling of a 0.5% error rate**

⟦m1028⟧, ⟦m1029⟧, one-sided 5% false alarms, 80% power: ⟦m1030⟧ requests.

At 50 req/s total and a 5% canary (2.5 req/s): about 11 minutes. At a 1% canary: about 53 minutes.

</div>

~~~dg-raw
%%chart Canary duration needed to catch a 0.5% → 1% error-rate rise, at 50 req/s total.
x = 1, 2, 5, 10, 25
logx = 1
logy = 1
xlab = Canary share of traffic (%)
ylab = Minutes
Minutes: 53, 27, 11, 5.3, 2.1
%%
~~~

Latency regressions need fewer requests than error rates, because every request carries a latency measurement, but they need the canary to see the same mix of prompt lengths as the baseline.

<div class="take">
<div class="lab">Key takeaway</div>

move model and engine changes through shadow and canary stages with LLM-specific signals, and keep rollback a routing switch.

</div>
