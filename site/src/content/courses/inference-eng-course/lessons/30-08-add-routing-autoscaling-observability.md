---
id: "30.08"
course: "inference-eng-course"
module: "30"
slug: "add-routing-autoscaling-observability"
title: "Add Routing, Autoscaling & Observability"
keyIdeas: ["Route for cache locality with load bounds.", "Autoscale on queue and KV signals with headroom.", "Size a warm pool from cold-start time and burst profile.", "Dashboards, alerts and traces as in Module 24.", "Replay the capstone trace with bursts to verify."]
question: "What routing, autoscaling and observability setup fits the capstone's support-chat workload?"
answer: "Session-affinity or prefix-aware routing with load bounds, so conversation histories hit the cache; autoscaling on waiting requests and KV usage relative to measured capacity, with headroom for cold-start time and a small warm pool for bursts; and dashboards for traffic, TTFT/TPOT vs SLO, saturation and errors, with burn-rate alerts and request tracing including cached-token counts."
sources: [{"title": "KEDA", "url": "https://keda.sh/", "note": "Autoscaling on Prometheus metrics."}, {"title": "OpenTelemetry documentation", "url": "https://opentelemetry.io/docs/", "note": "Tracing."}]
tags: ["capstone"]
tools: ["Capstone"]
---

## Components

| Area | Choice |
| --- | --- |
| Routing | session affinity, with a least-loaded fallback |
| Autoscaling | on waiting tokens and KV usage; scale up fast, down slow |
| Warm capacity | sized from the largest burst within one cold-start window |
| Dashboards | traffic, SLO latencies, saturation, errors |
| Alerts | SLO burn rate, replica down, GPU errors |
| Tracing | per request, with token counts |

Session affinity matters a lot for this workload. Each conversation's history is a prefix, and the 3,000-token system prompt is shared by everyone. Without affinity, the prefix hit rate falls and the capacity gained in 30.05 is lost.

~~~dg-raw
%%widget spike The widget models traffic doubling, a harsher test than the 1.5× bursts in the definition. Try a 5-minute ramp and your measured cold start, then add warm replicas until minutes over capacity reach zero.
ramp = 5
cold = 3
warm = 0
u = 0.7
%%
~~~

## Verification

Replay the capstone trace, including bursts. Record SLO attainment per minute, replicas over time and total GPU-hours.

<div class="ex">
<div class="lab">Worked example</div>

**what the replay might show**

With hourly autoscaling and no warm replicas, a 1.5× burst at a 3-minute cold start causes two minutes where TTFT p95 exceeds 1.5 s. Attainment for that 5-minute window is 96%, below the 99% target.

Adding one warm replica per 10 running replicas brings it to 99.4%, at about 6% more GPU-hours.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

routing for cache locality, autoscaling on queue and KV, and SLO-focused observability turn an optimized replica into an operable service.

</div>
