---
id: "24.01"
course: "inference-eng-course"
module: "24"
slug: "inference-metrics-prometheus-grafana"
title: "Inference Metrics, Prometheus & Grafana"
keyIdeas: ["Collect engine metrics, GPU metrics and gateway metrics into Prometheus.", "Dashboards: traffic, latency vs SLO, throughput, saturation, errors.", "Use histograms for latencies and compute percentiles across replicas from buckets.", "Label by model, version and replica.", "Alert on SLO burn rate, not on every spike."]
question: "Name the four dashboard panels you would build first for an LLM service."
answer: "Request rate and error rate by model; TTFT and TPOT percentiles (p50, p95, p99) against SLO lines; token throughput (prompt and generation tokens per second); and engine saturation (running and waiting requests, KV cache usage, preemptions) per replica."
sources: [{"title": "Prometheus: Overview", "url": "https://prometheus.io/docs/introduction/overview/", "note": "Metrics collection."}, {"title": "Grafana documentation", "url": "https://grafana.com/docs/grafana/latest/", "note": "Dashboards."}, {"title": "vLLM: Metrics", "url": "https://docs.vllm.ai/en/latest/usage/metrics.html", "note": "Engine metrics."}]
tags: ["observability"]
---

## Sources

| Source | Examples |
| --- | --- |
| Gateway | requests, status codes, latency as the client sees it |
| Engine | TTFT, TPOT, queue length, KV usage, tokens in and out |
| GPU (DCGM) | memory, SM activity, power, clocks, errors |
| Host | CPU, RAM, network |

Each source answers a different question. The gateway says what users experienced. The engine says why. The GPU and host metrics say whether the hardware is healthy.

## A first dashboard

1. **Traffic:** requests per second and tokens per second, by model and version.
2. **Latency:** TTFT and TPOT at p50, p95 and p99, with the SLO drawn as a line.
3. **Saturation:** waiting requests, KV usage and preemptions per replica.
4. **Errors:** 4xx and 5xx rates, timeouts and aborted streams.

## Percentiles across replicas

Engines export latency as histograms: counts per bucket. To get a fleet-wide p95, add the bucket counts across replicas first, then compute the quantile.

```
histogram_quantile(0.95,
  sum by (le, model) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
```

Averaging each replica's p95 is wrong. One overloaded replica with a p95 of 8 s and seven healthy ones at 0.4 s average to 1.35 s, which hides that one user in eight is having a bad time.

~~~dg-raw
%%widget pct Latency distributions have long right tails. Raise the spread and watch how far p99 moves away from the median and the mean.
m = 400
s = 0.6
%%
~~~

## Alerts

Alert on **SLO burn rate**: how fast you are using up the error budget.

<div class="ex">
<div class="lab">Worked example</div>

**burn-rate alert**

The SLO is 99% of requests with TTFT under 2 s over 30 days, so the error budget is 1% of requests.

If 14.4% of requests miss the target for one hour, the service is burning budget 14.4 times faster than allowed. At that rate a 30-day budget is gone in about 2 days. That is worth paging someone.

A single 30-second spike at 5% misses barely touches the budget and should not page anyone.

</div>

Also alert on hard failures: a replica down, GPU errors, the error rate above a fixed threshold. Avoid alerting on single latency spikes.

## Deep dive: error budgets and burn rate

An SLO of ⟦m1068⟧ (say 99.5% of requests good) over a window ⟦m1069⟧ allows a budget of ⟦m1070⟧ bad requests. The burn rate is how fast the current error rate spends it:

⟦M1067⟧

~~~dg-raw
%%chart Time to spend a 30-day error budget at a constant burn rate.
x = 1, 2, 6, 14.4, 36
logx = 1
logy = 1
xlab = Burn rate
ylab = Hours until the budget is gone
Hours: 720, 360, 120, 50, 20
%%
~~~

A burn rate of 14.4 sustained for 1 hour spends 2% of a 30-day budget, which is why "14.4 over 1 hour and over 5 minutes" is a common page-worthy alert. A burn rate of 1 is exactly on budget and needs no alert at all.

<div class="take">
<div class="lab">Key takeaway</div>

measure traffic, latency against the SLO, saturation and errors, labeled by model and version. Alert on burn rate, not on single spikes.

</div>
