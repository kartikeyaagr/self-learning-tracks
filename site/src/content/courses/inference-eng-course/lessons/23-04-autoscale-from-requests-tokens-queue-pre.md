---
id: "23.04"
course: "inference-eng-course"
module: "23"
slug: "autoscale-from-requests-tokens-queue-pressure"
title: "Autoscale From Requests, Tokens & Queue Pressure"
keyIdeas: ["Scale on signals that predict SLO breaches: queue, KV usage, concurrency vs capacity.", "GPU utilization and CPU usage are weak signals for LLM servers.", "Target utilization below capacity to leave headroom for cold-start time.", "Scale down slowly to avoid flapping; scale up fast.", "Combine reactive signals with schedules or forecasts for known peaks."]
question: "Why is GPU utilization a poor autoscaling signal for LLM servers, and what would you use?"
answer: "nvidia-smi utilization is near 100% whenever any kernel runs, even at low load, and does not track how close the server is to its SLO. Better signals are queue depth (waiting requests or tokens), KV cache usage, in-flight requests per replica relative to measured capacity, and SLO-related latencies (TTFT p95), possibly combined with predicted traffic."
sources: [{"title": "Kubernetes: Horizontal Pod Autoscaling", "url": "https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/", "note": "Autoscaling on custom metrics."}, {"title": "KEDA", "url": "https://keda.sh/", "note": "Event-driven autoscaling with Prometheus triggers."}]
tags: ["autoscaling"]
---

## Candidate signals

| Signal | Quality as a scaling signal |
| --- | --- |
| GPU utilization % | poor: reads near 100% long before the SLO is at risk |
| Requests per second | acceptable if the traffic mix is stable |
| In-flight requests per replica | good, when you know a replica's capacity |
| Waiting queue or waiting tokens | good leading indicator |
| KV cache usage | good for memory-bound workloads |
| TTFT p95 | lagging, but directly tied to the SLO |

GPU utilization in nvidia-smi only says a kernel was running. A decode engine with a small batch shows high utilization while having lots of spare capacity.

## Target

If one replica's SLO-safe capacity is 40 concurrent requests, set the target at about 28 (70%). The 30% gap covers traffic growth during the minutes it takes to start a new replica.

~~~dg-raw
%%widget queue Wait time grows sharply as utilization nears 100%. This is why scaling targets sit well below full load.
s = 800
r = 0.7
%%
~~~

## Policy

- **Scale up quickly** when the queue grows or KV usage passes a threshold. Add several replicas at once if the gap is large.
- **Scale down slowly**, only after several minutes of low usage, to avoid flapping and wasted cold starts.
- **Use scheduled scaling** for known daily peaks, such as business hours.

<div class="ex">
<div class="lab">Worked example</div>

**setting the target from cold-start time**

Traffic grows at most 15% per minute during normal peaks. A new replica takes 4 minutes to become ready.

In 4 minutes traffic can grow by about 1.15^4 ≈ 1.75 times. To survive that without new replicas, you would need to run at 1 ÷ 1.75 ≈ 57% utilization.

Cutting cold start to 2 minutes means growth of 1.32 times, so you can run at about 75%. Faster starts directly raise the safe target.

</div>

## Implementation

Export engine metrics to Prometheus, then drive the Kubernetes Horizontal Pod Autoscaler or KEDA with those custom metrics. Use the engine's own queue and KV metrics, not CPU or memory of the pod.

## Deep dive: headroom must cover the cold start

An autoscaler reacts after a lag of detection plus cold start, ⟦m1055⟧. If demand grows at ⟦m1056⟧ requests per second per second, it grows by ⟦m1057⟧ before new capacity arrives. The target utilization must leave room for that:

⟦M1054⟧

<div class="ex">
<div class="lab">Worked example</div>

**a morning ramp**

Traffic rises from 20 to 80 req/s over an hour: ⟦m1058⟧ req/s per minute. Detection takes 1 minute, cold start 4 minutes, so ⟦m1059⟧ min and demand grows by 5 req/s before a new replica serves.

At 40 req/s on 6 replicas of 8 req/s (48 req/s), spare capacity is 8 req/s: enough. With 5 replicas (40 req/s), there is no spare at all, and every ramp produces 5 minutes of queueing.

</div>

~~~dg-raw
%%chart Demand against capacity during a ramp, with a 5-minute scaling lag (illustrative).
x = 0, 5, 10, 15, 20, 25, 30
xlab = Minutes
ylab = Requests per second
Demand: 20, 25, 30, 35, 40, 45, 50
Capacity at 80% target: 32, 32, 40, 40, 48, 56, 64
Capacity at 100% target*: 24, 24, 32, 32, 40, 40, 48
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

autoscale on queue, KV and concurrency relative to measured capacity, with headroom sized to your cold-start time.

</div>
