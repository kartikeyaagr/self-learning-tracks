---
id: "25.04"
course: "inference-eng-course"
module: "25"
slug: "capacity-planning-from-benchmark-data"
title: "Capacity Planning From Benchmark Data"
keyIdeas: ["Inputs: forecast peak traffic and mix, per-replica SLO-safe capacity, target utilization, failure tolerance.", "Replicas = peak / (capacity x target utilization), rounded up, plus failure spares.", "Capacity must be measured on the forecast workload mix.", "Re-plan when the model, engine or mix changes.", "Track actual vs planned to calibrate."]
question: "You expect a peak of 120 requests/s. One replica's SLO-safe capacity for your workload is 9 req/s, and you plan to run at 70%. How many replicas do you need, and what else would you add?"
answer: "Planned per-replica load is 9 x 0.7 = 6.3 req/s, so 120 / 6.3 is about 19.05, round up to 20 replicas. Add N+1 or N+2 for failures (one or two replicas down), so 21 to 22, and check that the workload mix assumed in the benchmark matches forecast traffic."
sources: [{"title": "Google SRE Book: Software Engineering in SRE (capacity planning)", "url": "https://sre.google/sre-book/software-engineering-in-sre/", "note": "Intent-based capacity planning."}, {"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Producing per-replica capacity numbers."}]
tags: ["capacity"]
---

## The formula

```
replicas = ceil(peak_rate / (replica_capacity × target_util)) + failure_spares
```

## Inputs

| Input | Where it comes from |
| --- | --- |
| Peak rate and traffic mix | a traffic forecast |
| Replica capacity | the measured SLO-safe envelope (Modules 05 and 09) |
| Target utilization | your headroom policy, usually 60 to 80% |
| Failure spares | N+1 per zone, or a percentage |

## Worked example

| Item | Value |
| --- | --- |
| Peak | 120 requests/s |
| SLO-safe capacity per replica | 9 requests/s |
| Target utilization | 70% |
| Base replicas | ceil(120 ÷ 6.3) = 20 |
| Spares | 2 |
| Total | 22 replicas |

## Mix sensitivity

Capacity per replica depends on the traffic mix. If average prompt length doubles, capacity per replica may fall by a third.

<div class="ex">
<div class="lab">Worked example</div>

**the same peak, a different mix**

A new feature attaches documents to chats. Average prompts grow from 1,500 to 3,000 tokens.

Benchmarks show SLO-safe capacity drops from 9 to 6 requests/s per replica.

At the same peak and target: ceil(120 ÷ 4.2) = 29 base replicas, plus 2 spares, 31 in total. That is 9 more replicas, or 41% more GPUs, for the same request rate.

</div>

Plan for a few mix scenarios, and keep a capacity table per workload shape rather than one number.

## Deep dive: peak versus average

Capacity is bought for the peak, but paid for all day. With demand ⟦m1094⟧:

⟦M1093⟧

~~~dg-raw
%%chart A day of demand against a fixed fleet sized for the peak (illustrative).
x = 0, 3, 6, 9, 12, 15, 18, 21, 24
xlab = Hour of day
ylab = Requests per second
Demand: 12, 8, 10, 30, 42, 40, 36, 24, 12
Fixed fleet at 80% target*: 52, 52, 52, 52, 52, 52, 52, 52, 52
%%
~~~

Mean demand here is about 24 req/s against 52 of provisioned capacity: 46% average utilization. Autoscaling, batch jobs in the troughs, or a cheaper overflow tier are the ways to recover the other half.

<div class="take">
<div class="lab">Key takeaway</div>

capacity planning is peak divided by SLO-safe capacity at target utilization, plus spares. The measured envelope is what makes it trustworthy.

</div>
