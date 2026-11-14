---
id: "25.05"
course: "inference-eng-course"
module: "25"
slug: "peak-load-failure-headroom-idle-capacity"
title: "Peak Load, Failure Headroom & Idle Capacity"
keyIdeas: ["Peaks set fleet size; averages set useful work; the gap is idle cost.", "Failure headroom (N+1) adds more standing capacity.", "Reduce idle cost: autoscale, backfill with batch work, multi-region shifting.", "Commitment pricing for the baseline, flexible capacity for peaks.", "Track average utilization of the fleet as a cost KPI."]
question: "Your traffic has a daily peak 4 times the overnight minimum. What strategies reduce the cost of capacity sitting idle overnight?"
answer: "Autoscale down overnight (keeping enough warm capacity for morning ramps); fill idle GPUs with deferrable batch jobs (offline inference, evals, embeddings) at lower priority; shift workloads across regions or time zones; and use cheaper commitment pricing for the baseline and on-demand for peaks."
sources: [{"title": "OpenAI Batch API guide", "url": "https://platform.openai.com/docs/guides/batch", "note": "Example of discounted deferred inference."}, {"title": "Google SRE Book: Handling Overload", "url": "https://sre.google/sre-book/handling-overload/", "note": "Headroom planning."}]
tags: ["capacity", "cost"]
---

## Where idle capacity comes from

| Source | Typical share |
| --- | --- |
| Daily trough | large |
| Headroom for bursts | 20 to 40% |
| Failure spares | 5 to 15% |
| Warm standby | small to moderate |

~~~dg-raw
%%widget diurnal Compare a static fleet with hourly autoscaling, then turn on backfill. Raise the peak-to-trough ratio to see how much idle capacity a sharp daily cycle creates.
pk = 3
%%
~~~

## Strategies

- **Autoscale** with a floor high enough to cover the morning ramp.
- **Backfill:** run batch inference, evals and synthetic data generation at low priority, preempted by interactive traffic.
- **Shift:** route traffic across regions in different time zones to flatten global peaks.
- **Pricing:** reserved capacity for the always-on baseline, and on-demand or spot capacity for peaks and batch work.

<div class="ex">
<div class="lab">Worked example</div>

**what utilization is worth**

A fleet costs \$200,000 per month. Its average utilization of SLO-safe capacity is 35%.

Autoscaling the night trough and backfilling with batch evals raises it to 55%.

Cost per useful token falls by 1 − 35 ÷ 55 ≈ 36%. The same traffic now costs about \$127,000 per month, or the same budget serves about 57% more traffic.

</div>

## The KPI

Track fleet average utilization of SLO-safe capacity as a cost metric, next to latency and errors. Raising it from 35% to 55% cuts cost per useful token by about a third.

## Deep dive: utilization multiplies cost

Cost per token at average utilization ⟦m1096⟧ is the full-load cost divided by ⟦m1097⟧:

⟦M1095⟧

~~~dg-raw
%%chart Cost multiplier against average utilization.
x = 0.2, 0.3, 0.4, 0.5, 0.7, 0.9
xlab = Average utilization
ylab = Cost per token, × full-load cost
Multiplier: 5, 3.33, 2.5, 2, 1.43, 1.11
%%
~~~

Raising average utilization from 30% to 50% cuts cost per token by 40%, more than most kernel optimizations. That is why idle capacity, not kernel speed, is usually the largest cost item in a real deployment.

<div class="take">
<div class="lab">Key takeaway</div>

peaks and headroom create idle capacity. Autoscale it away or fill it with batch work, and track fleet utilization as a cost metric.

</div>
