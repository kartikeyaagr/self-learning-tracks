---
id: "23.05"
course: "inference-eng-course"
module: "23"
slug: "warm-capacity-gpu-startup-time"
title: "Warm Capacity & GPU Startup Time"
keyIdeas: ["Cold start time decides how much headroom you need.", "Warm pools: replicas loaded and ready but lightly used, to absorb spikes.", "Trade: idle GPU cost vs SLO risk.", "Shorten startup to reduce the headroom you must pay for.", "Use admission control as the last line during spikes."]
question: "Your replicas take 5 minutes to become ready, and traffic can double in 2 minutes. What capacity strategy do you need?"
answer: "Reactive autoscaling alone will miss the SLO during spikes, since new capacity arrives after the spike. You need standing headroom (run below capacity), a pool of warm standby replicas (loaded and warmed but receiving little traffic), and faster startup (local weight caches, compile caches). Load shedding or degraded modes cover what remains."
sources: [{"title": "Google SRE Book: Handling Overload", "url": "https://sre.google/sre-book/handling-overload/", "note": "Planning for spikes."}, {"title": "Run:ai Model Streamer", "url": "https://github.com/run-ai/runai-model-streamer", "note": "Faster weight loading."}]
tags: ["autoscaling", "cold-start"]
---

## The race

~~~dg-flow A spike against startup time.
Traffic doubles in 2 min -> Autoscaler reacts at 30 s -> *Replica ready at 5.5 min -> SLO missed for about 3 min
~~~

If traffic rises faster than new replicas can start, there is a window where demand exceeds capacity. Queues build, TTFT climbs and the SLO is missed until new capacity arrives.

~~~dg-raw
%%widget spike Set a 5-minute cold start and a 2-minute ramp, and read the minutes over capacity. Then add warm standby replicas, or cut the cold start.
ramp = 2
cold = 5
warm = 0
u = 0.7
%%
~~~

## Options

| Strategy | Cost | Effect |
| --- | --- | --- |
| Static headroom | always paying for spare capacity | absorbs spikes at once |
| Warm standby pool | paying for idle loaded GPUs | takes over in seconds |
| Faster startup | engineering time | less headroom needed |
| Predictive scaling | forecasting work | capacity ready before known peaks |
| Load shedding | some rejected requests | protects the rest |

## Sizing the warm pool

Look at your traffic history. Find the largest increase within one cold-start window. That increase, in replicas, is how much warm or spare capacity you need.

<div class="ex">
<div class="lab">Worked example</div>

**sizing from a trace**

Cold start is 5 minutes. Over the last month, the largest 5-minute increase was from 22 to 31 replicas' worth of load.

You need about 9 replicas of capacity that are either running with spare room or warm and ready in seconds.

At 70% target utilization, 22 replicas of load already runs on 32 replicas, which gives 10 replicas of spare room. Here the target alone covers it. At 85%, 22 replicas of load runs on 26 replicas, leaving 4 spare, so you would need about 5 warm replicas in addition.

</div>

## Deep dive: sizing a warm pool from arrival statistics

With scale-to-zero, a request that arrives with no warm replica waits the full cold start. To serve bursts while new replicas start, keep ⟦m1061⟧ warm replicas such that the arrivals during one cold start fit:

⟦M1060⟧

For ⟦m1062⟧ req/s, ⟦m1063⟧ s and one replica serving 3 req/s: expected arrivals 120, 99.9th percentile about 154. One warm replica handles 180 in that minute: ⟦m1064⟧ suffices. Real traffic is burstier than Poisson, so use percentiles from your own trace, as in the example above.

<div class="take">
<div class="lab">Key takeaway</div>

if spikes are faster than startup, you pay for warm capacity or you miss the SLO. Faster startup is how you pay less.

</div>
