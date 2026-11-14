---
id: "09.03"
course: "inference-eng-course"
module: "09"
slug: "find-the-slo-safe-configuration-headroom"
title: "Find the SLO-Safe Configuration & Headroom"
keyIdeas: ["The SLO-safe maximum: highest load where e.g. 99% of requests meet the SLO.", "Find it with a fine-grained rate sweep around the knee.", "Headroom covers bursts, mix shifts, slow autoscaling and failures.", "Typical target: 60 to 80% of SLO-safe maximum.", "Recheck when model, engine version or traffic mix changes."]
question: "Why should you run production at less than the measured SLO-safe maximum load, and how much headroom is typical?"
answer: "Real traffic bursts, prompt mixes shift, and hardware or replicas fail, so running at the measured maximum means any spike breaks the SLO. Teams typically plan to run at 60 to 80% of the measured SLO-safe capacity, leaving room for bursts and for autoscaling to react."
sources: [{"title": "Google SRE Book: Handling Overload", "url": "https://sre.google/sre-book/handling-overload/", "note": "Planning for load beyond normal."}, {"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "SLO attainment as a capacity criterion."}]
tags: ["slo", "capacity"]
---

## Finding the SLO-safe maximum

Your capacity number is the highest request rate at which the SLO still holds. Find it precisely, because every capacity plan, cost estimate and autoscaling threshold depends on it.

1. **Coarse sweep.** Run open-loop rates in large steps, such as 2, 4, 8 and 16 requests per second, to find roughly where attainment collapses.
2. **Fine sweep.** Between the last passing and first failing rate, step by about 5%.
3. **Compute attainment** at each rate: the fraction of requests that met every SLO target, counting errors as misses.
4. **Pick the highest rate with attainment above your target**, such as 99%.
5. **Confirm** with a longer run at that rate, ideally 15 to 30 minutes, to catch slow effects.

## Why you should not run at that maximum

The SLO-safe maximum is measured under ideal conditions: a fixed workload, steady arrivals and healthy hardware. Production has none of these.

- **Bursts.** Poisson traffic at an average rate R regularly has short periods at 1.3 to 1.5 R. Real traffic is burstier than Poisson.
- **Workload drift.** A new feature that adds retrieval context can raise average prompt length by 3 times overnight, cutting capacity by half.
- **Failures.** If one of four replicas fails, the other three take 33% more load each.
- **Autoscaling lag.** A new GPU replica takes minutes to start: provisioning, pulling the image, loading weights, capturing graphs and warming up.
- **Deploys.** Rolling restarts take replicas out of service for a while.

## How much headroom

| Situation | Plan to run at |
| --- | --- |
| Stable traffic, fast autoscaling | about 80% of the SLO-safe maximum |
| Bursty traffic or slow scale-up | about 60 to 70% |
| Strict SLO, revenue-critical | about 50 to 60% |

The steepness of the knee also matters. Look at your sweep: if attainment falls from 99% to 80% over a 10% rise in rate, you need more headroom than if it falls gradually.

~~~dg-raw
%%widget queue Compare utilization 0.6, 0.8 and 0.95. The wait grows slowly at first, then very fast, which is why headroom matters.
s = 600
r = 0.8
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**sizing for failure**

One replica's SLO-safe maximum is 9 requests per second. Peak traffic is 24 requests per second.

At 80% of the maximum (7.2 per replica), you need 24 ÷ 7.2 ≈ 3.3, so 4 replicas.

If one of 4 fails, the remaining 3 each take 8 requests per second, still below 9. The SLO holds.

With only 3 replicas, normal load is 8 per replica, and losing one puts 12 on each of the other two, far past the knee.

</div>

## Write a capacity statement

Summarize the result in one line that anyone on the team can use:

"One H100 running config v14 serves 9 requests per second of the chat-v3 workload at 99% SLO attainment (p95 TTFT under 800 ms, p95 TPOT under 40 ms). We plan 7 requests per second per replica."

Every part of that sentence matters. Change the GPU, the config, the workload or the SLO, and the number changes.

## Revisit it

Rerun the capacity measurement when any of these changes:

- The engine version or config.
- The model or its quantization.
- The workload, as seen in production length distributions.
- The SLO.

A capacity statement older than the last major change is a guess.

## Deep dive: from a safe rate to a replica count

A capacity statement should survive three things: normal peaks, random bursts within the peak, and the loss of one replica. With per-replica SLO-safe rate ⟦m592⟧ and target utilization ⟦m593⟧:

⟦M590⟧

Random arrivals add short bursts on top of the peak. For Poisson traffic, the busiest second in a long period exceeds the mean by about

⟦M591⟧

<div class="ex">
<div class="lab">Worked example</div>

**sizing for 40 req/s at peak**

Per replica: SLO-safe at 8 req/s. At ⟦m594⟧, plan for 6.4 req/s per replica: ⟦m595⟧ replicas, 8 with N+1.

Bursts: ⟦m596⟧ req/s in the worst second of a busy period. Eight replicas offer up to 64 req/s of SLO-safe rate for a short time, so the burst is absorbed by the queue without breaking the SLO for long.

Losing one replica during the peak leaves 7 × 8 = 56 req/s, still above 40. The plan holds.

</div>

~~~dg-raw
%%chart Replicas needed for 40 req/s as target utilization changes (8 req/s safe per replica, plus one spare).
x = 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
xlab = Target utilization u
ylab = Replicas
Replicas: 11, 10, 9, 8, 7, 6
%%
~~~

The difference between running at 0.8 and running at 1.0 is two GPUs out of eight. Saving them puts p99 latency on the steep part of the curve from 04.02.

<div class="take">
<div class="lab">Key takeaway</div>

measure the SLO-safe maximum precisely, then plan to run well below it. Headroom covers bursts, workload drift, failures and scaling lag, and a one-line capacity statement makes the result usable by everyone.

</div>
