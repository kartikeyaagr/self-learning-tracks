---
id: "04.02"
course: "inference-eng-course"
module: "04"
slug: "concurrency-request-rate-arrival-patterns"
title: "Concurrency, Request Rate & Arrival Patterns"
keyIdeas: ["Closed loop: fixed number of in-flight requests. Good for finding capacity, hides queueing.", "Open loop: requests arrive at a set rate regardless of completions. Matches real traffic.", "Poisson arrivals model independent users; bursts are realistic and stress the scheduler.", "Little's law links them: in-flight = arrival rate x average latency.", "Report results as a curve over load, not a single point."]
question: "Explain the difference between a closed-loop benchmark at concurrency 32 and an open-loop benchmark at 10 requests per second, and which one can reveal queueing collapse."
answer: "Closed loop keeps exactly 32 requests in flight: a new one starts only when one finishes, so load self-limits and latency cannot run away. Open loop sends requests on a schedule (10 per second, often Poisson) regardless of completions, like real users. If the server is slower than the arrival rate, the queue grows without bound and latency explodes. Only open loop reveals that collapse."
sources: [{"title": "Little's law", "url": "https://en.wikipedia.org/wiki/Little%27s_law", "note": "Concurrency = arrival rate x latency."}, {"title": "vLLM benchmarks directory", "url": "https://github.com/vllm-project/vllm/tree/main/benchmarks", "note": "Serving benchmark with request-rate and Poisson arrivals."}]
tags: ["load-testing"]
---

## Two ways to generate load

A load test sends requests to a server. How it decides when to send them changes the results completely.

- **Closed loop, with concurrency C.** C virtual users each send a request, wait for the full response, then send the next. The number of requests in flight never exceeds C.
- **Open loop, with rate R.** Requests arrive at an average of R per second, regardless of whether earlier ones have finished. Arrival times are usually random, drawn from a Poisson process.

## Why the difference matters

In a closed loop, the server controls the pace. If the server slows down, the load generator slows down too, because each virtual user waits longer before sending again. The server can never be overloaded. Latency rises with C, but only up to a point.

Real users do not wait for each other. A thousand people do not coordinate to send their next message only after someone else's answer finishes. They behave like an open loop. If requests arrive faster than the server can complete them, the queue grows every second, and TTFT climbs without limit.

~~~dg-flow Open-loop overload: the queue never drains.
Arrivals 10/s -> *Queue grows by 2/s -> Server completes 8/s -> TTFT rises every second
~~~

<div class="ex">
<div class="lab">Worked example</div>

**an overload that a closed loop would hide**

A server can complete 8 requests per second. Open-loop traffic arrives at 10 per second.

Each second the queue grows by 2. After one minute, 120 requests are waiting. At 8 per second, the newest one waits 15 s before it even starts.

A closed-loop test with 64 users on the same server would report a steady 8 requests per second and a stable latency of about 8 s. It never shows the runaway queue.

</div>

## Utilization and the knee

For a queueing system, utilization ρ is arrival rate divided by service capacity. As ρ approaches 1, waiting time does not rise gently. It explodes. For the simplest queue model (one server, random arrivals, random service times), the average time in the system is service time ÷ (1 − ρ):

- At 50% utilization, the wait is 2 times the service time.
- At 80%, it is 5 times.
- At 90%, it is 10 times.
- At 95%, it is 20 times.

LLM servers batch requests, which flattens this curve a lot, because more load raises the batch size instead of only the queue. But the knee still exists. Past it, a small increase in load produces a large increase in latency.

~~~dg-raw
%%widget queue Raise utilization past 0.8 and watch the wait time curve upward. That knee is why capacity plans keep headroom.
s = 800
r = 0.7
%%
~~~

## Little's law

For any stable system, the average number of requests in the system equals the arrival rate times the average time each spends in it:

L = λ × W

This holds regardless of arrival pattern or scheduling. It is useful in three ways:

- **Translating between loop types.** A closed-loop test at concurrency 64 with 8 s average latency is serving 64 ÷ 8 = 8 requests per second.
- **Sizing memory.** At 20 requests per second and 6 s average latency, you have 120 requests in flight. Each needs KV cache, so you need room for 120 sequences plus headroom.
- **Checking results.** If your tool reports 10 requests per second, 5 s average latency and only 20 requests in flight, something is wrong.

## Arrival patterns to test

- **Constant rate.** Requests evenly spaced. Easy to reason about, but unrealistic.
- **Poisson.** Random arrivals with exponential gaps. The standard model for many independent users. Gaps are irregular, so small bursts happen naturally.
- **Bursty.** A spike to 3 to 5 times the normal rate for 30 to 60 seconds. Tests queueing, admission control and how fast latency recovers afterwards.
- **Diurnal ramp.** A slow rise and fall over hours, as traffic follows the working day. Tests autoscaling.
- **Replayed traces.** Real production timestamps, replayed at 1× or faster. The most realistic, if you have them.

## Which mode to use when

- Use **closed loop** to find the maximum throughput of a configuration, and to compare engines on equal footing.
- Use **open loop** to see how latency behaves under real traffic, to find the highest rate that meets an SLO, and to test overload behaviour.
- Never compare a closed-loop result from one engine with an open-loop result from another.

## Deep dive: the response time law for closed loops

A closed-loop test with ⟦m361⟧ virtual users, each waiting ⟦m362⟧ seconds of "think time" between requests, settles where

⟦M358⟧

with ⟦m363⟧ the request rate and ⟦m364⟧ the mean response time. The equation runs both ways: if the server slows down, ⟦m365⟧ rises and ⟦m366⟧ falls by itself. A closed loop cannot overload the server, which is why it cannot find the overload point.

An open loop fixes ⟦m367⟧ instead. If ⟦m368⟧ exceeds the service rate ⟦m369⟧, the queue grows without limit:

⟦M359⟧

~~~dg-raw
%%chart Queue length during a 60 s test against a server that completes 8 requests/s.
x = 0, 10, 20, 30, 40, 50, 60
xlab = Seconds into the test
ylab = Requests waiting
Open loop, 10 req/s: 0, 20, 40, 60, 80, 100, 120
Open loop, 9 req/s: 0, 10, 20, 30, 40, 50, 60
Closed loop, 64 users*: 0, 0, 0, 0, 0, 0, 0
%%
~~~

## Deep dive: the knee, quantified

For a single queue with random arrivals and service (M/M/1), response time is exponentially distributed with rate ⟦m370⟧. Its mean and tail are

⟦M360⟧

with ⟦m371⟧ and ⟦m372⟧. A batching engine is not M/M/1, but measured curves have the same shape: flat, then a wall. The tail hits the wall first.

~~~dg-raw
%%chart Response time in multiples of the service time S (M/M/1).
x = 0.5, 0.6, 0.7, 0.8, 0.9, 0.95
logy = 1
xlab = Utilization ρ
ylab = Multiples of S
Mean: 2, 2.5, 3.33, 5, 10, 20
p99: 9.2, 11.5, 15.4, 23, 46, 92
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**why "80% busy" is a common target**

Service time 2 s. At ⟦m373⟧: mean response 10 s, p99 46 s. At ⟦m374⟧: mean 20 s, p99 92 s.

Going from 80% to 90% busy adds 12.5% throughput and doubles every latency number. The last few percent of utilization are the most expensive.

</div>

## Deep dive: Little's law as a test oracle

⟦m375⟧ needs no assumptions about arrivals or scheduling. That makes it a check on your measurements: sample the in-flight count every second, and compare its mean with throughput × mean latency.

<div class="ex">
<div class="lab">Worked example</div>

**catching a broken timer**

Measured: 12 req/s, mean E2E 4.0 s, so Little's law predicts 48 in flight. The server's metrics show a mean of 71 running plus waiting.

A 48% gap is not noise. Cause: the client measured E2E from when its HTTP connection opened, but requests sat in the client's own connection pool for about 1.9 s first. True ⟦m376⟧ s.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

closed-loop load finds capacity, while open-loop load shows how latency behaves under real traffic. Utilization near 1 causes queueing to explode, and Little's law connects rate, latency and requests in flight.

</div>
