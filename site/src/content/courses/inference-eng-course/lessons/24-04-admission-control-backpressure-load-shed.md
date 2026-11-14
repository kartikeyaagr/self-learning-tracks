---
id: "24.04"
course: "inference-eng-course"
module: "24"
slug: "admission-control-backpressure-load-shedding"
title: "Admission Control, Backpressure & Load Shedding"
keyIdeas: ["Admission control: accept a request only if it can likely meet its SLO.", "Backpressure: signal upstream to slow down (429, Retry-After).", "Load shedding: drop the lowest-priority work first.", "Estimate request cost by prompt length and max tokens.", "Without these, overload collapses goodput for everyone."]
question: "Why is rejecting some requests early better than accepting all of them during overload?"
answer: "Accepted requests beyond capacity queue up, so every request's latency grows, many time out after consuming GPU time, and goodput collapses. Rejecting excess requests quickly (with a retryable status) keeps latency within SLO for the admitted ones and gives clients a clear signal to back off or retry elsewhere."
sources: [{"title": "Google SRE Book: Handling Overload", "url": "https://sre.google/sre-book/handling-overload/", "note": "Load shedding and graceful degradation."}, {"title": "Google SRE Book: Addressing Cascading Failures", "url": "https://sre.google/sre-book/addressing-cascading-failures/", "note": "How overload spreads."}]
tags: ["reliability", "overload"]
---

## Overload without control

~~~dg-flow Unbounded queue under overload.
Arrivals above capacity -> Queue grows -> *Latency rises for everyone -> Timeouts after GPU work is done -> Goodput collapses
~~~

When arrivals exceed capacity and nothing says no, the queue grows without limit. Every request waits longer. Eventually most requests wait past their client timeout. The GPUs keep working on requests whose users have already given up, so goodput, the work that actually reaches users in time, falls toward zero.

~~~dg-raw
%%widget shed Move the offered load above 1× and compare the two lines. Then shorten the client timeout.
x = 1.3
T = 10
%%
~~~

## Controls

| Control | Where | Example |
| --- | --- | --- |
| Queue limit | engine or gateway | a maximum number of waiting tokens |
| Admission check | gateway | reject if the estimated wait exceeds the SLO |
| Priority shedding | gateway | drop free-tier traffic first |
| Max-token caps | gateway | bound the cost of any single request |
| Rate limits | per tenant | tokens per minute |

## Signals for admission

- Waiting tokens per replica, which predicts the wait for prefill.
- KV usage, which predicts preemption.
- Recent TTFT compared with the SLO.

<div class="ex">
<div class="lab">Worked example</div>

**an admission rule**

A replica prefills about 20,000 tokens per second. The TTFT SLO is 2 s.

If the replica already has 30,000 prompt tokens waiting, a new request would wait about 1.5 s before its own prefill starts. With a 4,000-token prompt, its TTFT would be about 1.7 s, just inside the SLO.

If 45,000 tokens are waiting, the wait alone is 2.25 s. The gateway tries another replica, and if all are this full, returns 503 with a Retry-After header.

</div>

## Client side

Return 429 or 503 with a Retry-After header. Clients should retry with exponential backoff and jitter, not immediately.

## Deep dive: an admission limit from the TTFT SLO

A request admitted now waits behind every prompt token already queued. With prefill capacity ⟦m1075⟧ tokens per second, keeping its TTFT under the SLO requires

⟦M1074⟧

At 40,000 prefill tokens/s and a 2 s TTFT SLO, about 80,000 tokens of queued prompts is the limit. Admitting more only creates requests that will miss the SLO anyway, while making the ones behind them miss it too. Rejecting early (HTTP 429 with a retry hint) is kinder than accepting and timing out.

~~~dg-raw
%%chart Goodput against offered load, with and without admission control (illustrative).
x = 50, 75, 100, 125, 150, 200
xlab = Offered load (% of capacity)
ylab = Goodput (% of capacity)
With admission control: 50, 75, 92, 95, 95, 94
Without: 50, 75, 88, 60, 25, 5
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

under overload, saying no early protects everyone else. Bound queues, shed low-priority work, and tell clients when to retry.

</div>
