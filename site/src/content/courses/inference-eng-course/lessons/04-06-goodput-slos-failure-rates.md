---
id: "04.06"
course: "inference-eng-course"
module: "04"
slug: "goodput-slos-failure-rates"
title: "Goodput, SLOs & Failure Rates"
keyIdeas: ["An SLO is a target on a metric, e.g. p95 TTFT under 500 ms and p95 TPOT under 50 ms.", "Goodput: throughput counting only requests that met every SLO.", "Raw throughput can rise while goodput falls, by overloading the server.", "Count failures (errors, timeouts, rejections) as misses, not as excluded samples.", "The useful capacity number is the highest load at which, say, 99% of requests meet the SLO."]
question: "A server does 5,000 tokens/s at 64 concurrency, but only 70% of requests meet the TTFT SLO. Another does 4,000 tokens/s with 99% meeting it. Which is better and how would you express it?"
answer: "The second, for a latency-sensitive product. Goodput counts only requests that meet the SLO: roughly 3,500 good tokens/s for the first vs 3,960 for the second. Report goodput (or max request rate at which 99% meet the SLO) rather than raw throughput."
sources: [{"title": "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving", "url": "https://arxiv.org/abs/2401.09670", "note": "Defines goodput for LLM serving."}, {"title": "Google SRE Book: Service Level Objectives", "url": "https://sre.google/sre-book/service-level-objectives/", "note": "SLIs, SLOs and error budgets."}]
tags: ["slo", "goodput"]
---

## From throughput to goodput

Throughput counts every token, even ones delivered too late to be useful. A server can raise its throughput by cramming in more requests while each one gets slower, until most users are unhappy. Products care about requests that arrive in time. **Goodput** counts only those:

goodput = work that met every SLO ÷ time

Work can be requests or tokens. A request counts only if it met all of its targets: TTFT, TPOT, E2E and no error.

## Defining SLOs

An SLO (service level objective) is a target for a metric at a percentile. Good SLOs come from the product:

- **Chat**: p95 TTFT under 500 ms, p95 TPOT under 50 ms.
- **Code completion**: p95 E2E under 300 ms for completions of up to 64 tokens.
- **Agents**: p95 E2E per tool call under 3 s.
- **Batch**: no latency target, but a completion deadline, such as all requests done within 24 hours.

An SLO also needs an attainment target: the fraction of requests that must meet it, such as 99%. "p95 TTFT under 500 ms" and "99% of requests with TTFT under 500 ms" are different statements, so be explicit.

## The capacity curve

As load increases, throughput and goodput first rise together. Past a knee, throughput flattens while goodput falls, because more and more requests miss their targets.

~~~dg-flow As load rises, throughput and goodput separate.
Low load -> Both rise together -> *Knee -> Throughput flat, goodput falls
~~~

Find the curve by sweeping request rate in open loop. At each rate, record throughput, SLO attainment and goodput.

| Rate (req/s) | Output tok/s | Met SLO | Goodput (tok/s) |
| --- | --- | --- | --- |
| 4 | 2,000 | 100% | 2,000 |
| 8 | 3,900 | 99% | 3,860 |
| 12 | 5,100 | 80% | 4,080 |
| 16 | 5,300 | 45% | 2,385 |

Goodput peaks near 12 requests per second, but only 80% of requests meet the SLO there. If the attainment target is 99%, the server's capacity is about 8 requests per second. That is the number to put in a capacity plan.

<div class="ex">
<div class="lab">Worked example</div>

**choosing between two configurations**

Configuration A (max batch 256) reaches 6,000 tokens per second, but its p95 TPOT crosses 50 ms at 9 requests per second.

Configuration B (max batch 96) reaches only 4,800 tokens per second, but stays within the SLO up to 11 requests per second.

By raw throughput, A wins by 25%. By SLO-safe capacity, B wins by 22%. For a chat product, B is the right choice.

</div>

## Failures are misses

Timeouts, 5xx errors, rejected requests and truncated streams all count as SLO misses. A common mistake is to compute latency percentiles only over successful requests. Under overload, a server that rejects half its traffic quickly looks fast, because only its fast requests are counted. Always report the error rate next to latency, and count errors against attainment.

## Using goodput to decide

- **Comparing configurations**: compare SLO-safe capacity at the same SLO, not peak throughput.
- **Setting autoscaling thresholds**: scale out before the knee, with enough lead time for a new replica to start.
- **Pricing and cost**: cost per million tokens should use the GPU-hours needed at SLO-safe capacity, not at peak.
- **Admission control**: when load exceeds capacity, rejecting a few requests early can keep goodput higher than letting everyone get slow.

## Deep dive: goodput in a toy model

Take the M/M/1 queue with service rate ⟦m412⟧ and an SLO that latency stay under ⟦m413⟧. The fraction of requests meeting the SLO and the goodput are

⟦M410⟧

Throughput keeps rising to ⟦m414⟧, but goodput peaks below it and then falls.

~~~dg-raw
%%chart Throughput and goodput against offered load (μ = 10 req/s, SLO d = 1 s).
x = 5, 6, 7, 8, 9, 9.5
xlab = Offered load (req/s)
ylab = Requests per second
Throughput: 5, 6, 7, 8, 9, 9.5
Goodput: 4.97, 5.89, 6.65, 6.92, 5.69, 3.74
%%
~~~

The peak sits near 7.8 req/s, 78% of capacity. Past it, every extra request makes others miss too. Real engines have a different curve, but the same shape: the right operating point is the peak of goodput, not of throughput.

## Deep dive: SLO attainment with several conditions

A request counts only if it meets every SLO at once. When the conditions are close to independent:

⟦M411⟧

Three conditions each met 98% of the time give only ⟦m415⟧ good requests. A "p98 on everything" target produces 6% bad requests.

<div class="take">
<div class="lab">Key takeaway</div>

optimize goodput under an explicit SLO, not raw throughput. A server's capacity is the highest load at which it still meets the SLO, and failures count as misses.

</div>
