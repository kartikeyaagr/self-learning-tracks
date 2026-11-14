---
id: "24.05"
course: "inference-eng-course"
module: "24"
slug: "timeouts-cancellation-retries-priorities"
title: "Timeouts, Cancellation, Retries & Priorities"
keyIdeas: ["LLM latency depends on output length; fixed short timeouts are wrong.", "For streams, use time-to-first-token and idle-between-tokens timeouts.", "Propagate cancellation from client to engine so work stops.", "Retry only idempotent failures, with backoff, jitter and a retry budget.", "Priorities let interactive traffic preempt batch traffic."]
question: "A client retries any request that takes over 10 seconds. What problem does this create for an LLM service?"
answer: "Long generations legitimately exceed 10 seconds, so the client cancels and resends them, and if cancellation does not propagate, the server keeps generating the old one too. Load multiplies exactly when the server is slow, causing a retry storm. Fix with timeouts tied to expected E2E (or streaming idle timeouts), propagated cancellation, retry budgets, and backoff."
sources: [{"title": "Google SRE Book: Addressing Cascading Failures", "url": "https://sre.google/sre-book/addressing-cascading-failures/", "note": "Retries and cascading failure."}, {"title": "AWS Builders' Library: Timeouts, retries and backoff with jitter", "url": "https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/", "note": "Retry design."}]
tags: ["reliability"]
---

## Timeouts that fit LLMs

| Timeout | Typical setting |
| --- | --- |
| Connect | short, a few seconds |
| Time to first token | based on the SLO, for example 10 s |
| Idle between chunks | for example 30 s |
| Total | max tokens × TPOT, plus margin |

A single total timeout of 30 s, common in web services, cuts off any response longer than about 1,000 tokens at 30 ms per token. Streaming needs separate timeouts for the first token and for gaps between chunks.

## Cancellation

When a client disconnects or times out, the gateway must cancel the upstream request, and the engine must abort it and free its KV.

~~~dg-seq Cancellation end to end.
Client > Gateway : disconnect
Gateway > Engine : abort request
Engine > Engine : free KV blocks
~~~

Test this end to end. A common bug: the gateway drops the client connection but keeps reading from the engine, so the engine keeps generating tokens nobody will see, holding KV the whole time.

## Retries

- Retry on connection failures, 429 and 503. Do not retry 4xx validation errors.
- Use exponential backoff with jitter.
- Use a retry budget: retries at most 10% of requests per client.
- Prefer retrying on a different replica.

<div class="ex">
<div class="lab">Worked example</div>

**how a retry storm forms**

A service is briefly overloaded and fails 30% of requests. Each client retries up to 3 times with no backoff.

Offered load becomes 1 + 0.3 + 0.09 + 0.027 ≈ 1.42 times normal, right when the service is already overloaded. More requests fail, so more are retried.

With a 10% retry budget, extra load is capped at 1.1 times, and with backoff the retries spread out instead of arriving together.

</div>

## Priorities

Tag requests as interactive, standard or batch. The scheduler serves higher priorities first. Under pressure, batch work is preempted or shed before interactive work.

## Deep dive: how retries multiply load

If each attempt fails with probability ⟦m1077⟧ and clients retry up to ⟦m1078⟧ times, each original request becomes

⟦M1076⟧

attempts. Overload raises ⟦m1079⟧, and the extra attempts raise load further: a feedback loop.

~~~dg-raw
%%chart Load amplification from retries against the per-attempt failure rate.
x = 0, 0.1, 0.25, 0.5, 0.75, 0.9
xlab = Failure probability per attempt
ylab = Attempts per request
Up to 1 retry: 1, 1.1, 1.25, 1.5, 1.75, 1.9
Up to 3 retries: 1, 1.11, 1.33, 1.88, 2.73, 3.44
Up to 5 retries: 1, 1.11, 1.33, 1.97, 3.29, 4.69
%%
~~~

At 50% failure with three retries, the server sees almost twice its real demand, exactly when it can least afford it. Exponential backoff with jitter spreads the retries in time; a retry budget (for example, retries at most 10% of requests) caps the multiplier at 1.1.

<div class="take">
<div class="lab">Key takeaway</div>

LLM timeouts must follow streaming behavior, cancellation must reach the engine, and retries need backoff and budgets to avoid storms.

</div>
