---
id: "24.03"
course: "inference-eng-course"
module: "24"
slug: "distributed-request-tracing"
title: "Distributed Request Tracing"
keyIdeas: ["Traces follow one request across services: gateway, retrieval, router, engine.", "Propagate trace context through every hop, including the engine.", "Record LLM attributes: tokens in, cached tokens, tokens out, TTFT, finish reason.", "Sample traces; keep all slow or failed ones.", "Traces explain individual slow requests that metrics only show in aggregate."]
question: "What spans would a trace of one RAG chat request contain, and which attributes are specific to LLM serving?"
answer: "Spans for gateway auth, retrieval, reranking, prompt assembly, routing, engine queue wait, prefill, decode, and streaming back, plus tool calls if any. LLM-specific attributes: model and version, prompt tokens, cached prompt tokens, output tokens, TTFT, TPOT, replica id, preemptions, and finish reason."
sources: [{"title": "OpenTelemetry documentation", "url": "https://opentelemetry.io/docs/", "note": "Tracing standard and SDKs."}, {"title": "OpenTelemetry semantic conventions for generative AI", "url": "https://opentelemetry.io/docs/specs/semconv/gen-ai/", "note": "Standard attribute names for LLM calls."}]
tags: ["tracing"]
---

## A trace

A trace is the record of one request's path through every service, split into timed spans.

~~~dg-seq Spans in one RAG request.
Gateway > Retriever : retrieve (40 ms)
Gateway > Router : route (2 ms)
Router > Engine : queue (120 ms)
Engine > Engine : prefill (300 ms)
Engine -> Gateway : decode (3.1 s)
~~~

TTFT in this trace is 40 + 2 + 120 + 300 = 462 ms. Without the trace, all you would know is "TTFT p95 is 460 ms". With it, you can see that a quarter of that is queueing, which routing or capacity can fix, and two thirds is prefill, which caching or a faster kernel can fix.

## LLM-specific attributes

| Attribute | Example |
| --- | --- |
| model, version | chat-v42 |
| prompt tokens and cached tokens | 5,200 and 4,800 |
| output tokens | 310 |
| ttft_ms, tpot_ms | 470, 10 |
| replica | gpu-pod-7 |
| finish reason | stop, length or abort |

## Sampling

Tracing every request is expensive at scale. Keep:

- a small random sample, for example 1%, to see normal behavior,
- every request that was slow, errored or aborted (tail-based sampling).

Tail-based sampling decides after the request finishes, so the collector must buffer spans briefly. It is worth it: the slow requests are the ones you need to explain.

## Privacy

Do not put prompts or outputs in trace attributes by default. Store ids and token counts. If you need content for debugging, store it separately with access control and a short retention period.

## Deep dive: the cost of tracing everything

Trace storage grows with request rate, spans per request and bytes per span:

⟦M1071⟧

At 500 req/s, 20 spans of 1 KB each: 864 GB per day. Tail-based sampling keeps every error and every request over the latency SLO, plus a small random share of the rest:

⟦M1072⟧

With 2% bad requests and ⟦m1073⟧, about 3% of traces are kept: 26 GB per day, with every interesting trace still there.

<div class="take">
<div class="lab">Key takeaway</div>

traces show where one request's time went across services. Add LLM-specific attributes and keep every slow trace.

</div>
