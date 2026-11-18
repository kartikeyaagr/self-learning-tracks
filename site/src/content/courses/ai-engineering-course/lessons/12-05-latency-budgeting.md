---
id: "12.05"
course: "ai-engineering-course"
module: "12"
slug: "latency-budgeting"
title: "Latency budgeting"
keyIdeas: ["Pick the metric the user feels: time to first visible token for streamed chat, end-to-end time for JSON a UI or agent waits on. End-to-end is roughly TTFT + (output tokens - 1) x TPOT, so output length usually dominates.", "Turn one p95 target into a budget per stage (gateway, guardrails, retrieval, rerank, model) and measure each stage from traces of the same requests. Per-stage p95s do not add up to the end-to-end p95.", "Parallelise independent work: a group costs the max of its calls, but the tail of a max grows with fan-out (k calls each at p95 meet together only 0.95^k of the time).", "Cheaper wins first: shorter outputs, prompt caching for TTFT, a small model for routing and rewriting, fewer rerank candidates. Reasoning models add thinking time before the first visible token.", "Every stage gets a timeout derived from its budget and a fallback; streams need a first-token timeout and an idle timeout, and cancellation must reach the provider."]
question: "Your RAG chat has a p95 time-to-first-token target of 1 second. Traces show p95s of 50 ms gateway, 120 ms input guardrail, 350 ms LLM query rewrite, 100 ms embed and search, 150 ms rerank and 600 ms model TTFT, all sequential. How do you get under budget, and what do you tell the product team about total response time?"
answer: "The stages sum to about 1,370 ms (the true end-to-end p95 is usually a little lower, since tails rarely coincide, so measure it from traces). Run the guardrail in parallel with rewrite and retrieval, which takes it off the path; cache the stable prompt prefix to cut TTFT from about 600 to 350 ms; rerank 20 candidates instead of 50 (150 to 70 ms); and skip the LLM rewrite on first turns, which have no history to resolve. First turns land near 590 ms and follow-ups near 920 ms. Give each stage a timeout with a fallback (skip rerank, answer without rewrite). Total time is set by output length: 300 tokens at 25 ms each adds about 7.5 s, so stream the answer, ask for shorter outputs and cap max_tokens; users see text after 0.6 s."
rubric: ["Adds stage latencies on the critical path and notes that per-stage p95s only approximate the end-to-end p95", "Moves independent stages into parallel and explains that only the slowest member counts", "Uses prompt caching or a shorter prompt to reduce TTFT, and drops or shrinks optional stages", "Separates TTFT from total time and identifies output length as the driver of total time", "Sets per-stage timeouts with fallbacks and recommends streaming"]
sources: [{"title": "OpenAI: Latency optimization", "url": "https://platform.openai.com/docs/guides/latency-optimization", "note": "Seven principles: faster tokens, fewer output and input tokens, fewer requests, parallelise, make users wait less, avoid the LLM."}, {"title": "Anthropic: Reducing latency", "url": "https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency", "note": "Measuring TTFT and baseline latency; model choice, prompt and output length, streaming."}, {"title": "Dean and Barroso: The Tail at Scale (2013)", "url": "https://research.google/pubs/the-tail-at-scale/", "note": "Why fan-out amplifies tail latency, and hedged requests to cut it."}]
practice: ["latency-budget-allocation", "ttft-tpot-itl-and-token-throughput", "p50-p95-and-p99-inference-latency", "optimize-throughput-and-cost-under-latency-slo", "dynamic-request-batching"]
---

## Which latency the user feels

Three numbers describe a model call (defined precisely in [TTFT, TPOT, ITL](lesson:ieng:04.01)): time to first token (TTFT, queueing plus prefill), time per output token (TPOT, decode speed) and end-to-end time,

$$
T_{\text{e2e}} \approx \text{TTFT} + (n_{\text{out}} - 1)\cdot \text{TPOT}
$$

Which one matters depends on what waits. In streamed chat the user starts reading at the first visible token, so TTFT is the number that feels like speed, as long as tokens then arrive faster than people read. When the output is JSON that a UI renders, or an agent step whose result feeds the next call, nothing happens until the last token, so end-to-end time is what counts. Reasoning models add a third phase: thinking tokens are generated before the first visible token, so 2,000 thinking tokens at 15 ms each is 30 seconds of blank screen. [One model call, end to end](lesson:aieng:01.03) shows why output length dominates; this lesson is about the whole request around the call.

Measure at the client. Provider-reported TTFT excludes your network, gateway, guardrails and retrieval, and those are often half the wait. Report p50 and p95 (or p99) from raw samples; never average percentiles across servers ([P50, P95, P99](lesson:ieng:04.05)).

## A budget per stage

A latency SLO such as "p95 time to first token under 1 s" is too coarse for anyone to act on. Split it into a budget per stage that one team owns, measure each stage from traces of the same requests, and look for the stage whose tail blows furthest past its share.

<div class="ex">
<div class="lab">Worked example</div>

**A RAG chat against a 1 s p95 TTFT budget (illustrative latencies)**

| Stage | p95 before | Change | p95 after | Budget |
| --- | --- | --- | --- | --- |
| Network + gateway | 50 ms | none | 50 ms | 50 ms |
| Input guardrail (classifier) | 120 ms | run beside retrieval | 120 ms, off the path | 450 ms for the group |
| Query rewrite (small LLM) | 350 ms | follow-up turns only | 0 or 350 ms | (in the group) |
| Embed + vector search | 100 ms | none | 100 ms | (in the group) |
| Rerank | 150 ms | 50 candidates to 20 | 70 ms | 100 ms |
| Model TTFT | 600 ms | cache the 8k-token prefix | 350 ms | 400 ms |
| **Critical path** | **1,370 ms** | | **590 ms first turn, 920 ms follow-up** | **1,000 ms** |

Before: everything is sequential, $50 + 120 + 350 + 100 + 150 + 600 = 1{,}370$ ms. After, on a first turn: $50 + \max(120, 100) + 70 + 350 = 590$ ms. On a follow-up the rewrite comes back: $50 + \max(120, 350 + 100) + 70 + 350 = 920$ ms.

Total time is a different budget. A 300-token answer at 25 ms per token adds $299 \times 25 \approx 7.5$ s after the first token. Halving the answer saves 3.7 s, more than every change in the table combined, which is why the first latency fix is usually "ask for shorter answers and cap `max_tokens`".

</div>

Two cautions about the table. First, the sum of per-stage p95s is not the end-to-end p95: slow guardrail calls and slow model calls are usually different requests, so the true p95 tends to be lower than the sum, and higher when stages share a cause (one overloaded region). Compute the end-to-end percentile from per-request traces and use stage numbers to explain it. Second, the budget only holds at the load you tested; queueing grows sharply near capacity, at the provider as much as in your own services ([Latency, throughput and cost](lesson:mlsd:06.03)).

~~~dg-seq One chat turn after the changes: the guardrail runs beside retrieval, and the answer streams.
Client > Gateway : question (50 ms)
Gateway > Guardrail : classify (120 ms, in parallel)
Gateway > Retriever : embed and search (100 ms, in parallel)
Retriever -> Gateway : 50 candidates
Gateway > Reranker : rerank, keep top 20 (70 ms)
Gateway > Model : prompt with cached prefix
Model -> Client : first token near 590 ms, then about 25 ms per token
~~~

## Parallel calls and the tail of a max

Independent work should run concurrently: guardrails beside retrieval, several retrieval queries at once, parallel tool calls, a moderation check on the previous chunk while the next streams. A parallel group costs the maximum of its members, so its tail is worse than any one member's. If $k$ independent calls each have latency distribution $F$, the group finishes by time $x$ with probability $F(x)^k$. Five calls that each meet their p95 all meet it together only $0.95^5 = 77\%$ of the time. For the group to meet a p95 target, each call needs

$$
F(x) = 0.95^{1/k}
$$

which is the 97.5th percentile for two calls and the 99th for five. Fan-out pays when it removes sequential steps; adding parallel calls for their own sake makes the tail worse.

Hedged requests, from the Tail at Scale paper, cut that tail: if a call has not returned by its p95, send a second copy and take whichever finishes first, for about 5% extra load. Hedge cheap, idempotent calls such as embedding and search. Hedging a model call doubles its cost and both copies are billed, so reserve it for TTFT stalls and cancel the loser.

## Smaller models first, and what they cost in latency

Small models are faster at both prefill and decode, so they suit every step on the path that is not the final answer: classifying intent, rewriting a query, choosing a tool, checking a guardrail. A cascade ([Cost budgeting](lesson:aieng:12.04)) saves money but adds latency for escalated requests, which pay for both calls in sequence. Once more than 5% of traffic escalates, the p95 is set by the escalated path, so budget for it. Predicted or speculative output and provider priority tiers can buy speed at a price; measure them on your own prompts.

## Perceived latency

Stream everything a person reads. During tool calls and retrieval, show what is happening ("Checking your order") rather than a spinner. Two features fight streaming: output guardrails that must see text before display (check sentence by sentence, at the cost of some latency, see [Guardrails](lesson:aieng:12.02)) and structured output, which a UI can only render once a field is complete (parse partial JSON as it streams, see [Structured outputs](lesson:aieng:12.01)).

## Timeouts and deadlines

Every stage needs a timeout derived from its budget and a fallback for when it fires: skip the rerank and use retrieval order, answer without the rewrite, return a cached answer. For model calls a single total timeout is wrong because long answers are legitimately slow; use a first-token timeout, an idle timeout between chunks and a total cap based on `max_tokens` ([Timeouts, cancellation, retries](lesson:ieng:24.05)). Pass a deadline down the chain so a retry deep in the stack does not start after the user has gone, and propagate cancellation to the provider: an abandoned stream keeps generating, and billing, until it is closed.

## Where this breaks

- **Long prompts.** TTFT grows with uncached input; retrieval that adds 10,000 tokens adds prefill time on every call.
- **Cold caches.** After a deploy or a quiet night, the first requests pay full prefill. Warm the prefix before shifting traffic.
- **Rate limits.** A request that waits in your client-side queue for token budget is slow even though every stage is fast ([Reliability](lesson:aieng:12.06)).
- **Retries inside the budget.** A retry with backoff can consume the whole budget; count retried requests in the latency percentiles.

## What interviewers probe

Expect to break a latency target into stages, argue which ones can run in parallel, explain why percentiles do not add, separate TTFT from total time, say what you would cut first, and design timeouts with fallbacks for a stream.

<div class="take">
<div class="lab">Key takeaway</div>

choose the latency the user feels, split its p95 target into per-stage budgets measured from traces, move independent stages into parallel, cut prompt and output length before buying speed, and give every stage a timeout with a fallback.

</div>
