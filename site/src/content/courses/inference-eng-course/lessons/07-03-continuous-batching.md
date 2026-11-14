---
id: "07.03"
course: "inference-eng-course"
module: "07"
slug: "continuous-batching"
title: "Continuous Batching"
keyIdeas: ["Continuous batching = iteration-level scheduling with requests joining and leaving freely.", "The batch stays full as long as there is demand and KV memory.", "New requests start on the next step instead of waiting for a batch to end.", "Side effect: a big prefill in a step slows that step for every decoding user.", "Memory management (paged KV) is what lets the batch grow large."]
question: "Describe continuous batching in one paragraph and give its main effect on throughput and on TTFT."
answer: "Continuous batching keeps a running batch that changes every step: finished sequences leave immediately, and waiting requests join as soon as there is a slot and KV memory, with their prefill folded into the next step. This keeps the GPU's batch full, raising throughput several times over static batching, and lowers TTFT because new requests do not wait for a batch to finish."
sources: [{"title": "Anyscale: How continuous batching enables 23x throughput", "url": "https://www.anyscale.com/blog/continuous-batching-llm-inference", "note": "Measured gains."}, {"title": "Efficient Memory Management for LLM Serving with PagedAttention", "url": "https://arxiv.org/abs/2309.06180", "note": "Continuous batching plus paged memory."}]
tags: ["batching"]
---

## How continuous batching works

**Continuous batching** is iteration-level scheduling applied to a serving engine. The batch is never "started" or "finished". It is a set of running sequences that changes every step.

~~~dg-flow Continuous batching across three steps.
Step n: A, B, C decode -> C finishes -> Step n+1: A, B decode, D prefills -> *Step n+2: A, B, D decode
~~~

- A sequence that emits its stop token leaves the batch at once, and its KV blocks return to the pool.
- A waiting request is admitted as soon as there is room in the step's token budget and enough free KV blocks for its prompt.
- Its prompt is processed in the next step, either all at once or in chunks, alongside the running decodes.
- From the following step, it decodes one token per step like everyone else.

## The effect on the metrics

| Metric | Static batching | Continuous batching |
| --- | --- | --- |
| Slot occupancy | falls as requests finish | stays high |
| TTFT for a new arrival | waits for the current batch to end | one or two steps |
| Latency of a short request | tied to the longest in its batch | depends only on its own length |
| Throughput | moderate | often several times higher |
| Per-step time | constant within a batch | varies with the step's contents |

~~~dg-raw
%%widget cb Compare total steps and slot utilization for the same requests under both schedulers. Shuffle the lengths a few times.
%%
~~~

## The cost: prefill interference

The new behaviour has one cost. When a step includes a long prefill, that step takes much longer than a pure decode step, and every decoding user waits for it.

<div class="ex">
<div class="lab">Worked example</div>

**one long prompt in a busy batch**

100 sequences are decoding at 12 ms per step. A request with a 6,000-token prompt arrives.

If the engine prefills all 6,000 tokens in one step, that step takes about 12 ms + 6,000 × 16 GFLOP ÷ 600 TFLOP/s ≈ 170 ms.

Every one of the 100 users sees a 170 ms gap between two tokens. That is a stall, and it shows up as a spike in p99 ITL.

If arrivals like this come several times a second, the stalls become constant, and decode throughput falls too.

</div>

The fix is **chunked prefill**: split long prompts into pieces of a few hundred to a few thousand tokens and process one piece per step, mixed with the decodes. Each step stays bounded, at the price of a slightly longer TTFT for the long prompt. Module 10 covers it in detail. Most engines now enable it by default.

## What continuous batching needs

Continuous batching only pays off when many sequences fit in memory at once. If each request reserves a contiguous KV buffer for its maximum length, memory runs out after a few dozen requests and the batch never grows. Fragmentation makes it worse as sequences of different sizes come and go. Paged KV memory (Module 08) is what makes large continuous batches practical. The two ideas arrived together for this reason.

It also needs:

- **Variable-length attention kernels**, so sequences of different lengths can share a step without padding.
- **A fast scheduler**, since it now makes decisions every step.
- **Per-request sampling state**, since every sequence in the batch can have its own temperature, stop strings and length limit.

## Continuous batching and latency

Continuous batching improves throughput and TTFT, but it does not make decode faster for any one user. As the batch grows, each step takes a little longer, and every user's TPOT rises. The engine's settings decide where on the curve you sit:

- A low maximum batch size keeps TPOT low but caps throughput.
- A high maximum lets throughput climb but slows every stream.

## Deep dive: how often a long prefill stalls a stream

Each long prompt processed in one step creates one long gap in every running stream. A user generating at ⟦m502⟧ tokens per second while long prompts arrive at rate ⟦m503⟧ sees a share of stalled gaps of about

⟦M501⟧

At 12 ms steps and one long prompt per second, 1.2% of all gaps are stalls. That is more than 1%, so p99 ITL is no longer a decode step: it is the stall, 170 ms in the example above.

~~~dg-raw
%%chart p99 ITL against the arrival rate of 6,000-token prompts, without chunked prefill (12 ms decode steps, illustrative).
x = 0, 0.25, 0.5, 0.8, 1, 2
xlab = Long prompts per second
ylab = p99 ITL (ms)
Unchunked: 13, 14, 15, 30, 170, 172
Chunked at 2,048 tokens*: 13, 14, 15, 30, 54, 55
%%
~~~

The jump is sharp because p99 asks about the slowest 1%: once stalls are more than 1% of gaps, p99 equals the stall length. Chunking does not remove stalls. It caps their length.

<div class="take">
<div class="lab">Key takeaway</div>

continuous batching keeps the batch full every step by admitting and retiring requests as they arrive and finish. It raises throughput and cuts TTFT, but long prefills cause stalls that need chunking, and it only works well with paged KV memory.

</div>
