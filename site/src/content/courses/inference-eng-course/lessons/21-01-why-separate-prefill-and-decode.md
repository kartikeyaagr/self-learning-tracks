---
id: "21.01"
course: "inference-eng-course"
module: "21"
slug: "why-separate-prefill-and-decode"
title: "Why Separate Prefill and Decode"
keyIdeas: ["Colocated prefill and decode interfere: one phase's work delays the other.", "Chunked prefill reduces but does not remove interference.", "Disaggregation runs prefill on one pool and decode on another, transferring KV between them.", "Each pool gets its own parallelism, batch policy and scaling.", "Cost: KV transfer, more complex scheduling and routing."]
question: "Give two reasons to run prefill and decode on separate GPU pools."
answer: "They interfere when colocated: long prefills stall decode steps (ITL spikes) and decode batches slow prefills, so it is hard to meet both TTFT and TPOT SLOs. And they have different resource profiles (prefill compute bound, decode memory bound), so separate pools can use different parallelism, batch sizes or even hardware, and be scaled independently to match the traffic's ratio of prompt to output tokens."
sources: [{"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "Disaggregation for goodput."}, {"title": "Splitwise", "url": "https://arxiv.org/abs/2311.18677", "note": "Phase splitting across machines."}]
tags: ["disaggregation"]
---

## The problem with colocation

In a normal engine, prefill and decode share the same GPUs and often the same batch. The two phases want opposite things.

| | Prefill | Decode |
| --- | --- | --- |
| Work per step | thousands of tokens of one prompt | one token for each of many sequences |
| Bound by | compute | memory bandwidth |
| Step length | tens to hundreds of ms | about 10 to 40 ms |
| Best batch | a token budget that fills the GPU | as many sequences as KV allows |
| SLO it drives | TTFT | TPOT and inter-token latency |

When they share a step, each hurts the other:

- **A prefill slows every decode in the batch.** A 2k-token prefill chunk adds about 30 to 60 ms to an 8B model's step, so every user streaming at that moment sees a stall.
- **Decodes slow the prefill.** A large decode batch leaves less of the token budget for prompt chunks, so TTFT grows.
- **Tight TTFT and tight TPOT together force compromise.** Small prefill chunks protect TPOT but raise TTFT. Large chunks do the opposite.

<div class="ex">
<div class="lab">Worked example</div>

**the stall a user sees**

An engine runs 64 decoding sequences at about 12 ms per step. A 6k-token prompt arrives and is split into three 2k chunks.

Each step that carries a chunk takes about 45 ms instead of 12 ms. For those three steps, all 64 users get tokens every 45 ms.

Their p99 inter-token latency jumps from about 15 ms to about 45 ms, even though average TPOT barely moves.

</div>

## The idea

Run the two phases on different GPUs.

~~~dg-flow Disaggregated serving.
Request -> *Prefill pool (compute bound) -(KV transfer)-> Decode pool (memory bound) -> Streamed tokens
~~~

## Benefits

| Benefit | Why |
| --- | --- |
| No interference | the phases never share a step |
| Phase-specific tuning | tensor parallelism for prefill latency, very large batches for decode |
| Independent scaling | the ratio of prefill to decode GPUs follows traffic |
| Hardware choice | compute-rich GPUs for prefill, memory-rich GPUs for decode |

## Costs

- **KV transfer:** gigabytes per long request, which needs fast links.
- **Two pools to size, route and scale,** plus a coordinator.
- **Light load:** with few requests, colocated serving with chunked prefill is often just as good and simpler.

## Deep dive: how much colocated prefill slows decode

If prefill chunks take a share ⟦m989⟧ of every step's time on a colocated GPU, each decoding user's TPOT is inflated by

⟦M988⟧

~~~dg-raw
%%chart TPOT inflation against the share of step time spent on prefill.
x = 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
xlab = Prefill share of each step φ
ylab = TPOT multiple
Inflation: 1, 1.11, 1.25, 1.43, 1.67, 2, 2.5
%%
~~~

A prompt-heavy workload (RAG, agents) can spend half of its GPU time on prefill, doubling TPOT for everyone. Disaggregation removes ⟦m990⟧ from the decode GPUs entirely, at the cost of moving KV between pools.

<div class="take">
<div class="lab">Key takeaway</div>

disaggregation removes prefill-decode interference and lets each phase run its own way, at the cost of KV transfer and extra complexity.

</div>
