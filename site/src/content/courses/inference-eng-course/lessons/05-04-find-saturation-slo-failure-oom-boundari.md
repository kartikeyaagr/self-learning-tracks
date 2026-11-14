---
id: "05.04"
course: "inference-eng-course"
module: "05"
slug: "find-saturation-slo-failure-oom-boundaries"
title: "Find Saturation, SLO Failure & OOM Boundaries"
keyIdeas: ["Saturation: throughput plateau.", "SLO failure: the percentile you promise crosses its target.", "Memory boundary: KV exhausted, leading to queueing, preemption or OOM.", "These boundaries come at different loads. The lowest one sets capacity.", "Watch engine metrics: running vs waiting requests, KV usage, preemption count."]
question: "Name the three boundaries you look for when stressing one GPU and the signal for each."
answer: "Saturation: throughput stops increasing as load rises. SLO failure: the chosen percentile of TTFT or TPOT crosses its target, or goodput starts falling. Memory boundary: the engine starts preempting requests, queueing because KV blocks are exhausted, or crashes with out-of-memory. Each happens at a different load and you operate below the lowest."
sources: [{"title": "vLLM: Optimization and tuning", "url": "https://docs.vllm.ai/en/latest/configuration/optimization.html", "note": "Preemption and memory-related tuning."}, {"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "SLO attainment as the capacity criterion."}]
tags: ["capacity"]
---

## Three boundaries

As load rises on one GPU, three different limits appear. Capacity is the lowest of them, so you must find all three.

~~~dg-flow As load rises you meet these boundaries, often in this order.
Light load -> *SLO failure -> Saturation -> KV exhaustion or OOM
~~~

- **SLO failure.** Latency at the target percentile crosses the target. The server still works, and throughput may still rise, but users are unhappy.
- **Saturation.** Throughput stops rising with load. Any additional load just waits longer.
- **Memory exhaustion.** The KV pool fills and the engine starts preempting, or an allocation fails outright.

The order varies. With a tight latency SLO, SLO failure comes well before saturation. With long contexts, KV exhaustion can come first, before the GPU is computationally busy.

## Signals to watch

| Boundary | What you see |
| --- | --- |
| SLO failure | p95 TTFT or TPOT above target, goodput falls while throughput still rises |
| Saturation | output tok/s flat as load rises, waiting queue grows |
| KV exhaustion | KV usage near 100%, waiting queue grows, preemption counter rises |
| Out of memory | engine crash or failed allocation, often during a long prefill |

vLLM and SGLang expose running and waiting request counts, KV cache usage and preemption counters on their metrics endpoint. Scrape them every few seconds during every sweep, and store them with the benchmark results.

~~~dg-raw
%%widget queue Utilization near 1 is where saturation shows up as unbounded waiting. Compare 0.8 with 0.95.
s = 1200
r = 0.85
%%
~~~

## Preemption is a warning sign

When the KV pool runs out while sequences are still growing, the engine must free memory. It picks some running sequences, usually the most recently admitted, and either:

- **Recomputes**: discards their cache and later runs prefill again over prompt plus generated tokens.
- **Swaps**: copies their cache to CPU memory and later copies it back.

Either way, the preempted request stalls. Average throughput may look fine, because the GPU stays busy, but the tail latency of preempted requests can rise by seconds. A rising preemption counter means you are past the memory boundary even if no request has failed.

<div class="ex">
<div class="lab">Worked example</div>

**a hidden boundary**

A server at 140 concurrent requests shows 8,700 tokens per second and p50 TPOT of 16 ms. It looks healthy.

The preemption counter shows 300 preemptions per minute. Each preempted request repeats a 3,000-token prefill.

p99 E2E is 41 s, against a p50 of 9 s. The real memory-safe concurrency is about 110. At that level preemptions drop to zero and p99 E2E falls to 12 s, while throughput drops only 6%.

</div>

## Finding the real OOM

Engines reserve the KV pool at startup, so running out of KV blocks causes queueing or preemption, not a crash. True out-of-memory errors usually come from activation spikes the startup profiling did not cover:

- A prefill longer than any profiled during startup.
- A batch with an unusually large number of prompt tokens.
- A large sampling request, such as many logprobs or many parallel samples.
- Other processes allocating memory on the same GPU.

Test for these deliberately. Send the longest prompt you allow, at the highest concurrency you allow, with the largest `n` and `logprobs` values you accept. If the engine survives that, it will survive production traffic. If it crashes, lower `gpu_memory_utilization` or `max_num_batched_tokens`, or reject such requests at the gateway.

## Turning boundaries into limits

Once you know the boundaries, set protective limits below them:

- A maximum number of concurrent sequences per replica, below the preemption point.
- A maximum queue depth, so overload produces fast rejections instead of minutes of waiting.
- Input and output length caps at the gateway.
- An autoscaling trigger before the SLO boundary.

## Deep dive: three boundaries, three equations

For a fixed average context ⟦m443⟧, each boundary gives a maximum batch size:

⟦M440⟧

⟦M441⟧

The smallest one is the limit you hit first.

<div class="ex">
<div class="lab">Worked example</div>

**which wall comes first**

Llama 3 8B on an H100, average context 2,048 tokens (268 MB of KV per sequence), TPOT SLO 20 ms, 52 GB KV pool.

SLO: ⟦m444⟧.

KV capacity: ⟦m445⟧.

Compute: compute time grows 25 µs per sequence, memory time 96 µs per sequence. Compute never catches up at this context, so there is no compute boundary.

The SLO binds at 139. Raising max-num-seqs above that buys throughput the product cannot use.

</div>

~~~dg-raw
%%bars Maximum batch size from each boundary (same setup).
unit = sequences
TPOT SLO of 20 ms | 139 | flame
KV pool of 52 GB | 194 | c1
Compute | 0 | line | never binds at 2k context
%%
~~~

## Deep dive: what preemption costs

A preempted sequence that is recomputed repeats prefill for everything it held. At ⟦m446⟧ preemptions per second, each holding ⟦m447⟧ tokens:

⟦M442⟧

50 preemptions per minute of 2,500-token sequences is 2,083 tokens per second of repeated prefill. At 25 µs per token that is 52 ms of GPU per second, 5% of the machine, plus a TTFT-sized stall for every victim.

<div class="take">
<div class="lab">Key takeaway</div>

capacity is the lowest of three boundaries: the SLO, saturation and memory. Measure all three, treat rising preemptions as a warning, and test the worst-case request on purpose to find the real OOM.

</div>
