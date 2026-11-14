---
id: "05.03"
course: "inference-eng-course"
module: "05"
slug: "sweep-prompt-output-length"
title: "Sweep Prompt & Output Length"
keyIdeas: ["Prompt length drives TTFT and KV memory per sequence.", "Output length drives total decode steps and how long a sequence holds its KV.", "Longer contexts make each decode step slower, since attention reads more KV.", "Sweep both on a grid to map the GPU's envelope.", "Report throughput per configuration, but note it measures different things at each point."]
question: "How does doubling prompt length affect TTFT, max concurrency and decode speed, and why?"
answer: "TTFT roughly doubles (more at long context because attention prefill is quadratic). Max concurrency roughly halves because each sequence holds twice the KV cache. Decode TPOT rises somewhat because each step's attention reads twice as much KV per sequence; at short context the weight read dominates so the effect is small, at long context it becomes large."
sources: [{"title": "Splitwise", "url": "https://arxiv.org/abs/2311.18677", "note": "Phase characteristics across lengths."}, {"title": "vLLM benchmarks directory", "url": "https://github.com/vllm-project/vllm/tree/main/benchmarks", "note": "Random dataset with controllable lengths."}]
tags: ["workloads"]
---

## Two axes that change everything

Prompt length and output length shift the bottleneck from one part of the system to another. A single benchmark at one pair of lengths cannot tell you how a server will behave on another workload. So map both axes as a grid. A good starting grid is input lengths of 256, 1k, 4k and 16k tokens, and output lengths of 64, 256 and 1,024 tokens, run at a fixed, moderate concurrency.

## What input length changes

- **TTFT.** Prefill work grows with prompt length, linearly at first and faster than linearly at long lengths as attention's quadratic term grows.
- **KV per sequence.** Grows linearly. At 128 KB per token, a 16k-token prompt holds 2 GB before generating anything. Maximum concurrency falls in proportion.
- **Decode step time.** Each step reads every active sequence's cache. Once total KV bytes approach weight bytes, decode slows noticeably.
- **Interference.** Long prompts mean long prefills, which stall other users' decodes unless the engine chunks them.

## What output length changes

- **E2E latency.** Grows linearly with output tokens: E2E ≈ TTFT + output tokens × TPOT.
- **KV hold time.** Each sequence keeps its cache for the whole generation. Long outputs mean each slot is occupied longer, so fewer new requests can be admitted per second.
- **Throughput.** Prefill is paid once per request, then spread over all output tokens. Long outputs amortize prefill, so output tokens per second often looks better with long outputs even though requests per second falls.

## Reading the grid

| Input \ Output | 64 | 256 | 1,024 |
| --- | --- | --- | --- |
| 256 | light prefill, fixed overhead dominates | balanced | decode heavy |
| 4k | prefill heavy | mixed | KV pressure starts |
| 16k | TTFT dominated | KV limited | concurrency very low |

Each cell needs a different fix. Overhead-heavy cells need CUDA graphs and a faster scheduler. Prefill-heavy cells need prefix caching and chunked prefill. KV-limited cells need FP8 KV, more memory or more GPUs. Decode-heavy cells benefit from speculative decoding at low load and bigger batches at high load.

## When KV reads take over

At short context, the weights dominate every decode step. At long context, the cache does.

~~~dg-raw
%%widget kv Set a long context and a modest batch. The KV cache bar soon dwarfs the weights.
model = Llama 3 8B
gpu = H100 SXM (80 GB)
t = 15
b = 8
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**when attention, not weights, sets decode speed**

Llama 3 8B, 8 sequences at 32,000 tokens each.

KV per sequence: 32,000 × 128 KB = 4 GB. Across 8 sequences: 32 GB.

Each decode step reads 16 GB of weights and 32 GB of cache, 48 GB in total. At 3 TB/s that is 16 ms per step, three times the weight-only 5.3 ms.

Two thirds of each step now goes to attention. Making the weights smaller (FP8) saves only 8 GB of the 48. Making the cache smaller (FP8 KV) saves 16 GB. Here, KV quantization matters more than weight quantization.

</div>

## Planning from the grid

Once you have the grid, you can estimate the cost of any workload mix. If production is 70% at 1k/256, 20% at 4k/256 and 10% at 16k/64, combine the throughput of each cell weighted by its share of GPU time, not its share of requests. The 16k requests may be only 10% of traffic but take 40% of the prefill compute.

## Deep dive: the input and output grid in one formula

For a request with ⟦m434⟧ prompt and ⟦m435⟧ output tokens, sharing a batch of ⟦m436⟧ whose sequences hold ⟦m437⟧ tokens on average:

⟦M432⟧

Input length appears in three places: prefill FLOPs (linear), attention FLOPs (quadratic) and, through ⟦m438⟧, every later decode step. Output length is linear, but each output token also grows ⟦m439⟧: a response's own TPOT gets worse as it goes.

~~~dg-raw
%%chart Predicted decode step time against average context, three batch sizes (Llama 3 8B, H100).
x = 1024, 2048, 4096, 8192, 16384
logx = 1
xlab = Average context per sequence (tokens)
ylab = Step time (ms)
Batch 1: 6.76, 6.81, 6.91, 7.10, 7.48
Batch 8: 7.10, 7.48, 8.25, 9.78, 12.85
Batch 16: 7.48, 8.25, 9.78, 12.85, 18.99
%%
~~~

At batch 1 context hardly matters. At batch 16 and 16k context, step time is 2.5 times the short-context value. The share of each step spent reading KV is

⟦M433⟧

which is 68% at batch 16 and 16k. Above about 50%, KV-side optimizations (FP8 KV, smaller GQA groups, prefix sharing) pay more than weight-side ones.

<div class="take">
<div class="lab">Key takeaway</div>

prompt length sets TTFT and KV per user, and output length sets how long that KV is held. Map both, because the bottleneck shifts across the grid, and at long context the cache, not the weights, sets decode speed.

</div>
