---
id: "28.08"
course: "inference-eng-course"
module: "28"
slug: "project-benchmark-your-engine-against-vllm"
title: "Project: Benchmark Your Engine Against vLLM"
keyIdeas: ["Benchmark mini engine and vLLM with the same harness, model and workload.", "Compare throughput, TTFT, TPOT and max concurrency.", "Profile both and break down step time.", "Attribute the gap to specific missing features or kernels.", "Close one gap and measure the improvement."]
question: "Your mini engine reaches 40% of vLLM's throughput. How do you find out where the gap comes from?"
answer: "Profile both on the same workload and compare step time breakdowns: kernel time for attention and GEMMs, small-op count (fusion), GPU idle gaps (CPU overhead, no CUDA graphs), scheduling overhead per step, and batch sizes achieved (KV capacity). Then close the biggest gap first, for example by adding CUDA graphs or a better attention kernel, and re-measure."
sources: [{"title": "vLLM documentation", "url": "https://docs.vllm.ai/en/latest/", "note": "The comparison engine."}, {"title": "NVIDIA Nsight Systems User Guide", "url": "https://docs.nvidia.com/nsight-systems/UserGuide/index.html", "note": "Profiling both engines."}]
tags: ["project"]
tools: ["Project"]
---

## Comparison table

| Metric | Mini engine | vLLM | Ratio |
| --- | --- | --- | --- |
| Throughput at concurrency 32 | | | |
| TTFT p95 | | | |
| TPOT p50 | | | |
| Maximum concurrency at 2k context | | | |

Use the same model, GPU, workload and harness for both.

## Typical gaps

| Gap | Cause | Fix |
| --- | --- | --- |
| GPU idle between kernels | Python loop, no CUDA graphs | CUDA graphs, asynchronous scheduling |
| Slow attention | gather plus standard attention | a paged attention kernel |
| Many small ops | unfused norms and RoPE | fused kernels |
| Smaller batches | memory overhead | a tighter KV pool |
| Slow sampling | per-sequence top-p | a batched sampler |

<div class="ex">
<div class="lab">Worked example</div>

**attributing the gap**

A mini engine reaches 1,100 tokens/s at concurrency 32; vLLM reaches 3,300.

A profile of one decode step shows 30 ms in total: 9 ms of matmuls, 6 ms of attention with gather, 4 ms of small kernels, and 11 ms where the GPU is idle while Python prepares the next step.

The idle time is the largest gap. Capturing the decode step as a CUDA graph and preparing inputs for the next step while the current one runs could cut most of it.

</div>

## Deliverable

Results, profiles, a breakdown of the gap by cause, and one improvement implemented and measured.

## Deep dive: attributing the gap to a production engine

Write the gap as a product of factors and measure each one by switching a single feature:

⟦M1145⟧

~~~dg-raw
%%bars A 6× throughput gap, split by factor (illustrative, share of the log gap, %).
unit = %
Host overhead (graphs, scheduler in C++) | 45 | c1
Attention and fused kernels | 25 | c2
More sequences per GPU (KV layout, FP8) | 20 | c3
Everything else | 10 | c4
%%
~~~

The shares add because they are logarithms: ⟦m1146⟧, and 45% of it is a factor of about 2.2 from host overhead alone.

<div class="take">
<div class="lab">Key takeaway</div>

benchmarking your own engine against a production one, and explaining the gap with profiles, is the best test of everything in this course.

</div>
