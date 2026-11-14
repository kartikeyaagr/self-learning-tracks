---
id: "06.06"
course: "inference-eng-course"
module: "06"
slug: "lab-pytorch-vs-vllm"
title: "Lab: PyTorch vs vLLM"
keyIdeas: ["Run your Module 01 server and vLLM with the same model, dtype and workload.", "Use your harness to sweep concurrency 1 to 128.", "Compare TTFT, TPOT, throughput and max concurrency before SLO failure.", "Explain each gap using engine features.", "Keep these results as the baseline for Module 09."]
question: "You see vLLM win by 15 times at concurrency 64 but only 1.5 times at concurrency 1. Explain both numbers."
answer: "At concurrency 1 the gain comes only from lower per-step overhead and better kernels (CUDA graphs, fused attention), since both run one sequence. At concurrency 64, vLLM batches all sequences into every decode step and uses paged KV memory to fit them, while the naive server processes them one at a time, so throughput scales with batch size for vLLM and stays flat for the naive server."
sources: [{"title": "Efficient Memory Management for LLM Serving with PagedAttention", "url": "https://arxiv.org/abs/2309.06180", "note": "The paper's comparison with HF baselines."}, {"title": "vLLM documentation", "url": "https://docs.vllm.ai/en/latest/", "note": "Engine used in the lab."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Measure how much a real engine improves on the plain PyTorch server from Module 01, and explain each part of the improvement with an engine feature. This is the lab where batching and memory management stop being abstract.

## Setup

Fairness matters more than speed here. Keep these identical in both servers:

- The same model, revision and dtype.
- The same GPU, driver and power settings.
- The same workload file, for example 1,000 tokens in and 256 out, with fixed output length.
- The same warm-up (at least 30 requests) and measurement duration.
- The same client and harness from Module 04.

## Runs

Sweep closed-loop concurrency 1, 4, 16, 64 and 128 against each server. At each level record output tokens per second, p50 and p95 TTFT, and p50 and p95 TPOT. For the engine, also scrape running, waiting and KV usage.

## What you should see

| Concurrency | Plain PyTorch tok/s | vLLM tok/s | Ratio |
| --- | --- | --- | --- |
| 1 | 40 | 120 to 160 | about 3 to 4× |
| 16 | 40 | about 1,700 | about 40× |
| 64 | 40 | about 5,500 | about 140× |
| 128 | 40 | about 8,000 | about 200× |

Your numbers will differ with the model and GPU. The pattern should not: a modest gain at concurrency 1 and a huge one at high concurrency.

## Explaining the gaps

Tie each part of the difference to a feature:

- **At concurrency 1**, there is no batching to exploit. The 3 to 4 times gain comes from CUDA graphs that remove launch gaps, fused kernels and a faster attention implementation.
- **From 4 to 64**, continuous batching shares each weight read across many sequences. The plain server's throughput stays flat because it never runs more than one sequence.
- **At high concurrency**, paged KV memory lets the engine fit many more sequences than per-request pre-allocation would, so the batch can keep growing.
- **TTFT**: on the plain server it grows linearly with concurrency because requests queue. On the engine it stays low until the KV pool or token budget fills.

## Optional ablations

To prove which feature gave which gain, turn features off one at a time in the engine and rerun:

- `--enforce-eager` disables CUDA graphs. Expect batch-1 TPOT to rise by 30 to 100%.
- `--max-num-seqs 1` disables batching in practice. Expect throughput to fall close to the plain server's.
- Turning off prefix caching changes nothing on random prompts, and a lot on shared-prompt workloads.

<div class="ex">
<div class="lab">Worked example</div>

**an ablation table**

Llama 3 8B, H100, concurrency 64, 1k in and 256 out.

Full vLLM: 5,600 tok/s. With enforce-eager: 4,300 tok/s (graphs worth about 30% at this batch).

With max-num-seqs 8: 1,150 tok/s. With max-num-seqs 1: 150 tok/s.

Batching explains most of the gap. Graphs and kernels explain the rest.

</div>

## Report

Two charts: throughput against concurrency and p95 TTFT against concurrency, with both servers on each. Below them, a short paragraph per concurrency range explaining the gap with the feature responsible, and the ablation table if you ran it.

## Deep dive: Amdahl's law for ablations

If an optimization removes a fraction ⟦m486⟧ of step time, the speed-up is

⟦M485⟧

Ablations do not multiply cleanly. Each one's ⟦m487⟧ is measured against a step time that the others have already shrunk.

<div class="ex">
<div class="lab">Worked example</div>

**two ablations measured separately**

Baseline step: 25 ms. CUDA graphs alone remove 12 ms (speed-up 1.92). Batched sampling alone removes 3 ms (speed-up 1.14).

Multiplied, that predicts ⟦m488⟧. Combined, the step is ⟦m489⟧ ms, a speed-up of 2.5, because the 3 ms saved is a larger share of the smaller step.

Report ablations as milliseconds removed from a named step, not as percentages, and they add up.

</div>

~~~dg-raw
%%bars Batch-1 step time as features are added (illustrative, ms).
unit = ms
Plain generate | 25 | c4
+ static KV cache | 19 | c3
+ CUDA graphs | 8.5 | c2
+ fused kernels | 7.2 | c1
+ batched sampling | 7.0 | go
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

most of an engine's advantage appears under concurrency. That is continuous batching and paged memory at work, which the next two modules explain in detail. At batch 1 the smaller gain comes from graphs and kernels.

</div>
