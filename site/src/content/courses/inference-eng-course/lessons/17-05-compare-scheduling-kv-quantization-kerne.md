---
id: "17.05"
course: "inference-eng-course"
module: "17"
slug: "compare-scheduling-kv-quantization-kernels"
title: "Compare Scheduling, KV, Quantization & Kernels"
keyIdeas: ["Compare engines feature by feature: scheduler, KV management, prefix cache, quantization, kernels, parallelism.", "For each dimension, design a workload that isolates it.", "Control everything else so the difference is attributable.", "Features that do not touch your workload do not matter.", "Record the outcome in a feature matrix with your measurements."]
question: "Pick one dimension where engines differ and describe how you would test whether that difference matters for your workload."
answer: "Prefix caching: vLLM uses block-hash caching, SGLang a radix tree with cache-aware scheduling. Build a workload with realistic shared prefixes (multi-turn chat with system prompts), run both engines with identical settings under load, and compare hit rate, TTFT p95 and goodput. Then repeat with no shared prefixes to confirm the difference comes from caching."
sources: [{"title": "vLLM documentation", "url": "https://docs.vllm.ai/en/latest/", "note": "Feature references."}, {"title": "SGLang documentation", "url": "https://docs.sglang.ai/", "note": "Feature references."}]
tags: ["engines"]
---

## Comparing engines one dimension at a time

An end-to-end benchmark tells you which engine is faster on one workload, but not why, and not whether the result will hold for your next workload. To understand the difference, compare engines one dimension at a time, with workloads designed to isolate each one.

## The dimensions

| Dimension | Questions to answer |
| --- | --- |
| Scheduler | How is the token budget filled? Is chunked prefill on? Are there priorities? Is CPU work overlapped with GPU work? |
| KV management | What is the block size? Recompute or swap on preemption? Is there CPU or SSD offload? |
| Prefix cache | Hash-chained blocks or a radix tree? Does scheduling favour cache hits? |
| Quantization | Which formats are supported, and how good are the kernels at your batch sizes? |
| Attention kernels | Which backends? How fast is long-context decode at small batch? |
| Speculation | Which methods? Is draft length adaptive? Does it back off under load? |
| Parallelism | Which of tensor, pipeline, expert and data parallelism? Is prefill-decode disaggregation supported? |
| Structured output | Which grammar backends? How much overhead per token? |

## Isolation workloads

Design small workloads that stress one dimension and leave the others quiet:

- **Prefix cache**: identical runs with fully shared prompts and with fully unique prompts. The difference between the two runs, for each engine, measures its prefix cache.
- **Scheduler overhead**: many tiny requests (20 tokens in, 5 out) at a high rate. GPU work is small, so CPU overhead dominates.
- **Long-context decode**: 32,000-token prompts at concurrency 1 to 4. Attention kernels dominate.
- **Large-batch decode**: short prompts, long outputs at concurrency 256. Matmul efficiency and batching dominate.
- **Quantization**: the same run in BF16 and FP8 on each engine. The relative gain per engine measures its quantized kernels.
- **Structured output**: the same prompts with and without a JSON schema constraint.

<div class="ex">
<div class="lab">Worked example</div>

**what isolation revealed**

In an end-to-end chat benchmark, two engines were within 2% of each other.

Isolation workloads showed engine A was 25% faster on the tiny-request test (lower scheduler overhead) and engine B was 30% faster on long-context decode (a split-KV attention kernel).

The chat workload exercised neither strongly, so the totals matched. The team's upcoming document-analysis product would be dominated by long-context decode, so engine B was the better choice for it.

</div>

## Interpreting the results

A feature difference matters only to the extent that your production workload exercises it. Weight each dimension by its share of your traffic:

- If 90% of traffic is short chat with no shared prefixes, a prefix-cache advantage is nearly irrelevant.
- If most traffic is agent calls with long shared prefixes, it may decide everything.
- If you serve at low concurrency, large-batch efficiency matters little.

Profile your production traffic first (prompt and output lengths, prefix sharing, concurrency per GPU) and use it to weight the isolation results.

## Keep the tests

Turn the isolation workloads into a standard suite. When a new engine version comes out, rerun the suite and compare each dimension with the last run. This catches regressions in one area that an end-to-end benchmark would hide behind gains in another.

## Deep dive: splitting a difference into causes

If isolation workloads each change one factor, the total ratio between engines is approximately the product of the factor ratios, so their logarithms add:

⟦M889⟧

~~~dg-raw
%%bars Engine A over engine B, split into causes by isolation tests (illustrative, % of the total log gain).
unit = %
Scheduler and batching (1.12×) | 43 | c1
Attention kernel at long context (1.08×) | 29 | c2
Prefix cache hit rate (1.075×) | 28 | c3
%%
~~~

Product: ⟦m890⟧, matching a measured 1.30 on the mixed workload. If the product and the mixed result disagree by more than the noise, the factors interact, and the isolation tests are missing something.

<div class="take">
<div class="lab">Key takeaway</div>

compare engines one dimension at a time with workloads that isolate each, such as prefix sharing, scheduler overhead and long-context decode. Weight each result by how much your traffic depends on it, and keep the suite for future upgrades.

</div>
