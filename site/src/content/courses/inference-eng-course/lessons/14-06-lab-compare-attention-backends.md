---
id: "14.06"
course: "inference-eng-course"
module: "14"
slug: "lab-compare-attention-backends"
title: "Lab: Compare Attention Backends"
keyIdeas: ["Compare at least two attention backends in one engine.", "Cover a grid: context length, batch size, prefill vs decode, KV dtype.", "Measure kernel time and end-to-end metrics.", "Check outputs match the reference within tolerance.", "Recommend a backend per workload shape."]
question: "What grid of conditions should an attention backend comparison cover?"
answer: "Prefill-heavy and decode-heavy workloads; short (1k) and long (32k) contexts; small and large batch sizes; BF16 and FP8 KV; and, if relevant, shared-prefix traffic. Measure both kernel time from a profile and end-to-end TTFT, TPOT and throughput, and verify outputs match."
sources: [{"title": "vLLM: Attention backends", "url": "https://docs.vllm.ai/en/latest/design/attention_backends.html", "note": "How to select backends."}, {"title": "FlashInfer", "url": "https://github.com/flashinfer-ai/flashinfer", "note": "One of the backends."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Compare the attention backends available in your engine across a grid of workloads, and recommend which to use for which workload, with profile data to explain why. Backends often look similar at short context and differ a lot at long context or small batch, so a single benchmark point is not enough.

## The grid

| Axis | Values |
| --- | --- |
| Context length | 1k, 8k, 32k tokens |
| Concurrency | 1, 16, 128 |
| Phase focus | prefill-heavy (long prompts, short outputs), decode-heavy (short prompts, long outputs) |
| KV precision | BF16, FP8 |

That is 36 combinations per backend. If time is short, start with the corners: short context at high concurrency, and long context at low concurrency, for both phases.

## Measurements

For each backend and grid point:

- **End to end**: p50 and p95 TTFT, p50 and p95 TPOT, output throughput.
- **Profile**: attention kernel time per step and its share of the step, from one Nsight Systems capture per point.
- **Correctness**: greedy outputs on 50 fixed prompts compared with a reference backend. Report how many match exactly and, for the others, how many tokens in they diverge. Alternatively compare logits within a tolerance.

Keep everything else fixed: engine version, model, scheduler settings and harness.

## Expected findings

- **Short context**: differences are small, because attention is a small share of the step. Choose by features and stability.
- **Long context, small batch, decode-heavy**: differences are large. Backends with split-KV and GQA grouping can be several times faster at the attention kernel.
- **Long context, prefill-heavy**: FlashAttention-3 on Hopper often leads.
- **FP8 KV**: support varies. Some backends support it fully, some fall back to slower paths, and some do not support it at all.
- **Shared prefixes at high concurrency**: backends with cascade attention pull ahead.

<div class="ex">
<div class="lab">Worked example</div>

**part of a results table**

Llama 3 8B on an H100, decode-heavy, 32k context, concurrency 1.

Backend A: attention 1.9 ms of a 7.4 ms step, TPOT 7.6 ms. Profile: 132 programs, 84% of peak bandwidth (split-KV active).

Backend B: attention 5.1 ms of a 10.6 ms step, TPOT 10.9 ms. Profile: 8 programs, 22% of peak bandwidth (no split at batch 1).

At concurrency 128 and 1k context, the two backends differ by less than 2% in TPOT.

</div>

## The recommendation

Write the result as a short table that maps workloads to backends, each with a one-line reason from the profile:

| Workload | Recommended backend | Reason |
| --- | --- | --- |
| Chat, short context, high concurrency | either; default | attention under 10% of step |
| Long-document Q&A, low concurrency | backend A | split-KV fills the GPU at batch 1 |
| Shared 4k system prompt, high concurrency | backend with cascade attention | reads the prefix once per step |

Add any correctness issues you found and the exact engine and backend versions tested.

## Deep dive: predicting the grid before running it

For each cell of the benchmark grid, predict attention's share of the decode step from bytes:

⟦M800⟧

| Batch | Context | Predicted attention share (Llama 3 8B) |
| --- | --- | --- |
| 1 | 1k | 1% |
| 1 | 32k | 21% |
| 32 | 4k | 52% |
| 128 | 1k | 52% |
| 16 | 32k | 81% |

Backends differ most where ⟦m801⟧ is large and parallelism is poor: long context at small batch. Where ⟦m802⟧ is small, any backend will look the same, as in the results table above.

<div class="take">
<div class="lab">Key takeaway</div>

attention backends differ most at long context and small batch. Measure across a grid, explain differences with profiles, and choose per workload rather than by reputation.

</div>
