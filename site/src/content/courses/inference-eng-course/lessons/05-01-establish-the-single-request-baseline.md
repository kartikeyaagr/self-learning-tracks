---
id: "05.01"
course: "inference-eng-course"
module: "05"
slug: "establish-the-single-request-baseline"
title: "Establish the Single-Request Baseline"
keyIdeas: ["Start every capacity study with one request at a time on an idle server.", "Record TTFT for representative prompt lengths and TPOT for steady decode.", "Compare with the bandwidth ceiling: weight bytes / memory bandwidth.", "If the baseline misses the SLO, scaling and batching cannot save you.", "Repeat a few times; the baseline should be very stable."]
question: "Why measure a single request on an idle server before any load test, and what two numbers do you take from it?"
answer: "It gives the best case the hardware and engine can deliver with no queueing or interference, so every later result can be expressed as degradation from it. Take TTFT for your typical prompt length and TPOT for steady decode. If these alone miss the SLO, no amount of scheduling will fix it; you need a smaller model, faster hardware or a different engine configuration."
sources: [{"title": "vLLM benchmarks directory", "url": "https://github.com/vllm-project/vllm/tree/main/benchmarks", "note": "Includes a latency benchmark for single batches."}, {"title": "kipply: Transformer Inference Arithmetic", "url": "https://kipp.ly/transformer-inference-arithmetic/", "note": "Expected latency from first principles."}]
tags: ["baseline"]
---

## Why start with one request

Under load, the latency a user sees is made of three parts: the baseline cost of their own request, the time spent queueing, and the slowdown caused by sharing the GPU with others. If you only ever measure under load, you cannot tell which part a regression came from. The single-request baseline isolates the first part. It is the floor that no amount of scheduling can go below.

## What to measure

Use an idle, warmed-up server and send one request at a time:

- **TTFT** at your typical prompt length, and at your longest allowed prompt.
- **TPOT** for a long output (at least 256 tokens) at concurrency 1.
- **E2E** for a typical request, so you know the best case users can ever get.

Repeat each measurement 20 to 50 times and report the median. At concurrency 1 the variance should be small. If it is not, something on the host is interfering.

## Compare with the physics

A measured number means little until you compare it with what the hardware allows.

- **Decode ceiling at batch 1**: TPOT ≥ weight bytes ÷ memory bandwidth.
- **Prefill ceiling**: TTFT ≥ 2 × parameters × prompt tokens ÷ peak FLOP/s, plus the attention term at long prompts.

~~~dg-raw
%%widget roof Decode at batch 1 sits far down the memory slope. The distance to the roof is the room that batching can use.
mode = Decode
n = 0
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**grading a baseline**

Llama 3 8B in BF16 on an H100: 16 GB of weights, 3.35 TB/s of bandwidth.

Decode ceiling: 16 GB ÷ 3.35 TB/s ≈ 4.8 ms per token.

Measured TPOT of 7 ms means about 69% efficiency. That is good for a real engine.

Measured TPOT of 25 ms means 19% efficiency. Python overhead, missing CUDA graphs or a slow attention kernel is wasting most of each step.

For a 1,000-token prompt: 2 × 8 × 10⁹ × 1,000 = 16 TFLOP. At a realistic 700 TFLOP/s that is about 23 ms. A measured 45 ms is about 50% efficient, typical for short prompts where fixed overheads matter.

</div>

| Measurement | Measured | Theoretical floor | Efficiency |
| --- | --- | --- | --- |
| TPOT at batch 1 | 7 ms | 4.8 ms | 69% |
| TTFT, 1k-token prompt | 45 ms | about 23 ms | about 50% |
| TTFT, 16k-token prompt | 1.1 s | about 0.6 s | about 55% |

## What poor efficiency points to

- **TPOT far above the ceiling at batch 1**: launch overhead. Check whether CUDA graphs are enabled and whether the engine is running in eager mode.
- **TTFT far above the prefill estimate for short prompts**: fixed costs such as tokenization, scheduling, HTTP handling or a slow chat template.
- **TTFT growing faster than expected at long prompts**: an inefficient attention kernel, or prefill running without FlashAttention.
- **Large spread between repeated runs**: other processes on the GPU or CPU, thermal throttling, or a cold cache.

## Gate on the SLO before scaling

Batching raises throughput but never makes one user faster. Every user's TPOT under load is at least the baseline TPOT, and usually more. So check the baseline against the SLO first:

- If baseline TPOT is 40 ms and the SLO is p95 TPOT under 30 ms, stop. No scheduler setting can fix this. Use a smaller model, a lower precision, faster hardware, tensor parallelism, or speculative decoding.
- If baseline TTFT for your longest prompts already exceeds the TTFT SLO, the same applies to prefill: you need prefix caching, more compute or tensor parallelism.
- If the baseline uses only half the SLO budget, you have room to batch. The other half is what queueing and interference may consume.

## Record it properly

Store the baseline with the same provenance as any benchmark: model revision, engine version and flags, GPU and driver. Rerun it whenever any of those change. A baseline that silently drifts makes every later comparison wrong.

## Deep dive: MBU and MFU, the two efficiency numbers

Decode is graded by memory bandwidth utilization, prefill by model FLOPs utilization:

⟦M422⟧

"Had to be read" means weights plus live KV, not whatever the profiler counted. Extra traffic from a bad kernel lowers MBU, which is the point.

Measured step time splits into parts you can attack separately:

⟦M423⟧

~~~dg-raw
%%bars Where a 7.0 ms batch-1 decode step goes, Llama 3 8B on an H100 (illustrative split, ms).
unit = ms
stack = Floor at peak bandwidth, Kernel inefficiency, Gaps between kernels, Host work
Measured step | 4.8+0.55+1.3+0.35
After CUDA graphs | 4.8+0.55+0.25+0.35
%%
~~~

The floor is fixed by the hardware and the bytes. Only the other three are engineering. At 7.0 ms, 31% of the step is above the floor. CUDA graphs mostly remove gaps; they cannot touch the floor.

## Deep dive: why short prompts score low MFU

With a fixed overhead ⟦m425⟧, prefill efficiency is

⟦M424⟧

which goes to zero for short prompts and to one for long ones.

~~~dg-raw
%%chart Prefill efficiency against prompt length (t₀ = 15 ms, 700 TFLOP/s achieved, linear layers only).
x = 128, 512, 1024, 4096, 16384
logx = 1
xlab = Prompt tokens
ylab = Share of TTFT spent on useful FLOPs (%)
Efficiency: 16, 44, 61, 86, 96
%%
~~~

A 50% efficient TTFT on a 1k prompt is not a bad kernel. It is the fixed cost of tokenization, scheduling and launch, paid once per request. Grade TTFT at several lengths before deciding.

<div class="take">
<div class="lab">Key takeaway</div>

the single-request baseline is your floor. Compare it with the bandwidth and compute ceilings to find waste, and check it against the SLO before scaling anything, because batching never makes one user faster.

</div>
