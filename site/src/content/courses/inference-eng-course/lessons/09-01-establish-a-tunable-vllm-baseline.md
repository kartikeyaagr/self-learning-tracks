---
id: "09.01"
course: "inference-eng-course"
module: "09"
slug: "establish-a-tunable-vllm-baseline"
title: "Establish a Tunable vLLM Baseline"
keyIdeas: ["Start from defaults, but write every flag explicitly in a config file.", "Measure with the target workload at three loads: light, expected, peak.", "Record engine metrics alongside latency and throughput.", "Identify the binding constraint before changing anything.", "Change one variable at a time from here on."]
question: "What makes a baseline \"tunable\" rather than just a first measurement?"
answer: "It is fully pinned (model, engine version, every flag written down), measured on the target workload with the harness at several loads, and recorded with engine metrics. That way each later change touches one variable and can be compared directly, and you know which constraint (TTFT, TPOT, KV memory) binds first."
sources: [{"title": "vLLM: Engine arguments", "url": "https://docs.vllm.ai/en/latest/configuration/engine_args.html", "note": "Flags to pin."}, {"title": "vLLM: Metrics", "url": "https://docs.vllm.ai/en/latest/usage/metrics.html", "note": "Metrics to record."}]
tags: ["tuning", "vllm"]
---

## Make every setting explicit

Optimization starts with a baseline you can reproduce exactly. Write every engine flag into a launch script or config file, including the defaults you rely on. Defaults change between releases: a new version may change the token budget, turn prefix caching on, or pick a different attention backend. If your baseline depends on a default you never wrote down, an upgrade can move your numbers by 20% without anyone noticing why.

A baseline config for vLLM might look like this:

```
model: meta-llama/Llama-3.1-8B-Instruct
revision: <commit-sha>
dtype: bfloat16
max_model_len: 8192
gpu_memory_utilization: 0.90
max_num_seqs: 256
max_num_batched_tokens: 2048
enable_prefix_caching: true
enable_chunked_prefill: true
kv_cache_dtype: auto
```

Store it in version control next to the benchmark results it produced.

## Measure at three load points

One number cannot describe a server. Measure at three points with your real workload:

| Load | Purpose |
| --- | --- |
| Light, about 10% of capacity | the latency floor with the real engine and real workload |
| Expected peak | the production operating point |
| Stress, about 2× expected | headroom and failure behaviour |

At each point, record:

- p50, p95 and p99 of TTFT, TPOT and ITL.
- Output throughput and goodput against your SLO.
- Engine metrics: average and peak KV usage, waiting queue length, preemptions per minute, prefix cache hit rate.
- GPU metrics: power draw, and SM or DRAM activity if you have DCGM.

## Find the binding constraint

At the expected peak, one constraint fails first or comes closest to failing. That constraint decides what to work on.

- **TTFT fails first.** Prefill is too slow, or requests are queueing. Check the waiting queue: if it is long, admission is the problem; if it is short, prefill itself is slow.
- **TPOT fails first.** Decode is too slow for each user. The batch may be too large, the context too long, or the model too big for the GPU.
- **KV usage near 100%, with preemptions.** Memory is the constraint.
- **Nothing fails, but throughput is flat.** The GPU is saturated. Only more compute or a cheaper model helps.

<div class="ex">
<div class="lab">Worked example</div>

**reading a baseline**

An 8B model on one L40S, with a chat workload. At the expected peak of 6 requests per second:

p95 TTFT 1.4 s against an SLO of 1 s. p95 TPOT 34 ms against an SLO of 50 ms.

KV usage peaks at 99%, with 40 preemptions per minute. The waiting queue averages 12.

TTFT fails, but the cause is memory: requests wait because the pool is full. The next step is the memory lesson, not the scheduler.

</div>

## Plan the tuning

Map the constraint to the next step:

- **Memory**: lesson 09.02, tuning memory fraction and KV precision.
- **Scheduling**: revisit lesson 07.04, the token budget and max sequences.
- **Prefill**: Module 10, prefix caching and chunked prefill.
- **Precision**: Module 11, quantization of weights and KV.
- **Kernels and overhead**: Modules 12 to 15, profiling, attention kernels and CUDA graphs.

Write the plan down before you start, with the metric you expect each change to move. A change that moves a different metric than you expected deserves an explanation before you keep it.

## Guard quality

Some optimizations change outputs: quantization, FP8 KV, speculative decoding with sampling. Before you start, pick a small quality check you can run after every such change, such as an accuracy benchmark on a few hundred examples from your task, or a comparison of outputs against a reference. Record its baseline value with the performance baseline.

## Deep dive: naming the binding constraint from four numbers

At the target load, read four ratios. The one closest to 1 is the constraint, and it tells you which knob to turn:

⟦M579⟧

| Closest to 1 | What is binding | First knobs to try |
| --- | --- | --- |
| ⟦m580⟧ | KV capacity | FP8 KV, higher memory utilization, prefix caching |
| ⟦m581⟧ | The sequence cap | Raise max-num-seqs, if ⟦m582⟧ has room |
| ⟦m583⟧ | Per-user decode speed | Lower max-num-seqs, FP8 weights, speculative decoding |
| ⟦m584⟧ | Prefill or queueing | Chunk size, prefix caching, more replicas |

~~~dg-raw
%%bars Four ratios at 9 req/s for one baseline (illustrative). The SLO ratio binds first.
unit = × limit
KV used / pool | 0.71 | c4
Running / max-num-seqs | 0.66 | c4
p95 TPOT / SLO | 0.94 | c1
p95 TTFT / SLO | 0.48 | c4
%%
~~~

Here, raising max-num-seqs or the KV pool would buy nothing. The step is already too slow for the SLO, so the next gains must make each step cheaper.

<div class="take">
<div class="lab">Key takeaway</div>

a tunable baseline is pinned in a config file, measured at light, expected and stress loads, and annotated with its binding constraint. That constraint decides the next optimization, and every later change is judged against this baseline.

</div>
