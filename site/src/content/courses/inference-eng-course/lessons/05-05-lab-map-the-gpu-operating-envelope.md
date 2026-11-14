---
id: "05.05"
course: "inference-eng-course"
module: "05"
slug: "lab-map-the-gpu-operating-envelope"
title: "Lab: Map the GPU Operating Envelope"
keyIdeas: ["Combine the baseline, load sweep and length sweep into one map.", "For each workload shape, find the highest SLO-safe load.", "Label the binding constraint for each cell.", "Include headroom: run at 70 to 80% of the boundary in production.", "This map is the input for capacity planning in Module 25."]
question: "Summarize what an operating envelope chart shows and how a product team would use it."
answer: "For each workload shape (prompt and output length), it shows the maximum request rate or concurrency at which the SLO still holds, plus the throughput there and the binding constraint (SLO, compute or KV memory). A product team uses it to know how much traffic one GPU can take for their prompt sizes, and when they need more replicas or a different model."
sources: [{"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Tool for rate sweeps."}, {"title": "vLLM metrics documentation", "url": "https://docs.vllm.ai/en/latest/usage/metrics.html", "note": "Engine metrics to collect during sweeps."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Turn many benchmarks into one decision tool: the operating envelope of one GPU. For each workload shape, it says how much traffic one GPU can carry within the SLO, and what limits it. Capacity plans, cost estimates and optimization priorities all come from this table.

## Deliverable

A table and a chart. For each pair of input and output length:

- the maximum SLO-safe request rate,
- the output throughput at that rate,
- the limiting factor,
- engine metrics at that point: KV usage, waiting queue, preemptions.

## Steps

1. **Fix the SLO.** For example p95 TTFT under 1 s and p95 TPOT under 50 ms, with 99% attainment.
2. **Choose the shapes.** Use at least four: short and short, medium prompt, long prompt, long output. Add your real production mix if you know it.
3. **Sweep rate for each shape.** Use open-loop Poisson arrivals. Start at a low rate and increase in steps of about 20%, then refine with smaller steps near the boundary.
4. **Find the highest rate with 99% attainment.** Interpolate between the last passing and first failing rate, then confirm with one more run.
5. **Record engine metrics at that rate.** KV usage, average waiting queue, preemption rate.
6. **Label the limit.** TTFT SLO, TPOT SLO, KV memory or saturation.

## Example output

| Input / output | Max req/s | Output tok/s | Limited by |
| --- | --- | --- | --- |
| 512 / 256 | 18 | 4,600 | TPOT SLO |
| 2k / 256 | 11 | 2,800 | TTFT SLO |
| 8k / 512 | 3 | 1,500 | KV memory |
| 32k / 256 | 0.6 | 150 | KV memory |

## Interpreting the envelope

Each limiting factor points to a different family of fixes:

- **TTFT-limited rows** benefit from prefill optimizations: chunked prefill, prefix caching, FP8 compute, or separating prefill from decode.
- **TPOT-limited rows** benefit from cheaper decode: a smaller or quantized model, a better attention kernel, speculative decoding at low load, or a smaller maximum batch.
- **KV-limited rows** benefit from more cache capacity: FP8 KV, a GPU with more memory, tensor parallelism, or offloading idle conversations.
- **Saturation-limited rows** mean the GPU is simply full. More GPUs is the honest answer.

<div class="ex">
<div class="lab">Worked example</div>

**using the envelope for a capacity plan**

Production traffic is 60% at 512/256, 30% at 2k/256 and 10% at 8k/512, totalling 40 requests per second at peak.

Each shape's share of one GPU per request is 1 ÷ its max rate: 1/18, 1/11 and 1/3.

GPUs needed: 40 × (0.6 ÷ 18 + 0.3 ÷ 11 + 0.1 ÷ 3) = 40 × (0.033 + 0.027 + 0.033) ≈ 3.8. With 25% headroom for bursts, plan for 5 GPUs.

Note that the 10% of long requests consume about a third of the capacity. That is where an optimization would pay off most.

</div>

The mixing formula assumes the shapes do not interfere with each other. In reality, long prefills slow short requests sharing the same GPU, so verify the plan with one run using the real mix.

## Keep it current

Rerun the envelope when the model, the engine version, the GPU type or the SLO changes. Store each version, so you can see how an engine upgrade or a quantization change moved the limits. Over time, this table becomes the most useful document a serving team owns.

## Deep dive: why capacity mixes harmonically

If one GPU can sustain ⟦m449⟧ requests per second of shape ⟦m450⟧ alone, each such request uses ⟦m451⟧ GPU-seconds. With total rate ⟦m452⟧ and shape shares ⟦m453⟧:

⟦M448⟧

⟦m454⟧ is a weighted harmonic mean. It is always at or below the weighted arithmetic mean, and is dominated by the slowest shapes.

<div class="ex">
<div class="lab">Worked example</div>

**the mistake of averaging rates**

Shapes from 05.05: 60% at 18 req/s, 30% at 11 req/s, 10% at 3 req/s.

Arithmetic mean: ⟦m455⟧ req/s per GPU, so 40 req/s needs 2.8 GPUs.

Harmonic mean: ⟦m456⟧ req/s per GPU, so 3.8 GPUs. Averaging rates under-provisions by 26%.

</div>

~~~dg-raw
%%bars GPUs consumed by each shape at 40 req/s.
unit = GPUs
512 in / 256 out (60% of requests) | 1.33 | c3
2k in / 256 out (30%) | 1.09 | c2
8k in / 512 out (10%) | 1.33 | c1
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

the operating envelope turns many benchmarks into one decision tool. It tells you how much traffic of each shape one GPU can carry within the SLO, what limits it, and therefore which optimization to try next.

</div>
