---
id: "11.08"
course: "inference-eng-course"
module: "11"
slug: "project-build-the-quality-speed-memory-pareto-frontier"
title: "Project: Build the Quality/Speed/Memory Pareto Frontier"
keyIdeas: ["Evaluate several configs: BF16, FP8 W8A8, INT8, INT4 AWQ/GPTQ, each with BF16 or FP8 KV.", "For each, measure quality, SLO-safe capacity and memory.", "Plot quality vs capacity; mark the frontier.", "Pick the cheapest frontier point that passes your quality threshold.", "Document the decision with the data."]
question: "What is a Pareto frontier in this context, and how do you choose a point on it?"
answer: "Each configuration (precision of weights and KV, method) gives a point with quality, capacity (or cost) and memory. A configuration is on the frontier if no other one is better on all axes. You choose the cheapest point on the frontier that meets your quality threshold and SLO."
sources: [{"title": "Give Me BF16 or Give Me Death?", "url": "https://arxiv.org/abs/2411.02355", "note": "Accuracy vs performance trade-offs."}, {"title": "lm-evaluation-harness", "url": "https://github.com/EleutherAI/lm-evaluation-harness", "note": "Quality measurements."}]
tags: ["project"]
tools: ["Project"]
---

## Goal

Map the trade-off between quality, capacity and memory for every realistic quantization option on your model and GPU, then pick the cheapest option that meets your quality bar. The result is a decision backed by a chart, not an opinion.

## Configurations to test

| Weights | KV cache |
| --- | --- |
| BF16 | BF16, FP8 |
| FP8 W8A8 | BF16, FP8 |
| INT4 AWQ | BF16, FP8 |
| INT4 GPTQ | FP8 |

Seven configurations is enough to see the shape of the frontier. Add FP4 if you have Blackwell GPUs.

## For each configuration, measure

1. **Quality**: your product eval plus two standard benchmarks, run with the production template and sampling, with confidence intervals.
2. **Capacity**: SLO-safe requests per second on your workload, from a fine rate sweep with your harness.
3. **Memory**: weight memory and KV capacity in tokens, from the startup log.
4. **Latency at batch 1**: TTFT and TPOT, as a sanity check against the bandwidth math.

Keep everything else fixed: engine version, scheduler settings, workload and SLO.

## Plot the frontier

Put SLO-safe capacity on the x-axis and the product eval score on the y-axis. Each configuration is one point. Draw the quality threshold as a horizontal line. Points further right are cheaper to serve, and points higher are better quality.

The **Pareto frontier** is the set of points that no other point beats on both axes. Connect them. Any configuration below the frontier is dominated: another option is both better and cheaper.

<div class="ex">
<div class="lab">Worked example</div>

**a frontier for an 8B chat model on an H100**

BF16/BF16: 9.0 req/s, score 82.4. BF16/FP8 KV: 10.1 req/s, 82.3.

FP8/BF16: 11.0 req/s, 82.1. FP8/FP8 KV: 12.4 req/s, 82.0.

INT4 AWQ/FP8 KV: 12.0 req/s, 79.8. INT4 GPTQ/FP8 KV: 11.8 req/s, 79.5.

The threshold is 81.4 (at most 1 point below BF16). FP8 weights with FP8 KV is the highest-capacity point above it. The INT4 options are dominated: lower quality and no faster at this batch size.

</div>

## Write the decision note

A short note, half a page at most:

- **The chosen configuration**, with its quality, capacity and memory numbers.
- **The next-best alternative** and why it lost.
- **The cost effect**: GPUs needed for the expected peak at the new capacity, compared with BF16.
- **Risks and checks**: tasks to watch in production, and the canary plan.

## Things that move the frontier

Rerun the project when any of these change:

- A new GPU generation with new native formats.
- New kernels in the engine, which can make a dominated option competitive.
- A different workload, especially a change in typical batch size or context length.
- A new model, since tolerance to quantization differs between models.

## Deep dive: placing the threshold with noise in mind

A quality threshold of "at most ⟦m675⟧ below baseline" only makes sense if the measurement can resolve ⟦m676⟧. Require

⟦M674⟧

where ⟦m677⟧ is the standard error of the paired difference. If it is larger than half of ⟦m678⟧, add items before trusting any point on the frontier near the threshold.

<div class="take">
<div class="lab">Key takeaway</div>

quantization is a choice on a frontier of quality, capacity and memory. Measure each option at the system level and with product evals, plot the frontier, and pick the cheapest option that meets the bar you set in advance.

</div>
