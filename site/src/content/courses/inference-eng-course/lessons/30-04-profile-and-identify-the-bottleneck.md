---
id: "30.04"
course: "inference-eng-course"
module: "30"
slug: "profile-and-identify-the-bottleneck"
title: "Profile and Identify the Bottleneck"
keyIdeas: ["Use the workflow from Module 12 on the capstone baseline.", "Identify the binding constraint and its root cause.", "Estimate gains for candidate fixes.", "Rank by gain, effort and risk.", "Write a bottleneck report."]
question: "How do you decide which optimization phase to do first after profiling?"
answer: "Identify the binding constraint at the expected load (SLO metric that fails first, KV pressure, or throughput ceiling), break down step time and engine state to find its cause, and estimate each candidate fix's gain on SLO-safe capacity. Start with the fix with the largest expected gain per unit of effort and risk."
sources: [{"title": "NVIDIA Nsight Systems User Guide", "url": "https://docs.nvidia.com/nsight-systems/UserGuide/index.html", "note": "Profiling."}, {"title": "PyTorch Profiler recipe", "url": "https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html", "note": "Profiling."}]
tags: ["capstone"]
tools: ["Capstone"]
---

## Steps

1. **At the expected load,** find which SLO fails first or which resource saturates.
2. **Engine metrics:** queueing, KV usage, preemptions.
3. **Timeline:** the step-time breakdown and GPU idle time.
4. **Kernel analysis** of the top kernels.
5. **Candidate fixes** with Amdahl estimates.

## Amdahl estimates

If a fix speeds up a part that takes fraction p of the time by a factor s, the overall speedup is:

```
speedup = 1 / ((1 − p) + p / s)
```

<div class="ex">
<div class="lab">Worked example</div>

**ranking two fixes for the support assistant**

Prefill takes 70% of GPU time. Prefix caching would skip the 3,000 shared tokens of a 4,500-token prompt, so prefill work falls to a third: s = 3 on p = 0.7. Speedup = 1 ÷ (0.3 + 0.7 ÷ 3) ≈ 1.9×.

FP8 weights would speed up decode, which is 30% of time, by about 1.6×: speedup = 1 ÷ (0.7 + 0.3 ÷ 1.6) ≈ 1.13×.

Prefix caching comes first, by a wide margin, and has no quality risk.

</div>

## Ranking table

| Fix | Targets | Estimated capacity gain | Effort | Risk |
| --- | --- | --- | --- | --- |
| Prefix caching | prefill work | | low | low |
| FP8 KV | KV pressure | | low | low |
| Chunk budget | TTFT tail | | low | low |
| FP8 weights | decode speed | | low | quality |
| Speculation | TPOT | | medium | depends on load |
| TP 2 | TPOT, KV | | medium | cost |

## Deep dive: ranking fixes by expected value per day of work

Each candidate fix has an expected gain from Amdahl's law, a probability that it works as hoped, and a cost in engineering days:

⟦M1160⟧

| Fix | ⟦m1161⟧ | ⟦m1162⟧ | Pr[works] | Days | Expected gain | Gain per day |
| --- | --- | --- | --- | --- | --- | --- |
| FP8 KV cache | 0.30 | 1.8 | 0.9 | 2 | 12% | 6.0% |
| Prefix-aware routing | 0.25 | 2.5 | 0.7 | 5 | 10.5% | 2.1% |
| Custom attention kernel | 0.30 | 1.3 | 0.5 | 15 | 3.5% | 0.2% |
| Speculative decoding | 0.60 | 1.6 | 0.6 | 8 | 13.5% | 1.7% |

The custom kernel is the most interesting work and the worst investment here. Doing the first two items first also changes ⟦m1163⟧ for the others, so re-profile after each one.

<div class="take">
<div class="lab">Key takeaway</div>

profile, name the binding constraint and rank fixes by estimated gain. That order is your optimization plan.

</div>
