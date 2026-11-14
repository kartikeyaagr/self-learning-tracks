---
id: "12.08"
course: "inference-eng-course"
module: "12"
slug: "lab-produce-a-bottleneck-report"
title: "Lab: Produce a Bottleneck Report"
keyIdeas: ["Pick one workload and load level where performance disappoints.", "Capture metrics, a timeline and kernel profiles.", "Break step time into categories and put top kernels on a roofline.", "Diagnose the bottleneck and rank fixes by estimated gain.", "Verify the top fix and report before and after."]
question: "What sections does a good bottleneck report have?"
answer: "Symptom and workload; environment; engine metrics summary; timeline breakdown of a steady-state step (kernel groups, idle time); top kernels with roofline position and efficiency; diagnosis naming the bottleneck class; ranked recommendations with estimated gain; and the verification plan."
sources: [{"title": "NVIDIA Nsight Systems User Guide", "url": "https://docs.nvidia.com/nsight-systems/UserGuide/index.html", "note": "Timeline capture."}, {"title": "NVIDIA Nsight Compute documentation", "url": "https://docs.nvidia.com/nsight-compute/NsightCompute/index.html", "note": "Kernel analysis."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Turn profiling data into a short, decision-ready document: a bottleneck report. It states the problem, shows where the time goes, ranks the fixes with estimated gains, and verifies the top one. Writing one makes your profiling disciplined, and it makes your conclusions checkable by others.

## The template

1. **Symptom.** The metric, the workload, the load level and the target. One or two sentences.
2. **Environment.** GPU, driver, engine version and flags, model and precision.
3. **Engine metrics.** Queueing, KV usage, preemptions and prefix hit rate at the time of the symptom. This rules out non-kernel causes.
4. **Step breakdown.** A table of kernel groups and GPU idle time for a representative steady-state step, from Nsight Systems.
5. **Top kernels.** For the three largest, the achieved bandwidth or FLOP/s as a percentage of peak, from Nsight Compute, and the regime they are in.
6. **Diagnosis.** Which category dominates: overhead, memory, compute or communication.
7. **Recommendations.** Ranked by expected gain, each with an Amdahl estimate and the effort it needs.
8. **Verification.** The measured result of applying the top recommendation, compared with its estimate.

## An example breakdown

| Category | ms per step | Share |
| --- | --- | --- |
| Matmuls | 6.1 | 55% |
| Attention | 2.4 | 22% |
| Small ops (norms, RoPE, adds, sampling) | 0.9 | 8% |
| GPU idle | 1.7 | 15% |
| Total | 11.1 | 100% |

## Estimating the recommendations

- **CUDA graphs** should remove most of the idle time. If 1.5 of the 1.7 ms goes: 11.1 → 9.6 ms, about 14% faster.
- **FP8 weights** could make matmuls about 1.6 times faster at this batch: 6.1 → 3.8 ms, saving 2.3 ms, about 21% faster on its own.
- **A better attention backend**, if attention is at 45% of peak bandwidth and could reach 80%: 2.4 → 1.35 ms, about 9% faster.
- **Fusing small ops** might save half of their 0.9 ms: about 4%.

Combined, the first two give about 11.1 → 7.3 ms if their savings add up, about 1.5 times faster. Rank: CUDA graphs first (cheap, large), then FP8 (needs a quality check), then attention, then fusion.

<div class="ex">
<div class="lab">Worked example</div>

**the verification section**

Recommendation 1 applied: CUDA graphs enabled for batch sizes up to 256.

Estimated step time: 9.6 ms. Measured: 9.9 ms. The difference is a small gap left for the sampler, which runs outside the graph.

System level: p50 TPOT at the target load fell from 13 ms to 11.4 ms, and SLO-safe capacity rose 9%. The estimate was close, which raises confidence in the other estimates.

</div>

## Calibrating your estimates

Verification is not a formality. Comparing estimated and measured gains tells you how good your mental model is. If estimates are consistently too optimistic, find out why: often a fix moves time into another category, such as faster matmuls exposing more idle time. Over several reports, your estimates get better, and so does your ranking of what to try.

## Deliverable

A report of one to two pages following the template, with the timeline screenshot, the kernel table, the ranked recommendations and the verification result. Keep the raw profiles alongside it so anyone can check the numbers.

## Deep dive: calibrating your estimates

For every recommendation, record the predicted and measured gain, and track the log ratio:

⟦M706⟧

The mean of ⟦m707⟧ shows bias (are you optimistic?); its spread shows how much to trust the next estimate. Most people are optimistic: predicted gains overshoot by 20 to 50% because they forget Amdahl's law and the parts of the step that do not change.

<div class="take">
<div class="lab">Key takeaway</div>

a bottleneck report turns a profile into a ranked, estimated plan. Verifying the top item closes the loop and calibrates your estimates for the next investigation.

</div>
