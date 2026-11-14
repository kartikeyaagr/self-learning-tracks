---
id: "12.01"
course: "inference-eng-course"
module: "12"
slug: "stop-guessing-build-a-profiling-workflow"
title: "Stop Guessing: Build a Profiling Workflow"
keyIdeas: ["Profile with a purpose: start from a measured symptom.", "Go top down: metrics, then timeline, then individual kernels.", "Separate CPU time, GPU idle time and GPU kernel time.", "Compare against theoretical limits to know how much is left.", "Change one thing at a time and re-measure end to end."]
question: "Outline a profiling workflow that goes from \"the server is slow\" to a specific fix."
answer: "Reproduce with a fixed workload and measure the symptom. Check engine metrics to rule out queueing or memory pressure. Capture a system-level timeline (Nsight Systems or the PyTorch profiler) of steady-state steps to find GPU idle gaps and which kernels dominate. Drill into the top kernels with Nsight Compute to see whether they are memory or compute bound and how close to peak. Form a hypothesis, change one thing, and re-measure."
sources: [{"title": "NVIDIA Nsight Systems User Guide", "url": "https://docs.nvidia.com/nsight-systems/UserGuide/index.html", "note": "System-wide timelines."}, {"title": "PyTorch Profiler recipe", "url": "https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html", "note": "Op-level profiling in PyTorch."}]
tags: ["profiling"]
---

## Stop guessing

Inference performance problems invite guesses. "It must be attention." "Probably the tokenizer." "Let's try a newer kernel." Some guesses are right, but each wrong one costs a day. Profiling replaces guessing with a short chain of measurements, each narrowing where the time goes.

## The top-down workflow

~~~dg-flow From symptom to fix, top down. Press Play to step through it.
Symptom -> Engine metrics -> *Timeline profile -> Kernel deep dive -> Hypothesis -> Change and remeasure
~~~

1. **State the symptom precisely.** A metric, a workload, a load level and a target. "p50 TPOT is 22 ms at concurrency 32 for 1k in and 256 out on one H100. The target is 15 ms." Vague symptoms produce vague investigations.
2. **Check engine metrics first.** Is the time spent queueing, preempted or waiting for memory? If p95 TTFT is high because 50 requests are waiting, no kernel change will help. Profile kernels only when the time is inside the forward pass.
3. **Capture a timeline.** Profile a few steady-state steps with Nsight Systems or the PyTorch profiler. Measure the step time, the GPU idle gaps and the kernels that take the most total time.
4. **Deep dive on the top kernels.** For the top three, use Nsight Compute to see how close each comes to the hardware's limit, and which limit.
5. **Form a hypothesis with a number.** "Attention is 40% of the step and achieves 30% of peak bandwidth. A better backend at 80% would save about 25% of step time."
6. **Change one thing and remeasure** with the same workload and load, at the system level.

## Classify the time

Every millisecond of a step falls into one of four categories. Each has a different family of fixes.

| Category | How you find it | Typical fixes |
| --- | --- | --- |
| CPU overhead and GPU idle | gaps between kernels in the timeline | CUDA graphs, async scheduling, less Python per step |
| Memory-bound kernels | high bandwidth use, low FLOP rate | fewer bytes: quantization, fusion, smaller KV |
| Compute-bound kernels | high tensor core use | lower-precision math, better kernels, less work |
| Communication | NCCL kernels in the timeline | better topology, a different parallelism layout, overlap |

<div class="ex">
<div class="lab">Worked example</div>

**a guess and a profile**

A team guessed their slow decode was caused by attention and spent a week testing three attention backends. TPOT moved by 4%.

A timeline profile then showed the GPU idle for 38% of each 22 ms step: Python was building input tensors and launching about 900 kernels per step, and CUDA graphs were disabled by a debugging flag left in the launch script.

Removing the flag cut TPOT to 13 ms. One profile would have found it on day one.

</div>

## Estimate before you change

Use Amdahl's law to decide what is worth doing. If a kernel group takes a fraction f of the step and you make it s times faster, the step becomes 1 ÷ ((1 − f) + f ÷ s) times faster.

- Attention at 15% of the step, made 2 times faster: 1.08 times overall.
- GPU idle at 35% of the step, removed entirely: 1.54 times overall.

Rank possible fixes by expected gain and effort, then start at the top.

## Keep a profiling log

Write down every profile you take: the symptom, the configuration, the breakdown and what you concluded. Over time this log becomes a map of where your system spends its time, and it makes regressions easy to spot when a new version changes the breakdown.

## Deep dive: the expected gain of a fix, before you make it

If a change speeds up something that takes a share ⟦m680⟧ of the time by a factor ⟦m681⟧, the time saved is

⟦M679⟧

The rule is simple: you cannot save more than ⟦m682⟧. A kernel that is 3% of the step is never worth a week, however slow it looks in isolation.

~~~dg-raw
%%bars Where the time goes in two profiles of the same model (illustrative shares of one step, %).
unit = %
stack = Matmuls, Attention, Norms and elementwise, GPU idle
Decode, batch 1 | 68+6+8+18
Decode, batch 128, 2k context | 58+30+7+5
Prefill, 4k tokens | 82+12+5+1
%%
~~~

Each bar has a different first target. At batch 1, the 18% idle time is the cheapest win (CUDA graphs). At batch 128, attention reads are 30% (FP8 KV, better kernel). In prefill, the matmuls are 82% and already near peak, so the real lever is doing fewer of them (prefix caching).

<div class="take">
<div class="lab">Key takeaway</div>

profile from a precise symptom, top down: engine metrics, then a timeline, then kernels. Classify time into overhead, memory, compute and communication, estimate each fix with Amdahl's law, and change one thing at a time.

</div>
