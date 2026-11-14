---
id: "29.07"
course: "inference-eng-course"
module: "29"
slug: "find-where-the-technique-stops-working"
title: "Find Where the Technique Stops Working"
keyIdeas: ["Every technique has a regime where it helps and one where it hurts.", "Sweep load, workload shape, model size and hardware.", "Find the crossover points and explain them with profiles.", "These boundaries are the most useful part of a reproduction.", "Report them as a map, not a single number."]
question: "Give three axes along which you would search for where an inference technique stops helping, with an example."
answer: "Load (concurrency or request rate): speculative decoding helps at low load and stops helping at high load. Workload shape (prompt and output lengths, prefix sharing): prefix caching does nothing without shared prefixes. Hardware and scale (GPU type, interconnect, model size): disaggregation stops paying on slow networks or at small scale."
sources: [{"title": "Optimizing Speculative Decoding for Serving LLMs Using Goodput", "url": "https://arxiv.org/abs/2406.14066", "note": "Speculation degrading under load."}, {"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "Where disaggregation helps."}]
tags: ["research"]
---

## Axes to sweep

| Axis | Example range |
| --- | --- |
| Load | concurrency 1 to 256 |
| Prompt length | 256 to 32k tokens |
| Output length | 32 to 4k tokens |
| Prefix sharing | 0 to 90% |
| Model size | 1B to 70B |
| Hardware | different GPU generations and interconnects |

## Map it

Show the gain as a heatmap over two axes, for example load against prompt length. Mark the cells where the gain is below 1, meaning the technique makes things worse.

~~~dg-raw
%%widget heat A gain map for speculative decoding over batch size and acceptance rate. Raise the draft length and watch the red region grow at high batch.
k = 4
c = 0.05
bc = 160
%%
~~~

## Explain each boundary

Use the profile from 29.05 to explain why each boundary sits where it does.

- **Speculative decoding** stops helping at high load. Compute is already saturated, so verifying several draft tokens costs real compute and rejected tokens are pure waste.
- **Chunked prefill** does nothing when all prompts are short. There is nothing to chunk.
- **Prefix caching** does nothing when no prompts share a prefix, and costs a little memory and bookkeeping.

<div class="ex">
<div class="lab">Worked example</div>

**reading the map**

On the widget's default settings, speculation gives about 2 to 3× at batch 1 to 16 with acceptance 0.8, about 1.5× at batch 64, and below 1 from batch 128 upward.

A paper that reports only batch 1 shows the best corner of this map. A production service running at batch 128 would be slowed down by it, so engines that speculate need to turn it off, or shorten the draft, under heavy load.

</div>

## Deep dive: a regime map you can compute

Before sweeping, compute which resource binds in each cell from the step model: weight reads, KV reads, compute, or memory capacity.

~~~dg-raw
%%grid What limits a decode step for Llama 3 8B on one H100, by batch size (rows) and context per sequence (columns).
cols = 256, 1k, 4k, 16k, 64k, 128k
key W = weight reads dominate | c3
key K = KV reads exceed weight reads | c1
key C = compute bound | c2
key X = does not fit in memory | line
B = 1 | W W W W W K
B = 4 | W W W W K X
B = 16 | W W W K X X
B = 64 | W W K X X X
B = 256 | W K X X X X
B = 512 | C X X X X X
%%
~~~

Compute-bound decode appears only at very large batches of very short sequences. Most of the map is memory: weight-bound in the upper left, KV-bound along the diagonal, out of memory below it. A technique's measured gain should follow its region. Weight quantization helps in the W region, KV quantization and better attention kernels along the K diagonal, and nothing but more memory or offloading below it.

<div class="take">
<div class="lab">Key takeaway</div>

the limits of a technique are as valuable as its gains. Sweep the axes, map where it crosses below 1, and explain why.

</div>
