---
id: "15.06"
course: "inference-eng-course"
module: "15"
slug: "lab-compile-and-capture-an-inference-workload"
title: "Lab: Compile and Capture an Inference Workload"
keyIdeas: ["Use your own generation loop with a KV cache from Module 03.", "Compare eager, compiled, and compiled with CUDA graphs.", "Measure TPOT at batch 1, 8, 64 and 128, plus startup time.", "Profile each to show the change in GPU idle time.", "Explain where each gain came from."]
question: "Which three configurations do you compare and what do you expect at batch 1 vs batch 128?"
answer: "Eager, torch.compile without graphs, and torch.compile with CUDA graphs. At batch 1, eager is slowest; compile helps through fusion; graphs give the biggest gain by removing launch overhead. At batch 128, all three are closer because kernels are long and overhead is a small share; compile still helps somewhat through fusion."
sources: [{"title": "PyTorch: Introduction to torch.compile", "url": "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html", "note": "Compile modes."}, {"title": "PyTorch blog: Accelerating PyTorch with CUDA Graphs", "url": "https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/", "note": "Graphs."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Take a decode step from your own generation loop through eager execution, torch.compile and CUDA graphs, and measure what each stage buys and costs. You will see directly where overhead goes and how much of it each technique removes.

## Step 1: make the step graph-friendly

Rewrite your decode step from Module 03 or 08 as a pure function of preallocated GPU tensors:

- input token ids, shape [B],
- positions, shape [B],
- the KV cache buffers, preallocated,
- context lengths or block tables, preallocated at their maximum size.

Move everything that needs the CPU, such as sampling decisions, stop checks and Python bookkeeping, outside the step. The step should only run tensor operations.

## Step 2: eager baseline

Measure TPOT at batch 1, 8, 64 and 128, with at least 100 steps after 20 warm-up steps. Use CUDA events or `torch.cuda.synchronize()` around the timed region.

## Step 3: compile

Wrap the step with `torch.compile(step, dynamic=False)`, or with a dynamic batch dimension. Measure compile time separately, then rerun the same timings. Check the logs for graph breaks and recompiles: there should be none after warm-up.

## Step 4: capture CUDA graphs

For each batch size, capture a CUDA graph of the compiled step, using static input buffers. Replay it and measure again. Verify that the outputs match the eager step on the same inputs.

## Step 5: profile

Capture one step of each configuration at batch 1 with Nsight Systems or the PyTorch profiler. Record the number of kernels and the GPU idle share.

## Expected results

| Batch | Eager | Compiled | Compiled + graphs |
| --- | --- | --- | --- |
| 1 | 1.0× | about 1.3× | about 2× |
| 8 | 1.0× | about 1.3× | about 1.7× |
| 128 | 1.0× | about 1.15× | about 1.2× |

These are illustrative. The pattern to look for: graphs give the largest gain at small batch, and fusion gives a modest gain everywhere.

<div class="ex">
<div class="lab">Worked example</div>

**one set of results**

An 8B model on an H100, own decode loop with paged KV.

Batch 1: eager 11.2 ms (710 kernels, 52% idle), compiled 8.3 ms (290 kernels, 33% idle), compiled + graphs 5.7 ms (idle 4%).

Batch 128: eager 16.8 ms, compiled 14.6 ms, compiled + graphs 14.0 ms.

Compile took 48 s; capturing graphs for 4 sizes took 6 s and used 0.9 GB.

</div>

## Also report

- Compile time and capture time.
- Extra memory used by graphs, from `torch.cuda.memory_reserved()` before and after capture.
- GPU idle share per configuration.
- Any graph breaks or recompiles you had to fix, and how.

## Write-up

One table of results, one profile screenshot per configuration at batch 1, and a paragraph that explains each speedup in terms of kernels and idle time. End with a recommendation for a real deployment, taking startup time into account.

## Deep dive: the result you should expect

Predicting each step's effect before the lab:

| Stage | What changes | Predicted TPOT at batch 1 |
| --- | --- | --- |
| Eager | Host-bound at about 1,000 ops | 18 to 30 ms |
| torch.compile | Fewer, fused kernels, host time falls | 10 to 14 ms |
| + CUDA graphs | Host time almost gone | 6 to 7 ms |
| Bandwidth floor | 16 GB / 3.35 TB/s | 4.8 ms |

If the graph version is above about 8 ms, look for a graph break that splits the step into several graphs, or a host synchronization inside the step (such as calling `.item()` on a tensor).

<div class="take">
<div class="lab">Key takeaway</div>

CUDA graphs win most at small batch, and fusion helps a little everywhere. Measure startup time and memory as well as speed, since that is what you pay for the gain.

</div>
