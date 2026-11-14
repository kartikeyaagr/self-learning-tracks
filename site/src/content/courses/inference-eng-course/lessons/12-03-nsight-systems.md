---
id: "12.03"
course: "inference-eng-course"
module: "12"
slug: "nsight-systems"
title: "Nsight Systems"
keyIdeas: ["Nsight Systems (nsys) records a system-wide timeline: CPU threads, CUDA API calls, kernels, memory copies, NCCL.", "Add NVTX ranges to label steps and phases.", "Look for GPU idle gaps, serialization, and unexpected host-device copies.", "Use it to find where to look, then use Nsight Compute for kernel detail.", "Capture short windows of steady state to keep files manageable."]
question: "In an Nsight Systems timeline of a decode step you see many short kernels separated by gaps of 5 to 10 microseconds. What is the likely cause and fix?"
answer: "The GPU is waiting for the CPU to launch each kernel: launch overhead dominates because each kernel finishes faster than the next can be issued. Fixes are CUDA graphs (record the whole step once and replay it with one launch), kernel fusion to reduce the number of kernels, and moving scheduling work off the critical path."
sources: [{"title": "NVIDIA Nsight Systems User Guide", "url": "https://docs.nvidia.com/nsight-systems/UserGuide/index.html", "note": "Capture and analysis."}, {"title": "NVTX documentation", "url": "https://nvidia.github.io/NVTX/", "note": "Annotating code ranges."}]
tags: ["profiling", "nsight"]
---

## A whole-system timeline

Nsight Systems (`nsys`) records everything that happens on the machine during a capture: CPU threads, CUDA API calls, kernels on each GPU, memory copies, NCCL communication and operating system events. It has low overhead and sees code the PyTorch profiler cannot, such as C++ schedulers, other processes and communication libraries. It is the main tool for answering "where does a step's time go?".

## Capturing a serving process

Capture only a short steady-state window. A full minute of a busy server produces an enormous file.

```
nsys profile -t cuda,nvtx,osrt,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o decode_trace python serve.py
```

In the server, wrap the steps you want with:

```
torch.cuda.profiler.start()
# ... a few steady-state steps ...
torch.cuda.profiler.stop()
```

Alternatively, use `--delay` and `--duration` to capture a time window after warm-up.

## Annotate with NVTX

Raw timelines are hard to read. NVTX ranges label regions of code, and they appear as named bars above the GPU rows:

```
import torch.cuda.nvtx as nvtx
nvtx.range_push("schedule"); batch = scheduler.schedule(); nvtx.range_pop()
nvtx.range_push("forward");  out = model_runner.execute(batch); nvtx.range_pop()
nvtx.range_push("sample");   tokens = sampler(out); nvtx.range_pop()
```

Most engines already include NVTX ranges you can enable with a flag or environment variable.

## Patterns to look for

| Pattern | What it means |
| --- | --- |
| Gaps between kernels on the GPU row | the GPU waits for the CPU: launch-bound |
| One kernel taking a large share | the main optimization target |
| cudaMemcpy on the critical path | host-device copies to remove or overlap |
| Frequent cudaStreamSynchronize | unnecessary synchronization, often from .item() |
| Long NCCL kernels | communication-bound (multi-GPU) |
| CPU thread busy in Python between steps | scheduler, tokenizer or bookkeeping overhead |
| Kernels on one GPU finishing much later than others | load imbalance across tensor-parallel ranks |

## The classic small-batch decode picture

At small batch without CUDA graphs, a decode step looks like hundreds of short kernels, each a few to tens of microseconds, separated by small gaps. The CPU row shows Python walking through the model's layers, launching each one.

<div class="ex">
<div class="lab">Worked example</div>

**reading a decode step**

An 8B model at batch 8, CUDA graphs off. Wall-clock step: 10.2 ms.

The GPU row shows 620 kernels. Their total execution time is 6.1 ms. The remaining 4.1 ms is gaps.

The CPU row shows the launch loop taking 9.8 ms: the CPU is the bottleneck, and the GPU waits for it.

With CUDA graphs on, the same step takes 6.4 ms. The 620 launches become one graph launch of about 15 µs.

</div>

## Mixed steps

With continuous batching and chunked prefill, steps differ from each other. Capture enough steps to see the variety, then use the NVTX ranges and the tokens per step (log it as an NVTX range name) to group similar steps. Compare a decode-only step with a step containing a prefill chunk to see how much the chunk costs.

## From timeline to next step

The timeline tells you which category dominates:

- Mostly gaps: fix overhead first (Module 15).
- One or two kernel types dominate: profile them with Nsight Compute (next lesson).
- NCCL dominates: revisit parallelism and topology (Module 18).
- CPU work between steps dominates: profile the scheduler with a CPU profiler such as py-spy.

## Deep dive: when the CPU cannot keep up

A decode step launches about 12 kernels per layer, so about 400 for a 32-layer model. If each launch costs the CPU ⟦m690⟧ and each kernel runs for ⟦m691⟧ on the GPU, the GPU starves whenever

⟦M689⟧

because the next kernel has not been queued when the previous one ends. At batch 1, many kernels run for 3 to 10 µs, and Python-side launch costs are about 5 to 10 µs.

~~~dg-raw
%%timeline CPU launching faster than the GPU runs (top) versus slower (bottom) (µs).
unit = µs
tick = 10
CPU, fast launch | 0-3 L1 c4 ; 3-6 L2 c4 ; 6-9 L3 c4 ; 9-12 L4 c4
GPU | 3-15 k1 c3 ; 15-27 k2 c3 ; 27-39 k3 c3 ; 39-51 k4 c3
CPU, slow launch | 0-14 L1 c4 ; 14-28 L2 c4 ; 28-42 L3 c4 ; 42-56 L4 c4
GPU | 14-20 k1 c3 ; 20-28 idle line ; 28-34 k2 c3 ; 34-42 idle line ; 42-48 k3 c3 ; 48-56 idle line ; 56-62 k4 c3
%%
~~~

In the bottom case the GPU is idle half the time, and no kernel optimization can help. CUDA graphs replace the 400 launches with one, which is why they matter most at small batch.

<div class="take">
<div class="lab">Key takeaway</div>

Nsight Systems shows the whole timeline of CPU, GPU, copies and communication. Capture a short steady-state window, label it with NVTX, and use it to find idle gaps and the kernels worth deeper study.

</div>
