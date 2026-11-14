---
id: "06.01"
course: "inference-eng-course"
module: "06"
slug: "why-plain-pytorch-serving-breaks-down"
title: "Why Plain PyTorch Serving Breaks Down"
keyIdeas: ["Static or no batching wastes the GPU and makes latency depend on others' lengths.", "Per-request KV allocation wastes memory through over-reservation and fragmentation.", "Python overhead and many small kernel launches dominate decode at small batch.", "No prefix reuse, no preemption, no admission control.", "Engines like vLLM, SGLang and TensorRT-LLM solve these together."]
question: "List four specific inefficiencies of serving with model.generate behind an HTTP server."
answer: "Requests run one at a time or in static batches, so the GPU is underused and short requests wait for long ones. KV cache is pre-allocated or grown per request, wasting and fragmenting memory. Python and kernel-launch overhead dominate small decode steps without CUDA graphs. There is no shared-prefix reuse, so identical system prompts are recomputed for every request."
sources: [{"title": "Efficient Memory Management for LLM Serving with PagedAttention", "url": "https://arxiv.org/abs/2309.06180", "note": "The vLLM paper, describing these problems."}, {"title": "Orca", "url": "https://www.usenix.org/conference/osdi22/presentation/yu", "note": "Iteration-level scheduling."}]
tags: ["engines"]
---

## Five problems at once

The FastAPI server from Module 01 works for one user. Under real traffic it fails in five separate ways, and each failure needs a different fix. Dedicated inference engines exist because fixing all five together takes a different architecture, not a few patches.

1. **Batching.** The plain server runs one request at a time, so the GPU runs at batch 1 and wastes most of its capacity. Static batching, which waits for a group of requests and runs them together, helps a little but makes short requests wait for long ones.
2. **Memory.** Each request reserves KV space for its maximum possible length. Most requests use a fraction of it. Memory runs out long before compute does.
3. **Overhead.** Each decode step launches hundreds of small kernels from Python. At small batch, the GPU often waits for the CPU between kernels.
4. **Redundant work.** Every request with the same system prompt recomputes that prompt's KV cache from scratch.
5. **No control.** There is no queue with limits, no priorities and no way to pause a request when memory runs short. Overload turns into timeouts and crashes.

## Why static batching is not enough

Static batching groups requests that arrive close together and runs them as one batch until all finish.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of waiting for the longest request**

A static batch holds 8 requests. Seven want about 50 output tokens, and one wants 1,000.

The batch runs for 1,000 steps. After step 50, seven slots sit idle for 950 steps. Average slot utilization is (7 × 50 + 1,000) ÷ (8 × 1,000) ≈ 17%.

Meanwhile, new requests wait for the whole batch to finish before they can start. Their TTFT includes up to 1,000 steps of someone else's generation.

</div>

Padding makes it worse. Prompts of different lengths must be padded to the longest one, and the padded positions still cost compute and memory.

## What an engine adds

| Problem | Engine feature | Module |
| --- | --- | --- |
| Idle slots and waiting | continuous batching: admit and retire requests every step | 07 |
| KV over-reservation | paged KV cache in small blocks, allocated on demand | 08 |
| Launch overhead | CUDA graphs and compiled or fused kernels | 15 |
| Recomputed prompts | prefix caching of shared KV blocks | 10 |
| Overload | a scheduler with queues, token budgets and preemption | 07 and 23 |
| Long prompts stalling others | chunked prefill mixed with decode | 10 |

These features depend on each other. Continuous batching needs a cache that can grow and shrink per request, which needs paging. Prefix caching needs blocks that can be shared, which also needs paging. CUDA graphs need fixed shapes, which needs a runtime that pads batches to a few captured sizes. That is why engines are designed around them from the start.

## The main open engines

- **vLLM**: introduced PagedAttention. Very wide model and hardware support, OpenAI-compatible server, large community.
- **SGLang**: known for RadixAttention prefix caching, a fast scheduler and strong structured-output support.
- **TensorRT-LLM**: NVIDIA's engine, compiled per model and GPU, often fastest on NVIDIA hardware, with more setup work.
- **Hugging Face TGI** and **llama.cpp** (for CPUs, Apple silicon and small GPUs) cover other niches.

They share the same core ideas. Learning one well makes the others easy to read.

## The expected gain

On a mixed chat workload, moving from a plain PyTorch server to a modern engine often raises throughput by 10 to 30 times at the same latency target. The original vLLM paper reported 2 to 4 times higher throughput than earlier serving systems that already batched, and much larger gains over plain Hugging Face generation. Most of the gain comes from batching and memory management. Kernel improvements add a further, smaller share.

## Deep dive: how much a static batch wastes, exactly

A static batch of ⟦m459⟧ requests runs until its longest member finishes. Slot utilization is

⟦M457⟧

If output lengths are exponential with mean ⟦m460⟧, the expected maximum of ⟦m461⟧ of them is ⟦m462⟧, where ⟦m463⟧. So

⟦M458⟧

which falls without limit as the batch grows. Real output lengths have heavier tails than exponential, so real numbers are worse.

~~~dg-raw
%%chart Slot utilization of a static batch with exponential output lengths.
x = 1, 2, 4, 8, 16, 32, 64, 128
logx = 1
xlab = Batch size
ylab = Busy slots (%)
Static batching: 100, 67, 48, 37, 30, 25, 21, 18
Continuous batching*: 100, 98, 97, 96, 95, 95, 94, 94
%%
~~~

The continuous-batching line is not 100% because a freed slot waits up to one step for a new request, and because admission is limited by KV space.

<div class="ex">
<div class="lab">Worked example</div>

**the engine's gain predicted from ⟦m464⟧**

At batch 32 with exponential lengths, static batching keeps 25% of slots busy and continuous batching about 95%.

If decode is memory-bound, a step costs about the same either way, so throughput rises by ⟦m465⟧ times from scheduling alone, before any kernel work.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a naive server fails on batching, memory, overhead, reuse and control all at once. Real engines are built around continuous batching, paged KV memory, CUDA graphs, prefix caching and a scheduler, and these features depend on each other.

</div>
