---
id: "15.01"
course: "inference-eng-course"
module: "15"
slug: "eager-execution-runtime-overhead"
title: "Eager Execution & Runtime Overhead"
keyIdeas: ["Eager mode runs Python and dispatches each op one by one.", "Each op costs microseconds of CPU time; a decode step has hundreds of ops.", "When kernels are short (small batch decode), the GPU waits on the CPU.", "Overhead shows as GPU idle gaps in timelines.", "Fixes: fuse ops (fewer kernels) and capture graphs (fewer launches)."]
question: "At batch size 1 decode, kernel time is 6 ms but step time is 11 ms. Where do the other 5 ms go?"
answer: "CPU-side overhead: Python interpreting the model code, PyTorch dispatching each op, building kernel arguments and launching hundreds of kernels, plus scheduler work and synchronization between steps. The GPU sits idle between kernels waiting for launches. Compilation and CUDA graphs target exactly this."
sources: [{"title": "Horace He: Making Deep Learning Go Brrrr", "url": "https://horace.io/brrr_intro.html", "note": "Overhead as the third bottleneck."}, {"title": "PyTorch blog: Accelerating PyTorch with CUDA Graphs", "url": "https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/", "note": "Launch overhead removal."}]
tags: ["runtime"]
---

## Where eager mode spends time

In eager PyTorch, every operation goes through the same path before any GPU work starts: the Python interpreter, PyTorch's dispatcher, checks on dtypes, devices and shapes, allocation of the output tensor, and finally a CUDA kernel launch. That path typically costs 5 to 20 microseconds of CPU time per operation. The GPU work for one operation can be shorter than that.

A decode step for a 32-layer model runs hundreds of operations: projections, norms, rotary embeddings, attention, activations, residual adds, KV cache writes and sampling. At 10 µs each, 600 operations cost 6 ms of CPU time per step.

## When it matters

| Situation | GPU time per operation | Overhead share of the step |
| --- | --- | --- |
| Prefill with large chunks | milliseconds | negligible |
| Decode at batch 256 | tens of microseconds | small |
| Decode at batch 1 | a few microseconds | can be half the step or more |

The CPU and GPU run asynchronously: the CPU queues launches and moves on. As long as the CPU stays ahead, overhead is hidden. When each kernel finishes faster than the CPU can launch the next one, the GPU starts waiting, and the step takes as long as the CPU needs.

<div class="ex">
<div class="lab">Worked example</div>

**overhead at batch 1**

Llama 3 8B, decode at batch 1 on an H100. The weight read is about 5 ms, so a perfect step would take a little over 5 ms.

Eager PyTorch launches about 700 kernels per step at roughly 12 µs each: 8.4 ms of CPU time. The GPU finishes each kernel quickly and waits for the next launch.

Measured step time: about 10 ms. Half the step is overhead, and TPOT is twice what the hardware allows.

At batch 128, the GPU work per step rises to about 15 ms, the CPU stays ahead, and the same 8.4 ms of CPU work is hidden.

</div>

## How to see it

In Nsight Systems or the PyTorch profiler, overhead has a clear signature:

- The GPU row shows short kernels separated by gaps.
- The CPU row is continuously busy with Python and launch calls.
- The sum of GPU kernel times is clearly less than the step time.

A quick test: measure the step time at batch 1 and at batch 8. If they are nearly identical and both well above weight bytes ÷ bandwidth, the step is overhead bound.

## Other sources of host overhead

Kernel launches are not the only CPU cost in a serving loop:

- **Building inputs**: creating position, slot mapping and block table tensors in Python every step.
- **Host-device synchronization**: `.item()`, `.tolist()` or `.cpu()` calls that make the CPU wait for the GPU, then make the GPU wait for the CPU.
- **Scheduling**: deciding which sequences run, which can take milliseconds at large batch.
- **Detokenization and streaming**: if done in the same process and thread.

## The fixes

- **Fusion** with torch.compile or custom kernels: fewer operations and fewer launches.
- **CUDA graphs**: record the whole step's sequence of kernels once and replay it with one launch.
- **Asynchronous scheduling**: prepare the next step's inputs on the CPU while the GPU runs the current one, so CPU work overlaps GPU work.
- **Process separation**: move tokenization, detokenization and HTTP handling to other processes.
- **Persistent buffers**: keep input tensors on the GPU and update them in place instead of rebuilding them.

The rest of this module covers the first two in detail.

## Deep dive: when the host sets the step time

With asynchronous launches, the CPU and GPU work in parallel, and a step takes as long as the slower of the two:

⟦M803⟧

Eager PyTorch spends roughly 10 to 50 µs of Python and dispatcher time per op. A decode step with about 1,000 ops can therefore take 10 to 50 ms on the CPU, whatever the GPU does.

~~~dg-raw
%%chart Predicted step time against batch size, eager (18 ms of host time) versus CUDA graphs (0.3 ms), Llama 3 8B on an H100.
x = 1, 16, 64, 128, 256, 512
logx = 1
xlab = Batch size
ylab = ms per step
Eager: 18, 18, 18, 18, 18, 18
CUDA graphs: 6.0, 6.0, 6.0, 6.0, 6.7, 13.1
GPU work alone*: 5.7, 5.7, 5.7, 5.7, 6.4, 12.8
%%
~~~

The graph line adds about 0.3 ms of host work per step that does not overlap (scheduling and sampling). In eager mode, step time is flat until the GPU work exceeds 18 ms, near batch 700. Every batch size below that is host-bound, so making kernels faster changes nothing. With graphs, the step tracks the GPU work, and kernel improvements show up directly.

<div class="take">
<div class="lab">Key takeaway</div>

at small batch, the CPU can be the bottleneck, because launching hundreds of tiny kernels takes longer than running them. Measure GPU idle time in a profile, and if it is large, compilation and CUDA graphs are the cheapest wins.

</div>
