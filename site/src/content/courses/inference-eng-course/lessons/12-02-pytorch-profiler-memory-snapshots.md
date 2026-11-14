---
id: "12.02"
course: "inference-eng-course"
module: "12"
slug: "pytorch-profiler-memory-snapshots"
title: "PyTorch Profiler & Memory Snapshots"
keyIdeas: ["torch.profiler records CPU ops and CUDA kernels with timings and shapes.", "Export to Chrome trace format and view in Perfetto.", "key_averages().table() gives a quick top-kernels summary.", "Memory snapshots record allocations with stack traces; view them in the PyTorch memory visualizer.", "Use a schedule (wait, warmup, active) to profile steady-state steps only."]
question: "What does the PyTorch memory snapshot show that torch.cuda.max_memory_allocated does not?"
answer: "It records every allocation and free with its size, time and Python stack trace, so you can see which code allocated memory, how the peak was formed, and where fragmentation occurs in the caching allocator. max_memory_allocated only reports a single peak number."
sources: [{"title": "PyTorch: Understanding CUDA Memory Usage", "url": "https://pytorch.org/docs/stable/torch_cuda_memory.html", "note": "Memory snapshots and the visualizer."}, {"title": "PyTorch Profiler recipe", "url": "https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html", "note": "Profiling basics."}]
tags: ["profiling", "pytorch"]
---

## The PyTorch profiler

The PyTorch profiler records which operators ran on the CPU, which CUDA kernels they launched, and how long each took. It needs no extra tools and works anywhere PyTorch runs, which makes it the quickest way to get a first breakdown.

```
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=5),
    record_shapes=True,
) as prof:
    for _ in range(9):
        run_decode_step()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace("trace.json")
```

The `schedule` skips two steps, records two warm-up steps without keeping them, then keeps five. This excludes one-time costs such as lazy compilation and allocator growth, so you see the steady state.

## Reading the summary table

| Column | What to use it for |
| --- | --- |
| CUDA time total | which kernels and operators dominate |
| Calls | many tiny calls point to launch overhead |
| CUDA time average | which single kernels are slow |
| Input shapes | confirm the batch and sequence sizes you meant to test |
| CPU time total | Python and operator dispatch cost |

A useful first check: add up the CUDA time per step and compare it with the wall-clock step time. If the GPU is busy 6 ms out of a 10 ms step, 40% of the step is overhead before you even look at kernels.

## Reading the trace

Open `trace.json` in Perfetto (ui.perfetto.dev) or Chrome's trace viewer. You will see:

- **CPU rows**: Python functions and PyTorch operators, with the launch calls they make.
- **GPU rows**: kernels as they actually executed.
- **Flow arrows**: links from each launch to its kernel.

Look for gaps on the GPU row, long CPU stretches with no GPU work, and synchronizations such as `.item()` or `.cpu()` calls that force the CPU to wait for the GPU.

## Memory snapshots

For memory problems, the profiler's companion is the memory snapshot:

```
torch.cuda.memory._record_memory_history(max_entries=100_000)
run_workload()
torch.cuda.memory._dump_snapshot("mem.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

Drag `mem.pickle` into the PyTorch memory visualizer (pytorch.org/memory_viz). It shows:

- a timeline of total allocated memory,
- every allocation as a band, sized by bytes,
- the Python stack trace that made each allocation, when you click it.

This is the fastest way to answer questions such as "what is the 3 GB spike during long prefills?" or "why does memory creep up over an hour?".

<div class="ex">
<div class="lab">Worked example</div>

**finding an activation spike**

An engine crashed with out-of-memory errors only on prompts longer than 24,000 tokens, although the KV pool was sized correctly.

A memory snapshot of one such prefill showed a 4.7 GB allocation from the sampler: it computed logits in FP32 for every prompt position, 24,000 × 128,256 × 4 bytes ≈ 12 GB in the worst case, before slicing the last one.

Computing logits only for the final position removed the spike. The startup profiling had used a shorter prompt and never saw it.

</div>

## Limits of the PyTorch profiler

- It sees what PyTorch launches. Custom C++ schedulers, separate processes and non-PyTorch libraries may be invisible.
- With CUDA graphs, replayed kernels may appear as a single graph launch, hiding the breakdown. Profile with graphs disabled to see individual kernels, then with graphs on to see the real step time.
- Profiling adds overhead, especially with `record_shapes` and stack traces. Treat absolute CPU times with some suspicion.

For a full-system view, use Nsight Systems, covered next.

## Deep dive: three numbers from one summary table

From the profiler's per-kernel table you can compute, without opening the trace:

⟦M683⟧

<div class="ex">
<div class="lab">Worked example</div>

**a decode profile in three lines**

100 decode steps take 950 ms. The table shows 41,000 kernels with 620 ms of total CUDA time.

GPU busy: 65%. Mean kernel: 15 µs. Launch rate: 43,000 per second, 410 per step.

A mean kernel of 15 µs is close to the few-microsecond cost of launching one from Python. A third of the wall time has no kernel running. Both point at launch overhead, not slow kernels.

</div>

## Deep dive: activation memory, predicted

Peak activation memory in a forward pass is roughly the largest set of tensors alive at once. For the MLP at ⟦m685⟧ tokens:

⟦M684⟧

for the input, the gate and up outputs, and their product. At ⟦m686⟧, ⟦m687⟧, ⟦m688⟧: about 770 MB. A memory snapshot whose peak is several times higher points to tensors kept alive by mistake, such as logits for every position instead of only the last.

<div class="take">
<div class="lab">Key takeaway</div>

the PyTorch profiler shows which operators and kernels cost time, and memory snapshots show who allocated what and when. Profile only steady-state steps, and compare busy GPU time with wall-clock step time to find overhead.

</div>
