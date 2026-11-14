---
id: "15.04"
course: "inference-eng-course"
module: "15"
slug: "cuda-graph-capture-replay"
title: "CUDA Graph Capture & Replay"
keyIdeas: ["CUDA graphs record a sequence of kernel launches and replay it with one call.", "Removes per-kernel launch overhead; big win for small-batch decode.", "Shapes and memory addresses are frozen at capture time.", "Engines capture graphs for a list of batch sizes and pad to the next size.", "Decode steps are graph-friendly; prefill with variable lengths often runs without graphs or with piecewise graphs."]
question: "What constraints does CUDA graph capture impose, and how do engines handle variable batch sizes with it?"
answer: "A captured graph replays fixed kernels with fixed memory addresses and shapes; no CPU-side logic, host syncs or new allocations can happen inside it, and inputs must be copied into the same static buffers before replay. Engines capture one graph per batch size from a chosen set (e.g. 1, 2, 4, 8, 16 ... 512), and at runtime pad the batch up to the nearest captured size."
sources: [{"title": "PyTorch blog: Accelerating PyTorch with CUDA Graphs", "url": "https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/", "note": "How capture and replay work."}, {"title": "CUDA C++ Programming Guide: CUDA Graphs", "url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs", "note": "Underlying API."}]
tags: ["cuda-graphs"]
---

## How CUDA graphs work

A CUDA graph records a sequence of GPU operations once and replays the whole sequence later with a single launch. The driver stores every kernel's launch parameters, so a replay skips Python, the dispatcher and per-kernel launch costs entirely.

```
static_ids = torch.zeros(B, 1, dtype=torch.long, device="cuda")
static_pos = torch.zeros(B, 1, dtype=torch.long, device="cuda")

# warm up on a side stream, then capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_logits = model_step(static_ids, static_pos)

# every decode step afterwards:
static_ids.copy_(next_ids)
static_pos.copy_(next_positions)
g.replay()
logits = static_logits          # read results from the static output buffer
```

The graph remembers the memory addresses of every tensor it used. To give it new inputs, you copy new values into the same buffers. Outputs appear in the same output buffers after each replay.

## The constraints

- **Fixed shapes.** Every replay runs exactly the same kernels with the same launch dimensions.
- **Fixed addresses.** Inputs, outputs and intermediate tensors live at the same addresses on every replay. PyTorch uses a private memory pool for the graph to guarantee this.
- **No CPU logic inside.** No Python branching on values, no `.item()`, no prints and no new allocations in the captured region. Anything the CPU decided during capture is frozen.
- **Graph-safe randomness.** Random sampling inside the graph needs a generator state that advances correctly on each replay. PyTorch handles this for its own random functions when registered properly.

## Handling variable batch sizes

The batch changes every step, but a graph has a fixed shape. Engines capture one graph per batch size in a list, and pad each step up to the nearest captured size.

| Running sequences | Graph used | Padding slots |
| --- | --- | --- |
| 5 | size 8 | 3 |
| 30 | size 32 | 2 |
| 130 | size 136 or 256, depending on the list | 6 or 126 |

Padded slots compute garbage that is thrown away. That wastes a little compute, far less than the launch overhead saved. A dense list of sizes at small batch (1, 2, 4, 8, then every 8) and a sparser list at large batch keeps padding low where it matters.

Above the largest captured size, engines run eagerly, where overhead matters less anyway.

## The attention complication

Attention depends on per-sequence context lengths and block tables, which change every step. Graph-compatible attention backends read these from fixed-size GPU tensors that the engine updates before each replay. The kernel launch shape stays fixed, sized for the maximum, and the kernel reads the real lengths from memory.

## Piecewise graphs

Full graphs work well for decode, where every step has the same structure. Prefill and mixed steps vary too much. **Piecewise** capture solves this: the engine captures graphs for the parts of the model between attention calls, and runs attention outside the graphs. Since attention is one call per layer, this still removes most launches, and works for any mix of prefill and decode.

<div class="ex">
<div class="lab">Worked example</div>

**what graphs do to TPOT**

Llama 3 8B on an H100, decode at batch 1 to 64, eager compared with CUDA graphs.

Batch 1: 10.1 ms eager, 6.0 ms with graphs, 41% faster.

Batch 8: 10.4 ms eager, 6.5 ms with graphs, 38% faster.

Batch 64: 12.0 ms eager, 9.8 ms with graphs, 18% faster.

The gain shrinks as GPU work grows and hides the CPU overhead, as expected.

</div>

## Debugging problems

- **Wrong outputs after capture**: something read a value on the CPU during capture, freezing it. Look for `.item()` or Python conditions inside the step.
- **Errors during capture**: an operation allocated memory or synchronized. The error usually names it.
- **Out of memory after enabling graphs**: each captured size reserves memory in the graph pool. Capture fewer sizes.

## Deep dive: what a graph replay costs

Replaying a captured graph is one launch, whatever it contains:

⟦M817⟧

At 400 kernels and 5 µs each, that is 2 ms down to about 10 µs per step. The cost moves to memory and startup: every captured batch size keeps its own static input and output buffers, and capture takes time.

~~~dg-raw
%%timeline One decode step's launches, eager versus graph replay (µs, illustrative).
unit = µs
tick = 500
CPU, eager | 0-2000 400 launches c4
GPU, eager | 5-2010 kernels with gaps c3
CPU, graph | 0-10 replay c4
GPU, graph | 10-1700 kernels back to back c3
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

CUDA graphs remove launch overhead by replaying a fixed sequence of kernels from static buffers. Engines capture a list of batch sizes and pad each step, and piecewise capture extends the benefit to mixed prefill and decode steps.

</div>
