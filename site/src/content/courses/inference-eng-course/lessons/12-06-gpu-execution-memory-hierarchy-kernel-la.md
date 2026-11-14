---
id: "12.06"
course: "inference-eng-course"
module: "12"
slug: "gpu-execution-memory-hierarchy-kernel-launches"
title: "GPU Execution, Memory Hierarchy & Kernel Launches"
keyIdeas: ["A GPU has many streaming multiprocessors (SMs); each runs many threads in groups of 32 (warps).", "Memory hierarchy: registers, shared memory/L1, L2, HBM. Each level is bigger and slower.", "Kernels are launched from the CPU; each launch has microseconds of overhead.", "Fast kernels keep data in fast memory and reuse it before going back to HBM.", "Occupancy (active warps per SM) helps hide memory latency."]
question: "Order the GPU memory levels from fastest and smallest to slowest and largest, and say where a FlashAttention kernel keeps its tiles."
answer: "Registers (per thread), shared memory and L1 (per SM, ~200 KB), L2 cache (tens of MB, shared), HBM global memory (tens of GB). FlashAttention loads tiles of Q, K and V from HBM into shared memory and registers, computes on them there, and writes only the output back to HBM, avoiding storing the full attention matrix in HBM."
sources: [{"title": "CUDA C++ Programming Guide", "url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/", "note": "Execution and memory model."}, {"title": "Modal GPU Glossary", "url": "https://modal.com/gpu-glossary", "note": "Short definitions of SMs, warps and memory levels."}]
tags: ["gpu"]
---

## The execution model

To understand why kernels are fast or slow, you need a working model of how a GPU runs code.

- **Streaming multiprocessor (SM).** The GPU's basic processing unit. An H100 SXM has 132. Each SM has its own tensor cores, CUDA cores, registers and shared memory.
- **Thread block.** A group of threads that runs on one SM and can share that SM's shared memory. A kernel launches many thread blocks.
- **Warp.** 32 threads that execute the same instruction together. The SM switches between warps to hide latency: while one warp waits for memory, another computes.
- **Kernel.** A function launched over a grid of thread blocks. The GPU distributes the blocks across SMs.

A kernel with fewer thread blocks than SMs leaves SMs idle. A kernel with too few warps per SM cannot hide memory latency. Both are common at small batch sizes.

## The memory hierarchy

| Level | Size on an H100 | Bandwidth | Shared by |
| --- | --- | --- | --- |
| Registers | about 256 KB per SM | highest | one thread |
| Shared memory and L1 | up to about 228 KB per SM | very high, tens of TB/s in total | one thread block |
| L2 cache | 50 MB | several times HBM | the whole GPU |
| HBM | 80 GB | about 3.35 TB/s | the whole GPU |

These are approximate values; check your GPU's datasheet. The ratios are what matter: each step down the hierarchy is larger and slower.

## Why the hierarchy decides performance

A fast kernel loads data from HBM once, reuses it many times from shared memory and registers, and writes its results once. A slow one makes repeated round trips to HBM.

Consider three operations in a row: a residual add, an RMSNorm and a multiplication by a gain. Run as three separate kernels on a [256, 4096] BF16 tensor:

- The add reads 2 tensors and writes 1: 6 MB.
- The norm reads 1 and writes 1: 4 MB.
- The multiply reads 1 and writes 1: 4 MB.

Total traffic: 14 MB. A fused kernel reads the two inputs once and writes the output once: 6 MB. Same math, less than half the traffic, and two fewer launches.

FlashAttention, fused norms, fused SwiGLU and quantized matmuls all follow the same pattern: load once, keep data in fast memory, write once.

## Kernel launches

Launching a kernel costs CPU and driver time, typically 3 to 10 microseconds through PyTorch. A decode step for a 32-layer model can launch 500 or more kernels: projections, norms, RoPE, attention, activation, residual adds, cache writes and sampling.

<div class="ex">
<div class="lab">Worked example</div>

**launch overhead at small batch**

600 kernels per step at 5 µs of launch time each is 3 ms of CPU time per step.

At batch 1 the GPU work for an 8B model is about 5 ms. If the CPU cannot launch fast enough to stay ahead, the GPU waits, and the step stretches toward 8 ms or more.

At batch 128, GPU work rises to about 15 ms, the CPU stays ahead, and launch overhead mostly disappears behind the GPU work.

</div>

This is why overhead matters most at small batch, and why CUDA graphs, which record a sequence of launches and replay it with one call, help small-batch decode the most.

## Occupancy and tail effects

Two more effects show up in profiles:

- **Occupancy.** Each SM can hold a limited number of warps, limited by registers and shared memory per thread block. A kernel that uses many registers per thread fits fewer warps and hides latency less well.
- **Wave quantization.** If a kernel launches 140 thread blocks on 132 SMs, the first wave runs 132 blocks and the second runs only 8, leaving most SMs idle for that second wave. Good kernels choose tile sizes so the block count divides evenly across SMs.

## Deep dive: wave quantization

A kernel split into ⟦m700⟧ tiles on ⟦m701⟧ SMs runs in ⟦m702⟧ waves. If the last wave is partly empty, part of the GPU idles:

⟦M699⟧

~~~dg-raw
%%chart Wave efficiency against tile count on a 132-SM H100.
x = 100, 132, 133, 200, 264, 265, 396, 1000
logx = 1
xlab = Tiles in the kernel
ylab = Efficiency (%)
Efficiency: 76, 100, 50, 76, 100, 67, 100, 95
%%
~~~

133 tiles is the worst case: one tile left for a whole second wave, 50% efficient. Matmul libraries choose tile sizes partly to avoid this. At large ⟦m703⟧ the effect fades.

## Deep dive: the memory hierarchy by capacity

| Level | Size on an H100 SXM | Shared by |
| --- | --- | --- |
| Registers | 256 KB per SM | One thread's values |
| Shared memory / L1 | up to 228 KB per SM | One thread block |
| L2 cache | 50 MB | The whole GPU |
| HBM3 | 80 GB at 3.35 TB/s | The whole GPU |

Each step down is roughly an order of magnitude slower to reach. Fast kernels load a tile from HBM once, reuse it many times from shared memory and registers, and write the result once. FlashAttention is this idea applied to attention.

<div class="take">
<div class="lab">Key takeaway</div>

GPUs are fast when work stays in registers and shared memory and keeps every SM busy. Every trip to HBM and every kernel launch costs time, so good kernels load once, fuse neighbouring operations and launch few, well-sized grids.

</div>
