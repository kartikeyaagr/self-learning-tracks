---
id: "12.04"
course: "inference-eng-course"
module: "12"
slug: "nsight-compute"
title: "Nsight Compute"
keyIdeas: ["Nsight Compute (ncu) profiles individual kernels in depth.", "Speed of Light section: percent of peak compute and memory throughput.", "Memory workload analysis: cache hit rates, bytes moved.", "Occupancy and warp stall reasons explain why a kernel underperforms.", "Profile only a few kernel launches; ncu replays kernels and is slow."]
question: "Nsight Compute reports a decode attention kernel at 85% of DRAM bandwidth and 8% of compute throughput. What does that tell you and what would make it faster?"
answer: "The kernel is memory bound and already near the bandwidth limit, so tuning its compute will not help. It can only get faster by moving fewer bytes: quantizing the KV cache (FP8), using GQA-aware kernels that read each KV head once for all its query heads, or avoiding redundant reads."
sources: [{"title": "NVIDIA Nsight Compute documentation", "url": "https://docs.nvidia.com/nsight-compute/NsightCompute/index.html", "note": "Metrics and sections."}, {"title": "Roofline model (Wikipedia)", "url": "https://en.wikipedia.org/wiki/Roofline_model", "note": "Interpreting compute vs memory limits."}]
tags: ["profiling", "nsight"]
---

## Inside one kernel

Nsight Compute (`ncu`) profiles individual kernels in depth. It replays each captured kernel several times while reading hardware counters, then reports how close the kernel came to the GPU's limits and why it did not get closer. Use it after Nsight Systems has told you which kernels matter.

## Capturing

Kernel replay is slow, so capture only a few launches of the kernels you care about:

```
ncu --set full -k regex:"attention|gemm" \
    --launch-skip 50 --launch-count 3 \
    -o kernels python bench_step.py
```

- `-k` filters kernels by name.
- `--launch-skip` skips warm-up launches.
- `--launch-count` limits how many are profiled.

Open the report in the Nsight Compute GUI, or print sections with `ncu --import kernels.ncu-rep --page details`.

## The sections that matter

| Section | What it tells you |
| --- | --- |
| GPU Speed of Light | percentage of peak SM throughput and of peak memory throughput |
| Roofline | where the kernel sits relative to the compute and memory ceilings |
| Memory Workload Analysis | bytes from DRAM, L2 and L1 hit rates |
| Occupancy | active warps compared with the maximum the SM supports |
| Warp State Statistics | why warps were stalled: waiting for memory, dependencies, barriers |
| Source counters | which lines of code or instructions take the time |

## Reading Speed of Light

Start with the two headline percentages.

- **High memory throughput, low compute throughput**: memory bound. The kernel is limited by moving bytes. Reduce the bytes: lower precision, fusion with neighbouring kernels, or better reuse of loaded data.
- **High compute, low memory**: compute bound. Use lower-precision math, better instruction choices or less work.
- **Both low**: latency bound. The kernel is neither moving data nor computing at full rate. Common causes are too few thread blocks to fill the GPU, low occupancy, long dependency chains or synchronization. Look at warp stall reasons and launch configuration.

## Typical inference kernels

| Kernel | Usual bound | Good result |
| --- | --- | --- |
| Decode matmul, small batch | memory | 80 to 90% of peak bandwidth |
| Prefill matmul | compute | 60 to 80% of peak tensor FLOPs |
| Decode attention | memory | 70 to 90% of peak bandwidth |
| Prefill attention (FlashAttention) | compute | 50 to 75% of peak tensor FLOPs |
| RMSNorm, RoPE, residual adds | memory | 70 to 90% of peak bandwidth, if large enough |

<div class="ex">
<div class="lab">Worked example</div>

**a decode attention kernel at 35% of bandwidth**

Nsight Compute shows a decode attention kernel at 35% of peak memory throughput and 8% of compute, with only 66 thread blocks launched.

The H100 has 132 SMs. With batch 8 and 8 KV heads, the kernel launched one thread block per sequence and head, 64 in total plus overhead, so half the SMs were idle.

Switching to a split-KV kernel that divides each long sequence across several thread blocks raised it to 78% of peak bandwidth, cutting attention time by more than half.

</div>

## A note on small kernels

For kernels that run for only a few microseconds, such as norms at small batch, Speed of Light percentages are often low because the launch and ramp-up dominate. The fix there is usually fusion or CUDA graphs, not optimizing the kernel itself.

## Locking clocks

Nsight Compute locks GPU clocks to base frequency by default during profiling, so its timings can be slower than in production. Its percentages of peak are still meaningful. Use `--clock-control none` if you need timings at production clocks.

## Deep dive: Little's law inside the memory system

To reach full bandwidth, enough bytes must be in flight to cover the memory latency:

⟦M692⟧

For an H100 at 3.35 TB/s and roughly 600 ns of latency, that is about 2 MB in flight across the chip, or about 15 KB per SM. A kernel with too few warps per SM, or one that waits for each load before issuing the next, cannot keep 15 KB outstanding. It reaches low bandwidth even though nothing is "slow".

<div class="ex">
<div class="lab">Worked example</div>

**why a decode attention kernel reaches 35% of bandwidth**

At batch 4 and 32 heads, the kernel launches 128 thread blocks on 132 SMs, one per SM. Each loads a 16-token KV block (32 KB of K and V per layer), waits, computes, then loads the next.

In flight per SM: at most one block's loads, often less while it computes. Bandwidth is about ⟦m694⟧ (bytes in flight per SM) / latency, far below peak.

Splitting each sequence across 4 thread blocks (split-K, 08.03) and prefetching the next KV block while computing the current one raises bytes in flight 4 to 8 times, and bandwidth to 70 to 80%.

</div>

## Deep dive: speed of light, defined

Nsight Compute reports, for each kernel,

⟦M693⟧

A kernel is well optimized when the larger of the two is above about 80%. If both are low, the kernel is latency bound: not enough parallel work, the case above.

<div class="take">
<div class="lab">Key takeaway</div>

Nsight Compute tells you how close a kernel is to the hardware limit and which limit it hits. High memory throughput means cut bytes, high compute means cut or cheapen FLOPs, and both low means look at parallelism and stalls.

</div>
