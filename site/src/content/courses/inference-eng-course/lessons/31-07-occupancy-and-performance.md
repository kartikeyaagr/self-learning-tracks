---
id: "31.07"
course: "inference-eng-course"
module: "31"
slug: "occupancy-and-performance"
title: "Occupancy and performance"
keyIdeas: ["GPUs hide memory latency by switching between resident warps; **occupancy** is active warps per SM divided by the maximum (64 on A100 and H100).", "Occupancy is capped by whichever resource runs out first: threads (2048 per SM), blocks (32 per SM), **registers** (65,536 per SM) or **shared memory** per SM. At 64 registers per thread, a 256-thread block uses 16,384 registers, so only 4 blocks fit: 50% occupancy.", "More occupancy is not always faster: kernels with lots of **instruction-level parallelism** (several independent loads in flight per thread, register tiling) often peak at 25 to 50% occupancy.", "The **roofline** bounds attainable throughput by $\\min(\\text{peak FLOP/s},\\; I \\times \\text{bandwidth})$ with arithmetic intensity $I$ in FLOP/byte; the ridge point (about 10 FLOP/byte for FP32 on A100, roughly 300 for BF16 Tensor Cores on H100) separates memory-bound from compute-bound kernels.", "Work in a loop: compute the kernel's intensity, measure achieved bandwidth or FLOP/s against the relevant roof in Nsight Compute, and only then tune occupancy, layout or fusion."]
question: "A custom attention-score kernel uses 128 registers per thread with 256-thread blocks and no shared memory. Nsight Compute reports 25% occupancy, 35% of peak memory bandwidth and 20% of peak compute, with warps mostly stalled waiting on global loads. Explain the occupancy figure and give two different ways to improve performance."
answer: "At 128 registers per thread, a 256-thread block needs 32,768 registers, so only 2 blocks (16 warps) fit within an SM's 65,536 registers: 16 of 64 warps is 25% occupancy. With neither memory nor compute near its roof and warps stalled on loads, the kernel is latency-bound: there are too few loads in flight to cover HBM latency. One fix is to increase thread-level parallelism: cap registers with `__launch_bounds__(256, 4)` or restructure to use fewer, aiming for 4 blocks per SM, and check that this does not cause spills. The other is to increase instruction-level parallelism and reuse without raising occupancy: issue several independent loads per thread before using them (unrolling, vectorised 16-byte loads), stage tiles in shared memory with double buffering so loads for the next tile overlap computation on the current one, and reuse loaded data across several outputs per thread. Either way, re-measure against the roofline after each change."
sources: [{"title": "CUDA C++ Best Practices Guide: Occupancy", "url": "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy", "note": "How registers, shared memory and block size limit residency, and the occupancy APIs."}, {"title": "Volkov, Better Performance at Lower Occupancy (GTC 2010)", "url": "https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf", "note": "Slides showing instruction-level parallelism hiding latency at low occupancy."}, {"title": "Williams, Waterman and Patterson, Roofline (UC Berkeley tech report, 2008)", "url": "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf", "note": "The original roofline model: arithmetic intensity, ridge points and ceilings."}, {"title": "Horace He, Making Deep Learning Go Brrrr From First Principles", "url": "https://horace.io/brrr_intro.html", "note": "Compute-, memory- and overhead-bound regimes in PyTorch with worked numbers."}]
practice: ["gpu-occupancy-calculator", "analyze-prefill-and-decode-with-roofline-model", "measure-kernel-timeline-utilization"]
tags: ["occupancy", "roofline"]
tools: ["CUDA", "Nsight Compute"]
---

## Latency hiding

A load from HBM takes several hundred cycles. A GPU does not wait: each SM keeps many warps resident and, every cycle, its schedulers issue an instruction from any warp that is ready. If enough warps (or enough independent instructions within each warp) are in flight, some are always ready and the memory pipeline stays busy. **Occupancy** measures the first of those two sources:

$$\text{occupancy} = \frac{\text{active warps per SM}}{\text{maximum warps per SM}}$$

On A100 and H100 the maximum is 64 warps (2048 threads). How many warps are active depends on how many of your blocks fit on an SM at once, which is limited by whichever resource each block consumes most.

## Computing occupancy

| Resource per SM (A100) | Limit | A block of 256 threads using... | Blocks that fit
| --- | --- | --- | --- |
| Threads | 2048 | 256 threads | 8
| Blocks | 32 | 1 block | 32
| Registers | 65,536 | 64 regs/thread = 16,384 | 4
| Shared memory | ~164 KB usable | 48 KB | 3

The binding limit is the minimum: here shared memory allows 3 blocks, so 24 warps are active, 37.5% occupancy. Drop to 32 registers per thread and 16 KB of shared memory, and threads become the limit at 8 blocks: 64 warps, 100%. Two details make hand calculations slightly off: registers are allocated per warp in chunks of 256, and each block reserves a little extra shared memory (1 KB on A100). The CUDA occupancy calculator, the runtime API and Nsight Compute all apply these rules for you.

```cpp
int blocks_per_sm = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, my_kernel, 256, smem_bytes);
int min_grid, best_block;
cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block, my_kernel, smem_bytes, 0);
// nvcc -Xptxas -v  prints registers and shared memory per kernel
// __launch_bounds__(256, 2) asks the compiler to fit 2 blocks of 256 per SM
```

Forcing fewer registers (`-maxrregcount`, `__launch_bounds__`) raises occupancy but can cause **spills** to local memory, which lives in global memory behind the caches. Spills often cost more than the occupancy gains; check the `-Xptxas -v` output for spill stores.

## Occupancy is a means, not a goal

Volkov's GTC 2010 talk, "Better Performance at Lower Occupancy", made the point that has held ever since: latency can be hidden either by many warps (thread-level parallelism) or by each warp having several independent operations in flight (instruction-level parallelism). A matmul thread that computes an 8 x 8 output tile in registers issues many independent FMAs and loads; it uses many registers, so occupancy falls, and it is faster anyway. Fast GEMM and attention kernels commonly run at 12 to 50% occupancy. Low occupancy is worth fixing when a kernel is latency-bound: Nsight Compute shows warps mostly stalled on memory dependencies with too few eligible warps per cycle, and neither bandwidth nor compute anywhere near peak.

## The roofline

The roofline model bounds a kernel's attainable throughput from two numbers: the device's peak compute rate $P$ and peak memory bandwidth $\beta$, and the kernel's arithmetic intensity $I$ = FLOPs performed / bytes moved to and from DRAM.

$$\text{attainable FLOP/s} = \min(P,\; I \cdot \beta)$$

~~~dg-flow Classify a kernel with the roofline before tuning it.
Count FLOPs and bytes -(intensity I)-> Compare with ridge P/β -(I below ridge)-> *Memory-bound: cut bytes -(I above ridge)-> Compute-bound: use Tensor Cores
~~~

| Kernel (A100 80 GB; FP32 ridge ~9.6, BF16 Tensor Core ridge ~153 FLOP/B) | Intensity | Bound by
| --- | --- | --- |
| Vector add, FP32 | 1 FLOP / 12 B = 0.08 | memory
| Softmax or LayerNorm, BF16 | ~2-5 | memory
| Tiled FP32 matmul, 32 x 32 tiles (global traffic) | 8 | memory, near the FP32 ridge
| Square GEMM, BF16, N = 4096 | $2N^3 / (3 \cdot 2N^2) = N/3 \approx 1365$ | compute
| Decode-time GEMV, BF16 weights, batch 1 | ~1 | memory

Inference Engineering's "Compute-Bound vs Memory-Bound & the Roofline Model" applies this to prefill and decode; at the kernel level it gives a target. A memory-bound kernel is done when it reaches 80 to 90% of $\beta$; making its arithmetic faster is pointless, and the remaining options are reducing bytes (fusion, lower precision, avoiding re-reads). A compute-bound kernel is done when it approaches $P$ for the instruction type it uses, which for FP16/BF16 means Tensor Cores; an FP32 CUDA-core matmul on an H100 can never exceed about 7% of the BF16 Tensor Core peak, however well it is tiled.

## A tuning loop

1. Profile end to end first (Nsight Systems) and pick the kernels that dominate time.
2. For each, compute intensity by hand and read Nsight Compute's "Speed of Light" section: achieved percentage of memory and compute peaks.
3. If neither is high, read the stall reasons and occupancy: uncoalesced access (sectors per request), bank conflicts, too few eligible warps, or a grid too small to fill all SMs (fewer blocks than SMs leaves hardware idle; aim for several waves).
4. Change one thing, re-measure, keep a log.

<div class="ex">
<div class="lab">Worked example</div>

**a kernel that is already done**

An RMSNorm kernel on 8192 x 4096 BF16 reads 64 MiB and writes 64 MiB in 46 microseconds on an H100 SXM: 128 MiB / 46 µs = 2.9 TB/s, about 87% of 3.35 TB/s. Its occupancy is 50% and Nsight flags it. Raising occupancy will not help, because there is almost no bandwidth left to gain; only fusing it with the preceding residual add (31.06) or the following matmul's prologue would make the layer faster.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

compute occupancy from registers, shared memory and block size to know what limits residency, but judge a kernel by where it sits under the roofline: push memory-bound kernels towards peak bandwidth by moving fewer bytes, and compute-bound ones towards the Tensor Core roof.

</div>
