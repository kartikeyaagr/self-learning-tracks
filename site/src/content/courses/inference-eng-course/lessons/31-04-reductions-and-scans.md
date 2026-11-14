---
id: "31.04"
course: "inference-eng-course"
module: "31"
slug: "reductions-and-scans"
title: "Reductions and scans"
keyIdeas: ["A **tree reduction** combines $n$ values in $\\log_2 n$ steps; within a block it runs through shared memory and, for the last 32 values, through **warp shuffles** (`__shfl_down_sync`) that exchange registers without touching memory.", "Mark Harris's classic sequence shows what matters: avoid divergent branches and bank conflicts with **sequential addressing**, do the first add while loading, and give each thread many elements so the kernel is bandwidth-bound, not instruction-bound.", "Reductions across the whole grid need either a **second pass** over per-block partials or one `atomicAdd` per block; the second pass is deterministic, the atomic is not for floating point.", "A **prefix sum (scan)** gives every element the sum of all before it. Hillis-Steele takes $\\log n$ steps but $O(n \\log n)$ work; Blelloch's up-sweep/down-sweep is work-efficient at $O(n)$.", "Large scans are hierarchical (scan blocks, scan the block totals, add offsets back) or single-pass with **decoupled look-back**; in practice call CUB's `DeviceScan`."]
question: "Your loss-averaging kernel sums 50 million FP32 values with one `atomicAdd` per thread into a single global float. It is slow and a unit test comparing the result bitwise to a reference fails intermittently. Explain both problems and the design you would use instead."
answer: "Speed: every one of 50 million threads performs an atomic on the same address, and atomics to one address are serialised at the L2, so throughput collapses to that of a single memory location instead of the full bandwidth. Bitwise failures: floating-point addition is not associative, and the order in which atomics land varies between runs, so the rounding differs in the last bits. Replace it with a hierarchical reduction: each thread accumulates many elements in a register via a grid-stride loop, each block reduces its threads with warp shuffles and a small shared-memory step, and each block writes one partial to an array. A second, single-block pass reduces the partials in a fixed order. That reads the data once at near-peak bandwidth and is deterministic. If one launch matters more than determinism, one atomic per block (a few thousand) is fast but still non-deterministic."
sources: [{"title": "Mark Harris, Optimizing Parallel Reduction in CUDA", "url": "https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf", "note": "The seven-step reduction tutorial with bandwidth after each change."}, {"title": "NVIDIA blog: Using CUDA Warp-Level Primitives", "url": "https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/", "note": "Shuffle and vote intrinsics, masks, and why implicit warp synchrony is unsafe since Volta."}, {"title": "GPU Gems 3, Chapter 39: Parallel Prefix Sum (Scan) with CUDA", "url": "https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda", "note": "Hillis-Steele vs Blelloch scans, bank-conflict avoidance, and multi-block scans."}, {"title": "Merrill and Garland, Single-pass Parallel Prefix Scan with Decoupled Look-back (2016)", "url": "https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back", "note": "The algorithm behind CUB's scan, matching memcpy throughput."}]
practice: ["dot-product-2", "max-of-array", "mean-and-variance", "argmax", "vector-sum-reduction"]
tags: ["reduction", "warp shuffle", "prefix sum"]
tools: ["CUDA", "CUB"]
---

## Reductions are everywhere

Sums, maxima, norms, means and variances, softmax denominators, loss averages, argmax for greedy decoding: all are reductions. They are memory-bound (one add per element loaded), so a good reduction reads its input once at close to peak bandwidth. Getting there exercises every idea from the previous lessons. Inference Engineering's Triton lessons ("Coalescing, Tiling & Reductions") show reductions as one-line `tl.sum` calls; this lesson shows the machinery that call compiles into.

## A block-level tree reduction

Each block reduces a chunk of the input to one value. The tree halves the number of active threads at every step:

~~~dg-flow Tree reduction of 8 values: log2(8) = 3 steps, each halving the active threads.
8 partials -(add pairs)-> 4 partials -(add pairs)-> 2 partials -(add pairs)-> *1 result
~~~

The order of pairing matters a lot. Harris's NVIDIA tutorial walks through seven versions on the same problem:

1. **Interleaved addressing with `if (tid % (2*s) == 0)`**: correct, but threads within a warp diverge and the modulo is slow.
2. **Interleaved with a strided index**: no divergence, but shared-memory bank conflicts.
3. **Sequential addressing**: thread `t` adds element `t + s` with `s` halving from `blockDim/2`; active threads are contiguous and conflict-free.
4. **First add during load**: each thread loads and adds two elements before the tree, halving idle threads.
5. to 7. **Unroll the last warp, unroll completely, and process many elements per thread**: the final kernel was around 30 times faster than the first on the hardware of the time.

The lasting lessons are 3, 4 and 7: keep active threads contiguous, and give each thread enough independent work (a grid-stride loop of loads into a register accumulator) that the tree is a small tail on a bandwidth-bound loop.

## Warp shuffles

Inside a warp, threads can read each other's registers directly. `__shfl_down_sync(mask, v, offset)` returns `v` from the thread `offset` lanes higher. Five shuffles reduce 32 values:

```cpp
__device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;                                   // lane 0 holds the warp's total
}
__device__ float block_sum(float v) {           // blockDim.x multiple of 32, <= 1024
    __shared__ float warp_part[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) warp_part[w] = v;
    __syncthreads();
    v = (threadIdx.x < blockDim.x / 32) ? warp_part[lane] : 0.f;
    return w == 0 ? warp_sum(v) : 0.f;          // thread 0 holds the block's total
}
```

No shared memory traffic or barriers are needed within the warp, and the `_sync` mask states which lanes participate; since Volta, threads of a warp are not guaranteed to run in lockstep, so the old trick of relying on implicit warp synchrony without the mask is a bug. Cooperative groups wrap the same operations in a typed API.

## Across the grid

Blocks cannot synchronise inside a launch, so a full reduction is finished one of two ways:

- **Two passes**: each block writes its partial to an array; a second, single-block launch reduces the partials. Deterministic: the same input always gives the same bits.
- **Atomics**: each block does `atomicAdd(out, block_total)`. One launch, but block completion order varies, and floating-point addition is not associative, so results can differ in the last bits between runs. That is acceptable for monitoring, not for tests that compare bitwise.

Use a numerically sensible accumulator too: accumulate FP16 inputs in FP32, and for mean and variance use Welford's update or combine (count, mean, M2) triples in the tree, rather than sum and sum of squares, which cancel catastrophically when the mean is large.

## Prefix sums (scan)

An inclusive scan of $[3, 1, 7, 0, 4]$ is $[3, 4, 11, 11, 15]$; the exclusive scan shifts right and starts with 0: $[0, 3, 4, 11, 11]$. Scans look sequential but parallelise well, and they are the building block for stream compaction (where does each kept element go?), radix sort, ragged-batch offsets (where does sequence $b$ start?), and top-p sampling (cumulative probabilities).

| Algorithm | Steps | Work | Notes
| --- | --- | --- | --- |
| Sequential | $n$ | $n$ | baseline on one thread
| Hillis-Steele | $\log_2 n$ | $n \log_2 n$ | simple; each step adds the element $2^d$ to the left
| Blelloch | $2 \log_2 n$ | $\approx 2n$ | up-sweep builds a sum tree, down-sweep pushes prefixes back
| Decoupled look-back | 1 pass | $\approx n$ | each block publishes its total and looks back at predecessors

For arrays larger than a block, the classic approach has three phases: each block scans its chunk and records its total; a scan over the block totals gives each block its offset; each block adds its offset. That reads and writes the data twice. Merrill and Garland's decoupled look-back does it in one pass, as fast as a memory copy, by having each block publish its aggregate and inspect its predecessors' status flags. CUB (`cub::DeviceScan::InclusiveSum`) and `torch.cumsum` already implement it; write your own only to learn or to fuse the scan into another kernel.

<div class="ex">
<div class="lab">Worked example</div>

**what bandwidth-bound means here**

Summing $2^{28}$ floats (1 GiB) on a GPU with 2 TB/s peak can take no less than about 0.54 ms. A first naive kernel with interleaved addressing might take around 5 ms; a version with sequential addressing, many elements per thread and a warp-shuffle tail typically lands within 10 to 15% of the 0.54 ms floor. Beyond that, nothing helps: the kernel is limited by HBM, not by how it adds.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

reduce with many elements per thread into registers, a shared-memory or shuffle tree for the tail, and a deterministic second pass across blocks when reproducibility matters; for scans, know the work-efficient algorithm and use the library.

</div>
