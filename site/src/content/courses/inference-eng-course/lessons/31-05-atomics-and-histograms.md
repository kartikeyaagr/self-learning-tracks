---
id: "31.05"
course: "inference-eng-course"
module: "31"
slug: "atomics-and-histograms"
title: "Atomics and histograms"
keyIdeas: ["An **atomic** (`atomicAdd`, `atomicMax`, `atomicCAS`, ...) performs read-modify-write on one memory word as an indivisible operation and returns the old value; it is how independent threads safely update shared counters and bins.", "Atomics to **different** addresses run in parallel; atomics to the **same** address serialise. Throughput therefore depends on the data: a histogram of uniform noise is fast, a histogram of a constant image is a queue.", "**Privatisation** gives each block (or warp) its own copy of the bins in shared memory, accumulates there, then merges with one global atomic per bin per block, cutting global atomics from $n$ to $\\text{blocks} \\times \\text{bins}$.", "**Warp-aggregated atomics** let one leader lane do a single atomic for all lanes that target the same address, using ballot or match intrinsics.", "Floating-point atomics make results order-dependent; PyTorch flags such kernels (`scatter_add_`, `index_add_` on CUDA) as non-deterministic."]
question: "You need per-expert token counts for an MoE layer with 8 experts and 1 million tokens per step, computed on the GPU. A first version does one global `atomicAdd` per token on an array of 8 integers and is surprisingly slow. Explain why, and design a faster kernel. Is the result deterministic?"
answer: "With only 8 counters, a million atomics are concentrated on 8 addresses, so the L2 serialises them: throughput is limited to about 8 concurrent updates, regardless of how many threads run. Use privatisation: each block keeps 8 counters in shared memory (or per warp, to cut shared contention further), threads accumulate with shared-memory atomics or, better, each thread first counts its own grid-stride elements in 8 registers and adds them once. After a barrier, each block issues at most 8 global atomics. With 1024 blocks that is 8192 global atomics instead of a million. Warp-level `__match_any_sync` or ballot-based aggregation can reduce shared atomics further when many lanes pick the same expert. The result is deterministic, because integer addition is associative and the final counts do not depend on the order of the atomics; only floating-point accumulations would vary."
sources: [{"title": "NVIDIA blog: GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell", "url": "https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/", "note": "Privatised histograms and how contention depends on the input image."}, {"title": "NVIDIA blog: CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics", "url": "https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/", "note": "The leader-lane pattern for stream compaction with a shared counter."}, {"title": "CUDA C++ Programming Guide: Atomic Functions", "url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions", "note": "Semantics, supported types and the atomicCAS recipe for custom operations."}, {"title": "PyTorch: Reproducibility", "url": "https://pytorch.org/docs/stable/notes/randomness.html", "note": "Which CUDA operations are non-deterministic and what use_deterministic_algorithms changes."}]
practice: ["sum-of-array", "argmin", "max-of-array", "min-of-array"]
tags: ["atomics", "privatisation"]
tools: ["CUDA", "PyTorch"]
---

## When reductions are not enough

A reduction combines everything into one value along a fixed tree. Many kernels instead scatter updates to data-dependent locations: histograms (which bin?), gradient accumulation for embedding tables (which rows were looked up?), stream compaction with a shared output counter, MoE token counts per expert. Two threads may target the same location at the same time, and a plain `bins[b] += 1` then loses updates: both threads read 7, both write 8. Atomics fix this by making the read-modify-write indivisible.

```cpp
__global__ void hist_naive(const unsigned char* x, unsigned int* bins, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&bins[x[i]], 1u);
}
```

Supported operations include add, sub, min, max, exchange, and/or/xor and compare-and-swap, on 32- and 64-bit integers, `float`, `double` (compute capability 6.0 and up) and `__half2`. Anything else, such as a float max with an index or a custom combine, is built from `atomicCAS` in a retry loop: read the old word, compute the new one, swap only if the word has not changed, otherwise retry. Packing (value, index) into one 64-bit word lets a single CAS update both atomically.

## Contention

Global atomics are executed in the L2 cache. Updates to distinct addresses proceed in parallel at high throughput; updates to the same address are serialised. For a 256-bin histogram of 8-bit pixels, throughput depends on the image: natural photos spread across bins, while a mostly-black frame sends almost every update to bin 0.

~~~dg-flow Privatised histogram: most updates hit fast per-block shared memory; global memory sees one atomic per bin per block.
Input pixels -(shared atomicAdd)-> *Per-block bins in shared memory -(__syncthreads)-> Merge -(one global atomic per bin)-> Global bins
~~~

## Privatisation

Give each block a private histogram in shared memory, initialise it to zero, accumulate with shared-memory atomics, synchronise, then add each non-zero private bin to the global histogram.

```cpp
__global__ void hist_private(const unsigned char* x, unsigned int* bins, int n) {
    __shared__ unsigned int local[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&local[x[i]], 1u);               // on-chip, per SM
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x)
        if (local[b]) atomicAdd(&bins[b], local[b]);
}
```

Shared-memory atomics have been native hardware operations since the Maxwell generation, so they are much cheaper than global ones, and contention is now limited to the threads of one block. Further steps when contention stays high: per-warp sub-histograms (several copies per block, merged at the end), having each thread pre-aggregate runs of identical values in registers before issuing an atomic, and launching only enough blocks to fill the GPU so the merge cost stays small.

| Strategy | Global atomics for $n = 10^8$, 256 bins, 1024 blocks | Worst-case contention
| --- | --- | --- |
| Naive global | $10^8$ | all threads on one address
| Per-block shared bins | at most $1024 \times 256 = 262{,}144$ | threads of one block on one shared address
| Per-warp shared bins | same global count | 32 lanes on one address
| Warp-aggregated | up to 32x fewer than naive | one atomic per distinct address per warp

## Warp-aggregated atomics

When many lanes of a warp target the same address, as when appending to an output buffer through one shared counter, it is wasteful to issue 32 atomics. With `__ballot_sync` (or `__match_any_sync` for per-address groups) the lanes find out how many of them want to update, a leader lane does one `atomicAdd` of that count, broadcasts the returned base offset with a shuffle, and each lane computes its own slot from its rank among the active lanes. Recent compilers do this automatically for the simple case of a uniform address, but the pattern is worth knowing for filtering and compaction kernels.

## Determinism

Integer atomics give the same final value whatever the order. Floating-point atomics do not, because $(a + b) + c \ne a + (b + c)$ in floating point and the order of arrival varies run to run. This is why PyTorch's reproducibility notes list CUDA `index_add_` and `scatter_add_` among operations that are non-deterministic by default; with `torch.use_deterministic_algorithms(True)` such operations either switch to slower deterministic implementations (typically sort-then-reduce) or raise an error. If a test needs bitwise reproducibility, replace float atomics with a reduction in a fixed order.

<div class="ex">
<div class="lab">Worked example</div>

**contention in numbers**

Take a 256-bin histogram over 100 million bytes. On random bytes the naive kernel spreads its atomics over 256 addresses; on an all-zero input every one of the $10^8$ updates queues on a single L2 address, and the same kernel can slow down by an order of magnitude or more. The privatised version stays close to bandwidth-bound for random data and degrades far less on constant data, because the queue now lives in each SM's shared memory and the global merge is only 262,144 atomics.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

use atomics for scattered, data-dependent updates, expect throughput to depend on how many threads collide, and privatise into shared memory (or aggregate per warp) so that global memory sees a few atomics per block rather than one per element.

</div>
