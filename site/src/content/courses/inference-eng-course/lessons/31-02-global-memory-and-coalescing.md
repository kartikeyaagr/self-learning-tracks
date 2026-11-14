---
id: "31.02"
course: "inference-eng-course"
module: "31"
slug: "global-memory-and-coalescing"
title: "Global memory and coalescing"
keyIdeas: ["Global memory (HBM, about 2 to 3.4 TB/s on A100/H100) is accessed in **32-byte sectors**; a warp's load is efficient only if its 32 addresses fall into as few sectors as possible.", "**Coalesced**: 32 threads reading 32 consecutive 4-byte words touch 128 bytes = 4 sectors, 100% of fetched bytes used. A stride of 2 doubles the sectors (50%); a stride of 8 or more floats means 32 sectors and 12.5% efficiency.", "Layout decides coalescing: map `threadIdx.x` to the contiguous dimension, prefer **structure of arrays** over arrays of structures, and stage strided accesses through shared memory when they cannot be avoided.", "Alignment matters less than it used to but still costs: an access offset by one element touches 5 sectors instead of 4. `cudaMalloc` returns 256-byte-aligned pointers; vector loads such as `float4` need 16-byte alignment.", "Judge memory-bound kernels by **effective bandwidth** $(\\text{bytes read} + \\text{bytes written}) / t$ against the device peak; well-written kernels reach 80 to 90% of it."]
question: "A row-wise L2-normalisation kernel for a 65536 x 1024 float matrix assigns one thread per row; each thread loops over its 1024 columns to compute the norm, then again to scale. It achieves about 150 GB/s on a GPU with 2 TB/s peak. Diagnose it and describe a better mapping."
answer: "With one thread per row, the 32 threads of a warp read the same column index of 32 different rows in each loop iteration. Rows are 4 KB apart, so every warp load touches 32 sectors and uses only 4 bytes of each 32-byte sector: 12.5% efficiency at best, and caches cannot absorb a working set this large. Map a warp (or a block) to each row instead: threads stride across the row's columns with step 32, so each warp load reads 128 consecutive bytes. Each thread accumulates a partial sum of squares, the warp combines partials with shuffle reductions, and then the same threads scale their elements, ideally from registers so the row is read once. Using `float4` loads further cuts instruction count. This mapping should approach 80 to 90% of peak bandwidth, around ten times faster."
sources: [{"title": "CUDA C++ Best Practices Guide: Coalesced Access to Global Memory", "url": "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory", "note": "Diagrams of aligned, misaligned and strided warp accesses and their transactions."}, {"title": "NVIDIA blog: How to Access Global Memory Efficiently in CUDA C/C++ Kernels", "url": "https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/", "note": "Offset and stride experiments with measured bandwidth curves."}, {"title": "NVIDIA: GPU Performance Background User's Guide", "url": "https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html", "note": "Memory versus math bound operations in DL, with A100 numbers."}]
practice: ["global-memory-coalescing-counter", "l1-normalization", "cosine-similarity-2", "matrix-vector-multiplication"]
tags: ["coalescing", "memory bandwidth"]
tools: ["CUDA", "Nsight Compute"]
---

## Why memory access patterns dominate

Most kernels in ML that are not matrix multiplies (activations, normalisation, softmax, elementwise fusions, embedding gathers) do a few arithmetic operations per byte they load. For them the arithmetic is nearly free and the only question is how close they get to peak memory bandwidth. An A100 80 GB delivers about 2.0 TB/s from HBM; an H100 SXM about 3.35 TB/s. A kernel that wastes 7 of every 8 bytes it fetches runs at an eighth of that, whatever else you optimise.

~~~dg-layers The memory hierarchy as seen by one kernel, fastest and smallest at the top.
Registers | per thread, private; ~255 x 32-bit max per thread
Shared memory / L1 | per SM, up to ~228 KB on H100; programmer-managed or cache
L2 cache | shared by all SMs: 40 MB (A100), 50 MB (H100)
*Global memory (HBM) | 40-80 GB; 2-3.4 TB/s; all loads and stores land here eventually
~~~

## Sectors and coalescing

The memory system serves requests in 32-byte sectors (four sectors make a 128-byte cache line). When a warp executes a load, the hardware collects the 32 addresses and fetches every distinct sector they touch. "Coalescing" is just the condition that those addresses are packed: the fewer sectors per warp request, the more of each transfer is useful.

| Warp access pattern (4-byte floats) | Sectors fetched | Bytes used / fetched
| --- | --- | --- |
| Consecutive, aligned: `a[base + tid]` | 4 | 100%
| Consecutive, offset by one: `a[base + tid + 1]` | 5 | 80%
| Stride 2: `a[2 * tid]` | 8 | 50%
| Stride 4 | 16 | 25%
| Stride 8 or more, or random gather | 32 | 12.5%
| All threads same address | 1 | broadcast, one sector

For 8-byte or 16-byte elements (double, `float2`, `float4`) a contiguous warp access covers 256 or 512 bytes and stays fully efficient. Random gathers, such as embedding lookups with arbitrary indices, cannot be coalesced across rows, but each row should still be read by consecutive threads so that the bytes *within* a row are coalesced.

## Common layout fixes

- **Contiguous axis on `threadIdx.x`.** For a row-major matrix, a warp should walk along a row. A kernel that sums each column with one thread per column is coalesced (threads read neighbouring columns of the same row in each step); a kernel that sums each row with one thread per row is not (threads are a whole row apart). The usual fix for row reductions is one warp or block per row, with threads striding across it.
- **Structure of arrays (SoA).** Storing particles as `struct {float x, y, z, w;} p[n]` and reading only `x` gives a stride of 4 floats (25% efficiency). Storing `float x[n], y[n], ...` makes each field contiguous.
- **Stage through shared memory.** A transpose must read rows and write columns. Reading a tile coalesced into shared memory and writing it out coalesced along the other axis turns one strided side into two coalesced ones (31.03).
- **Vectorised loads.** Having each thread load a `float4` (16 bytes) cuts the number of load instructions by four for the same bytes. It requires 16-byte alignment and `n` divisible by 4, so production kernels pair it with a scalar tail loop.

```cpp
// Experiment from NVIDIA's coalescing post: same bytes, different stride
__global__ void strided_copy(const float* in, float* out, int stride, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (i < n) out[i] = in[i];
}
// effective bandwidth = 2 * bytes_touched / seconds   (one read + one write)
```

## Measuring: effective bandwidth

For a memory-bound kernel, compute the bytes it must move and divide by measured time. Vector add on $n = 2^{26}$ floats reads 2 x 256 MB and writes 256 MB: 768 MB. At 0.44 ms that is 1.83 TB/s, about 90% of an A100 80 GB's peak, which is as good as it gets. If the same kernel took 2 ms you would look for uncoalesced access or a launch so small that the GPU never fills. Nsight Compute reports this directly ("Memory Throughput" and sectors per request in the memory workload analysis); a value well above 4 sectors per request for 4-byte loads is the signature of poor coalescing.

<div class="ex">
<div class="lab">Worked example</div>

**a cosine-similarity layout**

Computing pairwise cosine similarity between 4096 query vectors and 4096 keys of dimension 256, a naive kernel assigns one thread per (query, key) pair and loops over the 256 dimensions. With keys stored row-major, the 32 threads of a warp read 32 different key rows at the same dimension: 32 sectors per load. Transposing keys to dimension-major layout (`K[d][key]`) makes the same loop read 32 consecutive keys at dimension `d`: 4 sectors per load, an 8x reduction in traffic for the key side. Precomputing norms once removes a second pass entirely.

</div>

## What the cache does and does not fix

L1 and L2 can rescue some poor patterns: if neighbouring warps soon read the other bytes of the sectors you wasted, they may hit in cache. Do not rely on it. Caches hide waste only when the working set fits and the reuse comes soon, and both conditions break at production sizes. Design for coalescing first; treat cache hits as a bonus you verify in the profiler.

<div class="take">
<div class="lab">Key takeaway</div>

count sectors per warp request: make consecutive threads touch consecutive addresses on the contiguous axis, use SoA layouts and shared-memory staging to remove strides, and measure effective bandwidth against the device peak.

</div>
