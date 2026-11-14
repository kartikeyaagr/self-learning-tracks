---
id: "31.02"
course: "inference-eng-course"
module: "31"
slug: "global-memory-and-coalescing"
title: "Global memory and coalescing"
keyIdeas: ["Global memory (HBM, about 2.0 TB/s on an A100 80 GB and 3.35 TB/s on an H100 SXM, spec sheet values) is fetched in **32-byte sectors**. A warp load is efficient only if its 32 addresses fall into as few sectors as possible.", "**Coalesced**: 32 threads reading 32 consecutive 4-byte words touch 128 bytes = 4 sectors, so 100% of fetched bytes are used. Stride 2 doubles the sectors (50%); a stride of 8 or more floats means 32 sectors and 12.5% efficiency.", "Layout decides coalescing: map `threadIdx.x` to the contiguous dimension, give each row a warp rather than a thread, prefer **structure of arrays** when reading single fields, and stage unavoidable strides through shared memory.", "Alignment matters less than stride but still costs: an access offset by one element touches 5 sectors instead of 4 (80%). `cudaMalloc` returns 256-byte-aligned pointers; `float4` loads need 16-byte alignment.", "Judge memory-bound kernels by **effective bandwidth** $(\\text{bytes read} + \\text{bytes written}) / t$ against the device peak; well-written kernels reach about 80 to 90% of it, and Nsight Compute's sectors per request shows waste directly."]
question: "A row-wise L2-normalisation kernel for a 65536 x 1024 float matrix assigns one thread per row; each thread loops over its 1024 columns to compute the sum of squares, then loops again to scale. It achieves about 150 GB/s on a GPU with 2 TB/s peak bandwidth, and Nsight Compute reports close to 32 sectors per request for its global loads. Diagnose the problem, estimate how much traffic is wasted, and describe a better thread mapping and how you would confirm the fix."
answer: "With one thread per row, the 32 threads of a warp read the same column index of 32 different rows in each loop iteration. Rows are 1024 x 4 = 4096 bytes apart, so every warp load touches 32 separate 32-byte sectors and uses only 4 bytes of each: 128 useful bytes out of 1024 fetched, 12.5% at best, which matches the ~32 sectors per request in the profiler. The caches cannot absorb this: the next iteration would reuse the rest of each sector, but with many warps per SM each streaming its own 32 rows, lines are evicted before reuse at a 268 MB working set. The fix is to map a warp (or a block for very long rows) to each row. Lane j reads columns j, j+32, j+64, ..., so each warp load reads 128 consecutive bytes (4 sectors). Each lane accumulates a partial sum of squares, the warp combines the 32 partials with `__shfl_xor_sync` in five steps, every lane computes the same inverse norm, and the same lanes then scale their elements, ideally keeping them in registers (32 values per lane for 1024 columns) so the row is read from memory only once. `float4` loads (lane j reading columns 4j..4j+3, rows are 4096 bytes so alignment holds) cut instruction count by four. To confirm, compute effective bandwidth: the ideal kernel reads 268 MB and writes 268 MB, so at 80 to 90% of a 2 TB/s peak it should take about 0.3 ms, roughly ten times faster than the original, and Nsight Compute should show about 4 sectors per request for 4-byte loads (16 for float4 loads, which carry 512 bytes)."
sources: [{"title": "CUDA C++ Best Practices Guide: Coalesced Access to Global Memory", "url": "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory", "note": "Diagrams of aligned, misaligned and strided warp accesses and their transactions."}, {"title": "NVIDIA blog: How to Access Global Memory Efficiently in CUDA C/C++ Kernels", "url": "https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/", "note": "Offset and stride experiments with measured bandwidth curves."}, {"title": "NVIDIA: GPU Performance Background User's Guide", "url": "https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html", "note": "Memory versus math bound operations in DL, with A100 numbers."}]
practice: ["global-memory-coalescing-counter", "l1-normalization", "cosine-similarity-2", "matrix-vector-multiplication"]
tags: ["coalescing", "memory bandwidth"]
tools: ["CUDA", "Nsight Compute"]
---

## What you will learn

- Why most kernels outside matrix multiply are limited by memory traffic, not arithmetic, and what that means for how you write them.
- How the hardware turns the 32 addresses of one warp's load into **32-byte sectors**, and how to count those sectors by hand.
- What **coalescing** means precisely, and how stride, misalignment and element size change the number of bytes actually fetched.
- The standard layout fixes: contiguous axis on `threadIdx.x`, structure of arrays, warp-per-row mappings, shared-memory staging and vector loads.
- How to judge a memory-bound kernel by **effective bandwidth**, and which profiler counters reveal poor coalescing.

*Before you start:* you should know how a kernel computes its global index from `blockIdx`, `blockDim` and `threadIdx`, and that threads run in warps of 32 (lesson 31.01). The memory hierarchy and the roofline idea from "GPU Threads, Warps, Blocks & Memory Access" and the profiling module help but are not required.

Lesson 31.01 ended with a rule of thumb: put the contiguous axis of your data on `threadIdx.x`. This lesson explains why, how much it is worth, and what to do when the data does not cooperate. In short: global memory delivers fixed-size chunks, not single floats, and every unused byte in a chunk is wasted bandwidth.

## The idea in one picture

Take one warp (32 threads) and let each thread load one 4-byte float. Three ways of choosing the addresses:

| Code | Addresses requested (bytes) | Distinct 32-byte sectors | Useful fraction
| --- | --- | --- | --- |
| `a[tid]` | 0, 4, 8, ..., 124 | 4 | 100%
| `a[tid + 1]` | 4, 8, ..., 128 | 5 | 80%
| `a[2 * tid]` | 0, 8, 16, ..., 248 | 8 | 50%

All three ask for exactly 128 useful bytes. The memory system, however, moves data in **sectors**: aligned 32-byte blocks. A sector is fetched whole if even one of its bytes is needed. So the first pattern moves 128 bytes, the second 160, the third 256.

~~~dg-fig 31.02-sectors.svg
The same 32 loads of 4 bytes, three layouts. Filled bars are the words threads asked for; shaded boxes are the sectors the hardware has to fetch. Consecutive aligned addresses use every fetched byte; a one-word offset spills into a fifth sector; stride 2 fetches twice the bytes it needs.
~~~

**Coalescing** means a warp's addresses are packed into as few sectors as possible; a load is *fully coalesced* when bytes fetched equal bytes requested. For a memory-limited kernel, the useful fraction is roughly the fraction of peak speed you can hope for.

## Step 1: why memory decides the speed

Count the work in a typical elementwise kernel such as $y_i = \text{gelu}(x_i)$. Per element it reads 4 bytes, writes 4 bytes and does perhaps 10 to 20 floating-point operations. An A100 does about 19.5 TFLOPS in FP32 and moves about 2.0 TB/s from HBM (spec sheet values); an H100 SXM moves about 3.35 TB/s. At 2 TB/s the A100 can deliver roughly 250 billion 8-byte read-write pairs per second, which needs only a few TFLOPS of arithmetic. The arithmetic units sit mostly idle; memory sets the pace.

The same holds for normalisation, softmax, residual adds, optimizer updates and embedding lookups: their **arithmetic intensity** (operations per byte moved) is low, so they are **memory-bound**. Its speed is

$$t \approx \frac{\text{bytes actually moved}}{\text{achieved bandwidth}},$$

where *bytes actually moved* counts whole sectors, not the bytes your code asked for. A kernel that wastes 7 of every 8 bytes it fetches runs at an eighth of the speed, and no amount of arithmetic tuning recovers that.

~~~dg-layers The memory hierarchy as seen by one kernel, fastest and smallest at the top (spec sheet values).
Registers | per thread, private; up to 255 x 32-bit registers per thread
Shared memory / L1 | per SM, up to ~228 KB on H100; programmer-managed or cache
L2 cache | shared by all SMs: 40 MB (A100), 50 MB (H100)
*Global memory (HBM) | 40-80 GB; ~2-3.4 TB/s; all loads and stores land here eventually
~~~

**Global memory** is the large off-chip memory that `cudaMalloc` allocates from; every tensor lives there.

## Step 2: how a warp load becomes sectors

When a warp executes a load, the hardware collects the 32 addresses and works out which sectors they cover. Each sector is 32 bytes, aligned so that its starting address is a multiple of 32; four consecutive sectors make one 128-byte cache line.

For a thread that reads $s$ bytes starting at byte address $a$, the sectors it touches are numbered

$$\left\lfloor \frac{a}{32} \right\rfloor \text{ through } \left\lfloor \frac{a + s - 1}{32} \right\rfloor.$$

Here $a$ is the byte address, $s$ is the element size in bytes (4 for `float`), and $\lfloor\cdot\rfloor$ rounds down. The warp fetches the union of these sets over its 32 threads. Call the size of that union $N_{\text{sec}}$. Then

$$\text{useful fraction} = \frac{\text{distinct bytes requested}}{32 \cdot N_{\text{sec}}}.$$

Check with `a[tid]` on floats: thread $t$ reads bytes $4t$ to $4t+3$, all inside sector $\lfloor 4t/32 \rfloor = \lfloor t/8 \rfloor$. Threads 0-7 share sector 0, threads 8-15 sector 1, and so on: $N_{\text{sec}} = 4$, and $128 / (32 \times 4) = 100\%$.

The full catalogue for 4-byte elements:

| Warp access pattern (4-byte floats) | Sectors fetched | Bytes used / fetched
| --- | --- | --- |
| Consecutive, aligned: `a[base + tid]` | 4 | 100%
| Consecutive, offset by one: `a[base + tid + 1]` | 5 | 80%
| Stride 2: `a[2 * tid]` | 8 | 50%
| Stride 4 | 16 | 25%
| Stride 8 or more, or random gather | 32 | 12.5%
| All threads same address | 1 | broadcast, one sector

Thread *order* does not matter, only the set of sectors: a permutation within one 128-byte block is still 4 sectors. Wider contiguous elements are fine too: 32 `double`s cover 8 sectors and 32 `float4`s cover 16, both 100% useful.

## Step 3: stride, offset and the bandwidth ceiling

Stride is the most common way to lose bandwidth. With stride $k$ (thread $t$ reads element $k \cdot t$), consecutive threads are $4k$ bytes apart. While $4k < 32$, several threads still share a sector; once $k \ge 8$ every thread lands in its own sector and the warp fetches 32 sectors of which it uses 4 bytes each.

~~~dg-fig 31.02-stride.svg
Left: for 4-byte loads, sectors per warp request grow linearly with stride and saturate at 32 from stride 8 on. Right: the useful fraction falls like 1/stride and bottoms out at the element size divided by 32 bytes: 12.5% for floats, 25% for doubles, 50% for float4. Computed by counting sectors, not timed.
~~~

If DRAM moved every sector at full speed, the useful bandwidth you could get is simply peak bandwidth times the useful fraction. That gives a ceiling for a strided copy:

~~~dg-fig 31.02-ceiling.svg
A model, not a measurement: spec-sheet peak bandwidth multiplied by the useful fraction from the previous figure. At stride 8 the A100 ceiling drops from about 2.0 to 0.25 TB/s. Real measurements follow this shape but sit below it.
~~~

NVIDIA's coalescing experiments on real hardware (see the resources) show the same shape; measured numbers sit below the line because no kernel reaches 100% of peak and scattered sectors also lower DRAM efficiency.

**Offset** is the gentler cousin. If the warp's 128 bytes start one float past a 32-byte boundary, they straddle 5 sectors instead of 4: 80% useful. It is worth avoiding in hot loops, but it is a 20% tax, not an 8x one. Pointers from `cudaMalloc` are aligned to at least 256 bytes, so offsets usually come from your own indexing: slicing a tensor at an odd column, or a row length that is not a multiple of 8 floats.

<div class="ex">
<div class="lab">Worked example</div>

**count the sectors for three kernels by hand**

Kernel A reads `a[tid]` from a `float` array that starts at byte 0. Thread 31 reads bytes 124 to 127. Sector numbers run from $\lfloor 0/32 \rfloor = 0$ to $\lfloor 127/32 \rfloor = 3$, so 4 sectors and 128 bytes fetched for 128 requested: 100%.

Kernel B reads `p[tid].x` from an array of `struct {float x, y, z, w;}` (16 bytes per struct). Thread $t$ reads bytes $16t$ to $16t+3$. Threads 0 and 1 (bytes 0 and 16) share sector 0; threads 2 and 3 share sector 1; so the 32 threads need $32/2 = 16$ sectors, which is $16 \times 32 = 512$ bytes fetched for 128 useful: 25%.

Kernel C has one thread per row of a $65536 \times 1024$ float matrix and reads `x[tid * 1024 + j]`. Rows are $1024 \times 4 = 4096$ bytes apart, far more than 32, so every thread is in its own sector: 32 sectors, $1024$ bytes fetched for 128 useful: 12.5%.

Effective bandwidth check: if kernel C streams the whole 268 MB matrix, it moves up to $8 \times 268 = 2147$ MB through DRAM unless caches catch the rest of each sector. At a 2.0 TB/s peak that is at least 1.07 ms of pure transfer, versus 0.13 ms for a coalesced kernel.

</div>

## Step 4: fixing the layout

~~~dg-fig 31.02-layouts.svg
Sectors per warp load before and after four common fixes, from the sector counter. Each fix keeps the bytes the kernel needs the same and cuts what it fetches, up to 8x.
~~~

- **Contiguous axis on `threadIdx.x`.** For a row-major matrix, consecutive threads should walk along a row. A column-sum kernel with one thread per column is coalesced: at each step the warp reads 32 neighbouring columns of one row. A row-sum kernel with one thread per row is not: the threads are a whole row apart (kernel C above). The usual fix for row reductions is one warp (or one block) per row, with lanes striding across the row in steps of 32.
- **Structure of arrays (SoA).** Storing particles as an **array of structures (AoS)** `struct {float x, y, z, w;} p[n]` and reading only `x` is stride 4 (25%). Storing `float x[n], y[n], z[n], w[n]` makes each field contiguous. If a kernel genuinely reads all four fields of each particle, AoS with a `float4` load per thread is also fully coalesced; the problem is reading a subset.
- **Stage through shared memory.** A matrix transpose must read rows and write columns, so one side is strided whatever you do. Read a tile coalesced into shared memory, then write it out coalesced along the other axis. Both global sides become coalesced; the stride moves to fast on-chip memory (lesson 31.03).
- **Vectorised loads.** Having each thread load a `float4` moves the same bytes with a quarter of the load instructions. It needs 16-byte-aligned addresses and a length divisible by 4, so production kernels pair it with a scalar loop for the tail.

<div class="ex">
<div class="lab">Worked example</div>

**a cosine-similarity layout**

Compute cosine similarity between 4096 query vectors and 4096 key vectors of dimension 256. A naive kernel gives each thread one (query, key) pair; the 32 threads of a warp share one query and take 32 consecutive keys, then loop over the 256 dimensions.

With keys stored row-major, `K[key][d]`, the warp reads dimension $d$ of 32 different keys. Keys are $256 \times 4 = 1024$ bytes apart, so each load touches 32 sectors (12.5%).

Transpose the keys once to dimension-major, `K[d][key]`. Now at dimension $d$ the warp reads 32 consecutive keys: 4 sectors, an 8x cut in key-side traffic. The query element is the same address for all 32 threads, a one-sector broadcast.

The transpose costs one pass over $4096 \times 256 \times 4 = 4.2$ MB, tiny next to the repeated reads it saves. Precomputing all key and query norms once removes a second pass over the data entirely.

</div>

## Coalescing in code

The classic experiment copies the same number of elements with different strides. Time it with `cudaEvent`s, as in lesson 31.01:

```cpp
// Same instruction, different addresses: only the stride changes.
__global__ void strided_copy(const float* in, float* out, int stride, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (i < n) out[i] = in[i];
}
// effective bandwidth = 2 * 4 * (elements copied) / seconds   (one read + one write)
```

Here is the row-normalisation fix from kernel C, with one warp per row. Lane $j$ reads columns $j, j+32, j+64, \dots$, so every warp load covers 128 consecutive bytes:

```cpp
__global__ void l2norm_warp_per_row(const float* x, float* y, int rows, int cols) {
    int row  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;   // one warp per row
    int lane = threadIdx.x % 32;
    if (row >= rows) return;                                    // whole warp exits together
    const float* r = x + (size_t)row * cols;
    float ss = 0.f;
    for (int j = lane; j < cols; j += 32) ss += r[j] * r[j];    // coalesced
    for (int o = 16; o > 0; o >>= 1)                            // warp sum, no shared memory
        ss += __shfl_xor_sync(0xffffffff, ss, o);
    float inv = rsqrtf(ss + 1e-12f);
    for (int j = lane; j < cols; j += 32) y[(size_t)row * cols + j] = r[j] * inv;
}
```

You can check any indexing scheme before writing CUDA by counting sectors in Python. This is the rule from step 2, nothing more:

```python
import numpy as np

def warp_sectors(byte_addr, size=4, sector=32):
    # distinct 32-byte sectors touched when 32 threads each load `size` bytes
    first, last = byte_addr // sector, (byte_addr + size - 1) // sector
    return len(set(np.concatenate([np.arange(f, l + 1) for f, l in zip(first, last)])))

tid = np.arange(32)
patterns = {"a[tid]": tid * 4, "a[tid + 1]": (tid + 1) * 4, "a[2 * tid]": 2 * tid * 4,
            "p[tid].x (AoS)": tid * 16, "row[tid][j]": tid * 1024 * 4,
            "a[0] (all same)": np.zeros(32, int)}
for name, addr in patterns.items():
    n = warp_sectors(addr)
    print(f"{name:16s} {n:2d} sectors  {len(set(addr)) * 4 / (n * 32):6.1%} useful")
# a[tid]            4 sectors  100.0% useful
# a[tid + 1]        5 sectors   80.0% useful
# a[2 * tid]        8 sectors   50.0% useful
# p[tid].x (AoS)   16 sectors   25.0% useful
# row[tid][j]      32 sectors   12.5% useful
# a[0] (all same)   1 sectors   12.5% useful   (one broadcast sector, nothing wasted in traffic)
print(warp_sectors(tid * 16, size=16))   # float4 a4[tid]: 16 sectors, 100% useful
```

## Checking the result: effective bandwidth

For a memory-bound kernel the right score is **effective bandwidth**: the bytes the algorithm must move divided by the measured time,

$$B_{\text{eff}} = \frac{\text{bytes read} + \text{bytes written}}{t}.$$

Count the bytes the algorithm needs, not sectors fetched, so waste shows up as a low score.

Vector add on $n = 2^{26}$ floats reads two arrays and writes one: $3 \times 2^{26} \times 4 = 805$ million bytes (768 MiB). Suppose it runs in 0.44 ms; then $B_{\text{eff}} = 805 \times 10^6 / 0.44 \times 10^{-3} = 1.83$ TB/s, about 90% of an A100 80 GB's spec peak, which is about as good as kernels get. If it took 2 ms ($0.40$ TB/s, 20%), look for uncoalesced access, or a launch too small to fill the GPU.

Nsight Compute reports this as "Memory Throughput", and its memory workload analysis shows **sectors per request** for global loads: 4 is ideal for 4-byte loads, values near 32 signal strided or gathered access. The raw counters are `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` and `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`; their ratio is sectors per request:

```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./my_app
```

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Row-wise kernel (norm, softmax) far below peak | one thread per row: stride = row length | one warp or block per row, lanes stride by 32
| Reading one field of a struct is slow | AoS layout, stride = struct size | SoA, or read the whole struct with a vector load
| About 20% slower after slicing a tensor | base pointer not 32-byte aligned | pad rows to a multiple of 8 floats; align views
| Transpose or permute is slow on one side | reads coalesced, writes strided (or vice versa) | tile through shared memory (31.03)
| `float4` kernel crashes or gives misaligned-address errors | pointer or length not a multiple of 16 bytes | check alignment, add a scalar tail loop
| Fast on small inputs, slow on large | caches hid the waste until the working set outgrew L2 | fix coalescing; do not trust small benchmarks
| Embedding lookup at low bandwidth | random rows cannot be coalesced across rows | have consecutive threads read within each row; keep rows wide and aligned

## Going deeper

**What the cache does and does not fix.** L1 and L2 can rescue some poor patterns: if neighbouring warps soon read the rest of the sectors you half-used, they hit in cache and the DRAM traffic is not repeated. In kernel C, iteration $j+1$ reads the next 4 bytes of the same sectors, so on a small matrix L1 may hold them. Do not rely on it: at production sizes lines are evicted before the reuse arrives. Design for coalescing; treat cache hits as a bonus that you verify in the profiler.

**Loads, stores and gathers.** Stores follow the same sector rule: a warp writing scattered 4-byte values dirties many sectors for little data. Gathers with arbitrary indices, such as `emb[ids[tid]]`, are uncoalesced across threads by nature. The design response is to change *which* threads read *what*: give each row to a group of consecutive threads so the bytes within a row are coalesced, and let the row index vary across groups.

**Cache lines versus sectors.** L1 tracks 128-byte lines but fills and evicts them in 32-byte sectors, and the L2 to DRAM path also works in sectors. That is why the useful unit for reasoning is the 32-byte sector.

**Element size sets the floor.** A fully scattered warp still uses $s$ bytes of every 32-byte sector, so the worst-case useful fraction is $s/32$: 12.5% for `float`, 6.25% for `half`/`bf16`, 25% for `double`. Low-precision formats make coalescing *more* important, not less, because a 2-byte gather wastes 30 of every 32 bytes.

**Alignment rules.** `cudaMalloc` returns pointers aligned to at least 256 bytes. Vector types need natural alignment: `float2` 8 bytes, `float4` 16 bytes. A `reinterpret_cast<const float4*>` on a pointer that is only 4-byte aligned is undefined behaviour and typically faults with a misaligned-address error.

## Key terms

| Term | Meaning
| --- | --- |
| Global memory | The large off-chip GPU memory (HBM) that holds every tensor; slowest level, highest capacity
| Sector | An aligned 32-byte block, the unit in which global memory data is fetched
| Cache line | Four consecutive sectors, 128 bytes, the unit L1 tracks
| Coalescing | Packing a warp's 32 addresses into as few sectors as possible
| Stride | Distance in elements between the addresses of consecutive threads
| AoS / SoA | Array of structures (fields interleaved) versus structure of arrays (one contiguous array per field)
| Memory-bound | A kernel whose time is set by bytes moved rather than arithmetic
| Effective bandwidth | Bytes the algorithm must read and write divided by measured time
| Sectors per request | Profiler ratio of sectors fetched to warp load instructions; 4 is ideal for 4-byte loads

<div class="take">
<div class="lab">Key takeaway</div>

global memory moves 32-byte sectors, so count sectors per warp request: put consecutive threads on consecutive addresses of the contiguous axis, use SoA layouts, warp-per-row mappings and shared-memory staging to remove strides, and judge the result by effective bandwidth against the device peak.

</div>
