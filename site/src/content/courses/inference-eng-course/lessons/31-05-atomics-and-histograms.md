---
id: "31.05"
course: "inference-eng-course"
module: "31"
slug: "atomics-and-histograms"
title: "Atomics and histograms"
keyIdeas: ["An **atomic** (`atomicAdd`, `atomicMax`, `atomicCAS`, ...) does read-modify-write on one memory word as one indivisible step and returns the old value; without it, two threads can read the same count and one increment is lost.", "Atomics to **different** addresses run in parallel, atomics to the **same** address queue. A histogram's speed therefore depends on its data: uniform noise spreads the work, a constant frame puts every update in one queue.", "**Privatisation** gives each block (or warp) its own bins in shared memory and merges at the end, cutting global atomics from $n$ to at most blocks $\\times$ bins (for example 262,144 instead of $10^8$).", "**Warp-aggregated atomics** use `__ballot_sync` or `__match_any_sync` so one leader lane issues a single atomic per distinct address per warp; they help when values repeat within a warp, as in compaction with one shared counter.", "Anything not built in is a retry loop around `atomicCAS`. Float atomics are order-dependent, so PyTorch flags CUDA `scatter_add_` and `index_add_` as non-deterministic; integer counts are always reproducible."]
question: "You need per-expert token counts for an MoE layer with 8 experts and 1 million tokens per step, computed on the GPU. A first version does one global `atomicAdd` per token on an array of 8 integers and is surprisingly slow, and it gets slower on steps where the router sends most tokens to one expert. Explain why, design a faster kernel and estimate how many global atomics it issues with 1,024 blocks. Is the result deterministic? Would your answer change if you also accumulated each expert's float routing weights?"
answer: "With only 8 counters, a million atomics are concentrated on 8 addresses. Global atomics are applied in L2, where updates to one address are serialised, so at most about 8 updates make progress at a time however many threads run; when routing is skewed, the busiest expert's counter carries most of the million updates and its queue sets the run time. Faster design: privatise. Each block keeps 8 counters in shared memory (zero them, `__syncthreads()`), and each thread first counts its own grid-stride tokens in 8 registers, then adds each non-zero register count to the shared counters once; for still fewer shared atomics, use `__match_any_sync` so one leader per expert per warp adds the group size. After a second barrier, threads 0-7 each add one shared counter to the global array. With 1,024 blocks that is at most $1024 \\times 8 = 8{,}192$ global atomics instead of $10^6$, and each block only sees about 977 tokens, so shared queues are short. Launch only enough blocks to fill the GPU, since merge cost grows with the block count. The integer counts are deterministic: integer addition is associative, so the order in which atomics arrive does not change the result. Float routing weights are different: summing them with float atomics makes the low bits depend on arrival order, so totals can differ between runs. For reproducibility, write per-block partial sums to an array and reduce them in a fixed order (or use a deterministic sort-then-reduce), accepting a small cost."
sources: [{"title": "NVIDIA blog: GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell", "url": "https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/", "note": "Privatised histograms and how contention depends on the input image."}, {"title": "NVIDIA blog: CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics", "url": "https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/", "note": "The leader-lane pattern for stream compaction with a shared counter."}, {"title": "CUDA C++ Programming Guide: Atomic Functions", "url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions", "note": "Semantics, supported types and the atomicCAS recipe for custom operations."}, {"title": "PyTorch: Reproducibility", "url": "https://pytorch.org/docs/stable/notes/randomness.html", "note": "Which CUDA operations are non-deterministic and what use_deterministic_algorithms changes."}]
practice: ["sum-of-array", "argmin", "max-of-array", "min-of-array"]
tags: ["atomics", "privatisation"]
tools: ["CUDA", "PyTorch"]
---

## What you will learn

- Why a plain `bins[b] += 1` from many threads loses updates, and what an **atomic** operation guarantees instead.
- Why atomics to the same address queue up (**contention**), so the speed of a histogram depends on the data, not only on its size.
- How **privatisation** (per-block or per-warp bins in shared memory) turns millions of global atomics into a few hundred thousand.
- How **warp-aggregated atomics** use ballot and match intrinsics so one leader lane updates for a whole group.
- How to build any other read-modify-write (float max, argmin) from `atomicCAS`, and why float atomics make results non-deterministic.

*Before you start:* you should be comfortable with the thread hierarchy and grid-stride loops (31.01), coalesced global loads (31.02), shared memory and `__syncthreads` (31.03), and the tree reduction from 31.04. Warp shuffles from 31.04 reappear here.

Lesson 31.04 combined an array into one number along a tree fixed in advance. Many kernels instead do **scatter** updates, where the data decides where each write goes: a histogram learns a pixel's bin only by reading it, an embedding backward pass learns which rows get gradient from the token ids, and an MoE router learns each expert's token count only after routing. Two threads may pick the same place at the same moment. Atomics are the hardware tool for that.

## The idea in one picture

Take a toy warp of 8 threads (lanes), each holding one 2-bit pixel, and count how many pixels have each value:

| Lane | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pixel value (bin) | 3 | 3 | 0 | 3 | 1 | 3 | 3 | 0

The right answer is bins = (2, 1, 0, 5). Each thread runs `bins[x[i]] += 1`. That line looks like one step, but the hardware executes three: **read** the bin from memory into a register, **add** one, **write** it back. If lanes 0 and 1 both read bin 3 while it holds 7, both compute 8 and both write 8. One increment vanishes. This is a **lost update**, the classic race condition.

~~~dg-fig 31.05-race.svg
Left: two threads both read 7 and both write 8, so one count is lost. Right: a toy simulation (read and write 4 slots apart, random start times) of many threads incrementing one bin: with 128 threads the plain increment ends near 13 instead of 128, while atomics count everything.
~~~

An **atomic operation** performs the read, modify and write on one memory word as a single indivisible step; no other thread can touch the word in between. With `atomicAdd(&bins[x[i]], 1)` the 8 lanes always produce (2, 1, 0, 5).

Atomics are correct but not free. Five lanes target bin 3, so those five updates must be applied one after another. Updates to *different* addresses run in parallel; updates to the *same* address form a queue. That queue is **contention**, and managing it is the rest of this lesson.

## Step 1: what an atomic guarantees

The basic kernel is a grid-stride loop with one atomic per element:

```cpp
__global__ void hist_naive(const unsigned char* x, unsigned int* bins, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&bins[x[i]], 1u);
}
```

Every CUDA atomic acts on **one word** (32 or 64 bits, or a 16-bit pair), never on two separate variables. It **returns the old value**, which is how you claim a unique slot: `int slot = atomicAdd(&count, 1);` gives each caller a different `slot`. And it promises nothing about the order in which threads arrive.

The built-in family covers add, sub, min, max, wrap-around increment and decrement, exchange, bitwise and/or/xor, and compare-and-swap (`atomicCAS`). Supported types include 32- and 64-bit integers and `float`; `atomicAdd` on `double` needs compute capability 6.0 or newer, as does `__half2`. Anything else is built from `atomicCAS` (step 5).

## Step 2: why the same address is slow

Global-memory atomics are carried out in the **L2 cache**, shared by all streaming multiprocessors (SMs). Its many slices process updates to different addresses at the same time, but updates to one address must each see the previous result. So a naive histogram takes roughly the larger of two terms:

$$t \approx \max\left(\frac{n}{P},\; \max_b c_b\right) \times t_{\text{atomic}},$$

where $n$ is the number of updates, $P$ is how many different addresses the hardware can update concurrently, $c_b$ is the number of updates that land in bin $b$, and $t_{\text{atomic}}$ is the time for one update. The second term is the **longest queue**: the busiest bin sets a floor no number of threads can beat, and it depends entirely on the data. For 1,048,576 pixels in 256 bins:

| Input | Busiest bin | Longest naive queue ($\max_b c_b$)
| --- | --- | --- |
| Uniform noise | 0.4% | 4,252
| Photo-like (bell-shaped around 110) | 1.0% | 10,559
| Mostly black (90% zeros) | 90.1% | 944,392
| Constant frame | 100% | 1,048,576

~~~dg-fig 31.05-contention.svg
Top: four synthetic 1-million-pixel inputs and their histograms. Bottom: the longest line of updates waiting on one address (log scale), counted for three kernels. Naive global atomics on a constant frame queue the entire input on one address; per-block shared bins cut the worst queue to 1,024, and per-warp sub-histograms to 128. Counts from a CPU model, not GPU timings.
~~~

Same kernel, same $n$, a 250-fold difference in the longest queue. NVIDIA's histogram write-up reports the real-hardware version: naive global atomics slow down sharply on images dominated by one value. Benchmark on the nastiest realistic input, not only on random bytes.

## Step 3: privatisation

The fix for a shared bottleneck is to stop sharing it. **Privatisation** gives each block its own copy of the bins in shared memory. Threads of a block accumulate there with shared-memory atomics, then the block adds its partial histogram into the global one.

~~~dg-flow Privatised histogram: most updates hit fast per-block shared memory; global memory sees one atomic per non-empty bin per block.
Input pixels -(shared atomicAdd)-> *Per-block bins in shared memory -(__syncthreads)-> Merge -(one global atomic per bin)-> Global bins
~~~

```cpp
__global__ void hist_private(const unsigned char* x, unsigned int* bins, int n) {
    __shared__ unsigned int local[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;   // 1. clear
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&local[x[i]], 1u);                                     // 2. on-chip
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x)
        if (local[b]) atomicAdd(&bins[b], local[b]);                     // 3. merge
}
```

Shared memory starts with garbage, so phase 1 zeroes it and a barrier stops anyone counting into an uncleared bin. In phase 2, shared-memory atomics run inside the SM (native hardware operations since the Maxwell generation), far cheaper than a trip to L2, and only the threads of one block can collide. After the second barrier the counts are final, and thread $b$ merges bin $b$ with one global atomic, skipping empty bins.

How many global atomics remain? At most one per bin per block:

$$A_{\text{global}} \le B \times K,$$

where $B$ is the number of blocks and $K$ the number of bins. For $B = 1024$ and $K = 256$ that is at most $262{,}144$, against $10^8$ for the naive kernel on $10^8$ pixels: about 380 times fewer. In the CPU model the uniform input needed 257,357 merge atomics (a few bins of a few blocks were empty) and the constant frame only 1,024.

~~~dg-fig 31.05-merge.svg
The merge costs blocks times bins global atomics, a straight line on log axes. Even at 16,384 blocks the 256-bin histogram stays well below the naive 10^8; for 8 MoE experts and 1,024 blocks the merge is 8,192 atomics instead of a million.
~~~

Hence a tuning rule: launch only enough blocks to fill the GPU (a small multiple of the SM count) and let the grid-stride loop do the rest; extra blocks add merge work without useful parallelism.

When contention inside a block stays high (mostly black: a queue of 950 on bin 0), give each warp its own sub-histogram (`local[warp][256]`) so only 32 lanes can collide, and sum the warp copies before the merge. In the model this cuts the worst shared queue on the constant frame from 1,024 to 128. The price is shared memory: 8 warps × 256 bins × 4 bytes = 8 KB per block, which can lower occupancy (31.07).

## Step 4: warp-aggregated atomics

When many lanes of a warp want the *same* address, they can agree among themselves first using **warp vote** intrinsics. `__ballot_sync(mask, pred)` returns a 32-bit integer whose bit $i$ is set when lane $i$'s predicate is true; `__match_any_sync(mask, value)` returns, for each lane, the bitmask of lanes holding the same value (compute capability 7.0+); `__popc` counts set bits and `__ffs` finds the lowest one (1-based). The classic use is **stream compaction**: write the elements that pass a filter densely to an output buffer through one shared counter.

```cpp
__device__ int warp_append(int* counter, bool keep) {      // returns my output slot or -1
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, keep);           // who wants a slot?
    if (mask == 0) return -1;
    int lane = threadIdx.x & 31;
    int leader = __ffs(mask) - 1;                          // lowest keeping lane
    int base = 0;
    if (lane == leader) base = atomicAdd(counter, __popc(mask));   // ONE atomic per warp
    base = __shfl_sync(active, base, leader);              // broadcast the old value
    return keep ? base + __popc(mask & ((1u << lane) - 1)) : -1;  // my rank among keepers
}
```

The leader reserves `__popc(mask)` slots in one atomic, the shuffle broadcasts the starting slot, and each keeping lane offsets by the number of keeping lanes below it. For a histogram, group lanes by value with `__match_any_sync` and let each group's lowest lane add the group size: one atomic per **distinct** bin per warp.

~~~dg-fig 31.05-warpagg.svg
Left: distinct bins among 32 consecutive pixels, which equals the atomics left after match_any aggregation: about 30 for uniform noise (little gain), about 4 for mostly black, 1 for a constant frame. Right: in compaction, the ballot version needs at most one atomic per warp instead of up to 32.
~~~

The left panel is the honest caveat: aggregation pays off when values repeat within a warp and costs extra instructions when they do not. NVIDIA notes that recent compilers already aggregate the simplest case automatically (all active lanes adding to one uniform address), but grouping by value and compaction offsets you still write yourself.

<div class="ex">
<div class="lab">Worked example</div>

**count atomics by hand for one warp**

Use the 8-lane warp from the opening table: pixels $(3, 3, 0, 3, 1, 3, 3, 0)$, correct answer bins $= (2, 1, 0, 5)$.

Naive: every lane issues `atomicAdd`, so 8 atomics. Bin 3 receives 5 of them, so the longest queue is 5 updates deep; bin 0 has a queue of 2.

Aggregated with match_any: lanes $\{0,1,3,5,6\}$ share value 3, lanes $\{2,7\}$ share value 0, lane $\{4\}$ is alone with value 1. Three groups, so 3 atomics: lane 0 adds 5 to bin 3, lane 2 adds 2 to bin 0, lane 4 adds 1 to bin 1. Every queue is now 1 deep.

Compaction of the non-zero pixels: the predicate is true on lanes 0, 1, 3, 4, 5, 6, so the ballot mask is binary $01111011 = 123$ and $\text{popc} = 6$. The leader (lane 0) does `atomicAdd(counter, 6)`; say it returns 40. Lane 3 has two keeping lanes below it (0 and 1), so it writes to slot $40 + 2 = 42$; lane 6 has five below it, so slot 45. Slots 40 to 45 are filled once each.

Scale it up: an MoE step routes 1,000,000 tokens to 8 experts with 1,024 blocks. Naive: $10^6$ global atomics on 8 addresses (about 125,000 each if balanced). Privatised: each block sees about $10^6 / 1024 \approx 977$ tokens (about 122 per expert), then $1024 \times 8 = 8{,}192$ global atomics in total.

</div>

## Step 5: any operation from compare-and-swap

`atomicCAS(p, expected, desired)` writes `desired` only if `*p` still equals `expected`, and always returns what it found. Any single-word update becomes a retry loop: read, compute, try to swap; if the returned value is not what you expected, someone got there first, so retry with the fresh value.

```cpp
__device__ float atomicMaxFloat(float* addr, float v) {
    int* a = reinterpret_cast<int*>(addr);
    int old = *a, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= v) break;           // nothing to do
        old = atomicCAS(a, assumed, __float_as_int(v));     // try to install v
    } while (old != assumed);                               // someone beat us: retry
    return __int_as_float(old);
}
```

**Packing** lets one atomic update two values: for an argmin, put an order-preserving encoding of the float in the high 32 bits of an `unsigned long long` and the index in the low 32 bits, then one 64-bit `atomicMin` updates both together. (Flip all bits of negative floats and only the sign bit of positive ones, and integer order matches float order.) Under contention every failed CAS costs another round trip, so privatise first and CAS on merged values.

## Atomics in code (on the CPU)

You cannot run CUDA here, but NumPy and PyTorch expose exactly the same distinction between a buffered scatter (lost updates) and an accumulating one (atomic semantics):

```python
import numpy as np, torch
pix = np.array([3, 3, 0, 3, 1, 3, 3, 0])            # the toy warp

bins = np.zeros(4, dtype=np.int64)
bins[pix] += 1                   # fancy-index +=: duplicates collapse
print(bins)                      # [1 1 0 1]   <- lost updates, like plain +=

bins = np.zeros(4, dtype=np.int64)
np.add.at(bins, pix, 1)          # unbuffered: every update applied
print(bins)                      # [2 1 0 5]
print(np.bincount(pix, minlength=4))                 # [2 1 0 5]

t = torch.tensor(pix)
print(torch.bincount(t, minlength=4))                # tensor([2, 1, 0, 5])
print(torch.zeros(4, dtype=torch.long).index_add_(0, t, torch.ones(8, dtype=torch.long)))
                                                     # tensor([2, 1, 0, 5])
```

`bins[pix] += 1` is the Python twin of the opening race: every duplicate index reads the old value and writes old + 1, so bin 3 ends at 1 instead of 5. On a GPU, `torch.bincount`, `index_add_` and `scatter_add_` use atomics.

## Floating-point atomics and determinism

Integer atomics give the same final value in any order, because integer addition is associative. Floating-point addition is not: $(a + b) + c$ and $a + (b + c)$ can round differently. With float atomics the order in which updates arrive changes from run to run, so the last bits of the result change too.

~~~dg-fig 31.05-float.svg
The same 20,000 float32 values summed sequentially in 400 random orders, as float atomics would. 250 different totals appear, spread over about 0.013 around the exact value; a fixed-order sum gives one answer every time. Shuffling 20,000 integers 50 times gave a single total.
~~~

The spread is tiny next to the total of about $-148.6$, but it breaks bitwise reproducibility and can make two training runs drift apart. That is why PyTorch's reproducibility notes list CUDA `index_add_` and `scatter_add_` as non-deterministic. With `torch.use_deterministic_algorithms(True)` they either switch to a slower deterministic implementation (typically sort-then-reduce) or raise an error. For exact reproducibility, replace float atomics with a fixed-order reduction.

## Checking an atomic kernel

- **Correctness:** integer counts must match `torch.bincount` exactly, and the bins must sum to $n$; a smaller total means lost updates.
- **Speed:** time random, skewed and constant inputs. A much slower constant input means contention is the bottleneck.
- **Profile:** Nsight Compute's memory workload analysis shows how much L2 traffic comes from atomics; a high share in a kernel that should be bandwidth-bound points at contention.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Counts too low, vary run to run | Plain `+=` on shared bins, or merge before the barrier | Use atomics; `__syncthreads()` between accumulate and merge
| Garbage counts in a few bins | Shared bins never zeroed | Clear, then barrier, before accumulating
| Fast on random data, very slow on real images | Same-address contention in L2 | Privatise per block, then per warp
| Privatised kernel slower than expected | Too many blocks: merge cost $B \times K$ grows | Launch about enough blocks to fill the GPU, grid-stride the rest
| Occupancy drops after per-warp bins | Shared memory per block grew | Fewer copies, or 16-bit sub-counts flushed before they overflow
| Loss curves differ between identical runs | Float atomics (`scatter_add_`, `index_add_`) | Deterministic algorithms, or sort-then-reduce

## Going deeper

**Ordering and scope.** An atomic makes one word's update indivisible; it does not order your other memory accesses. CUDA's classic atomics are *relaxed*: if a thread writes data and then atomically sets a flag, another thread that sees the flag is not guaranteed to see the data unless you add a fence (`__threadfence()`) or use the C++ `cuda::atomic_ref` from libcu++ with acquire/release ordering. Atomics also have a **scope**: `atomicAdd_block` is atomic only among threads of the same block (cheaper, enough for shared bins), plain `atomicAdd` is device-wide, and `atomicAdd_system` extends to the host and peer GPUs on unified memory.

**RED versus ATOM.** When the return value is unused, the compiler can emit a fire-and-forget reduction instruction (RED) instead of a returning atomic (ATOM), so the thread does not wait for the old value.

**How long is the longest queue?** For $n$ uniform updates over $K$ bins the busiest bin is a little above $n/K$ (mean 4,096, model maximum 4,252 at $n = 2^{20}$, $K = 256$); for skewed data it approaches $p_{\max} n$, with $p_{\max}$ the share of the most common value. Privatisation divides that by $B$ but adds a merge queue of up to $B$ per hot global bin.

**The atomic-free alternative.** Radix-sort the keys, then count each run of equal keys with a scan (31.04). It is deterministic and contention-free but moves the data several times, so for integer histograms privatised atomics usually win.

## Key terms

| Term | Meaning
| --- | --- |
| Atomic operation | An indivisible read-modify-write on one memory word that returns the old value
| Lost update | Two threads read the same old value and both write old + 1, so one increment disappears
| Contention | Several atomics on the same address waiting in a queue to be applied one at a time
| Privatisation | Per-block or per-warp copies of the bins in shared memory, merged into global memory at the end
| Merge | The final step where each block adds its private bins to the global result, at most blocks times bins atomics
| Warp-aggregated atomic | One leader lane issues a single atomic on behalf of all lanes that target the same address
| Ballot / match_any | Warp vote intrinsics that return bitmasks of lanes with a true predicate or an equal value
| atomicCAS | Compare-and-swap; the building block for any custom atomic, used in a retry loop

<div class="take">
<div class="lab">Key takeaway</div>

atomics make scattered, data-dependent updates correct, but updates to one address are applied one at a time, so speed depends on how many threads collide; privatise into shared memory, aggregate within a warp, keep the merge small, and avoid float atomics wherever you need bitwise reproducible results.

</div>
