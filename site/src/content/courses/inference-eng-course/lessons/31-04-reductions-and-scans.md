---
id: "31.04"
course: "inference-eng-course"
module: "31"
slug: "reductions-and-scans"
title: "Reductions and scans"
keyIdeas: ["A **reduction** turns many values into one (sum, max, mean, softmax denominator). It does one operation per element loaded, so it is memory-bound: the best possible time is the input size divided by memory bandwidth.", "A **tree reduction** combines pairs in $\\log_2 n$ steps. Use **sequential addressing** (thread `t` adds `a[t + s]`, with `s` halving) so active threads stay contiguous: no warp divergence and no bank conflicts.", "Most of the speed comes from giving each thread many elements in a coalesced grid-stride loop, then finishing with **warp shuffles** (`__shfl_down_sync`), which combine 32 registers in five steps without touching memory.", "Blocks cannot wait for each other, so finish with a **second pass** over per-block partials (deterministic) or one `atomicAdd` per block (one launch, but floating-point results change in the last bits between runs). Accumulate in FP32.", "A **scan** (prefix sum) gives every position the total so far; it computes ragged-batch offsets, compaction slots and top-p cut-offs. Hillis-Steele takes $\\log_2 n$ steps but $O(n \\log n)$ work; Blelloch is work-efficient at about $2n$; libraries such as CUB use single-pass decoupled look-back."]
question: "Your loss-averaging kernel sums 50 million FP32 values with one `atomicAdd` per thread into a single global float. It is slow and a unit test comparing the result bitwise to a reference fails intermittently. Explain both problems and the design you would use instead."
answer: "Speed: all 50 million threads perform an atomic on the same address. Atomics to one address are serialised in the L2 cache, so throughput collapses to that of a single memory location instead of the full memory bandwidth; the kernel is limited by contention, not by reading the input. Bitwise failures: floating-point addition is not associative, and the order in which the atomics land depends on scheduling, which changes between runs, so the rounding (and therefore the last bits) differs. A CPU simulation of adding the same 1024 block partials in random orders gives dozens of distinct results spread over tens of ulps. The fix is a hierarchical, fixed-order reduction. Each thread accumulates many elements in an FP32 register with a coalesced grid-stride loop; each block reduces its threads with warp shuffles (`__shfl_down_sync`) and one small shared-memory step for the per-warp partials; thread 0 of each block writes one partial to an array. A second launch with a single block reduces the partials, again in a fixed order. This reads the 200 MB input once at close to peak bandwidth (on an A100 at about 2 TB/s the floor is about 0.1 ms) and always produces the same bits, so the test can compare exactly against the same algorithm or with a tolerance against a float64 reference. If one launch matters more than determinism, one atomic per block (a few hundred to a few thousand atomics) is fast, but still non-deterministic, so the test must then use a tolerance."
sources: [{"title": "Mark Harris, Optimizing Parallel Reduction in CUDA", "url": "https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf", "note": "The seven-step reduction tutorial with bandwidth after each change."}, {"title": "NVIDIA blog: Using CUDA Warp-Level Primitives", "url": "https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/", "note": "Shuffle and vote intrinsics, masks, and why implicit warp synchrony is unsafe since Volta."}, {"title": "GPU Gems 3, Chapter 39: Parallel Prefix Sum (Scan) with CUDA", "url": "https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda", "note": "Hillis-Steele vs Blelloch scans, bank-conflict avoidance, and multi-block scans."}, {"title": "Merrill and Garland, Single-pass Parallel Prefix Scan with Decoupled Look-back (2016)", "url": "https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back", "note": "The algorithm behind CUB's scan, matching memcpy throughput."}]
practice: ["dot-product-2", "max-of-array", "mean-and-variance", "argmax", "vector-sum-reduction"]
tags: ["reduction", "warp shuffle", "prefix sum"]
tools: ["CUDA", "CUB"]
---

## What you will learn

- What a **reduction** is, and why a good one is limited only by memory bandwidth.
- How a block adds up its values as a tree in shared memory, and why the pairing order decides whether warps diverge and banks conflict.
- How **warp shuffles** let 32 threads combine their registers without touching memory at all.
- How to finish a sum across the grid: a deterministic second pass, or atomics whose last bits change between runs.
- What a **prefix sum (scan)** is, and how Hillis-Steele and Blelloch compute it in parallel.

*Before you start:* you need grid-stride loops and warps (31.01), coalesced loads (31.02), and shared memory, `__syncthreads()` and bank conflicts (31.03). The Triton reduction lessons show `tl.sum`; this lesson shows what it does underneath.

Lesson 31.03 used shared memory so that threads in a block could reuse each other's loads. This lesson uses it for something new: threads that each hold a partial answer have to *combine* them into one. Sums, maxima, norms, variances, the softmax denominator, the loss average and argmax for greedy decoding all take many values and return one.

## The idea in one picture

Take eight numbers sitting in shared memory: 3, 1, 7, 0, 4, 1, 6, 3. One thread could add them one by one in 7 steps. Eight threads can do it in 3:

| Step | Active threads | What each thread `t` does | Values afterwards
| --- | --- | --- | --- |
| load | 0-7 | write its element to `a[t]` | 3 1 7 0 4 1 6 3
| 1 (s = 4) | 0-3 | `a[t] += a[t + 4]` | 7 2 13 3
| 2 (s = 2) | 0-1 | `a[t] += a[t + 2]` | 20 5
| 3 (s = 1) | 0 | `a[t] += a[t + 1]` | 25

~~~dg-fig 31.04-tree.svg
A tree reduction of eight values. At each step the number of partial sums halves; blue arrows keep a thread's own slot, orange arrows bring in the partner s places to the right. Three steps instead of seven, and every add in a step happens at the same time.
~~~

A **reduction** combines many values into one with an operation like `+` or `max`. A **tree reduction** combines pairs at every step, so $n$ values need $\log_2 n$ steps, yet still $n - 1 = 7$ additions in total. The tree does not save work; it lets the work happen side by side. The operation must be **associative**, $(a + b) + c = a + (b + c)$, because the tree adds in a different order than a loop. For integers and `max` that holds exactly; for floating point only approximately, which matters in Step 5.

## Step 1: why reductions are about bandwidth

A sum does one addition per 4-byte float it loads: 0.25 operations per byte, about the lowest **arithmetic intensity** a kernel can have. As 31.02 showed, such a kernel is **memory-bound**, so the speed limit is simple:

$$t_{\min} = \frac{\text{bytes read}}{\text{peak bandwidth}}.$$

Here *bytes read* is the size of the input (the output is a single number, so writes are negligible) and *peak bandwidth* is the spec-sheet memory speed of the GPU.

<div class="ex">
<div class="lab">Worked example</div>

**the speed limit for a 1 GiB sum**

Input: $2^{28}$ floats $= 268{,}435{,}456 \times 4$ bytes $= 1{,}073{,}741{,}824$ bytes (1 GiB).

On an A100 at about 2.0 TB/s (spec sheet value): $1.074 \times 10^{9} / 2.0 \times 10^{12} = 5.37 \times 10^{-4}$ s, about 0.54 ms.

No reduction can beat this, however clever its adds; on an H100 (about 3.35 TB/s) the floor is about 0.32 ms.

</div>

How far away can a naive kernel be? Mark Harris's NVIDIA tutorial measured seven versions of the same integer sum on a 2006-era G80 GPU, whose peak was 86.4 GB/s. The first reached 2.1 GB/s, about 2% of peak. The last reached 62.7 GB/s, 30 times faster, with the same algorithm and the same number of additions.

~~~dg-fig 31.04-harris.svg
Published numbers from Harris's tutorial (4M integers, G80), not measured here. Orange: interleaved addressing; blue: sequential addressing and first add during load; green: unrolling; last bar: many elements per thread. The arithmetic never changed; how well the hardware was used did.
~~~

The hardware is old but the reasons for each jump still hold, so the next steps follow them.

## Step 2: pick the pairing that keeps warps whole

Harris's first kernel used **interleaved addressing**: at step $s = 1, 2, 4, \dots$ thread `t` adds `a[t + s]` into `a[t]` only if `t % (2*s) == 0`. It is correct, but the active threads (0, 2, 4, ... then 0, 4, 8, ...) are scattered across every warp, and a warp with even one active lane must issue the add while the others wait: **warp divergence**. The second kernel packed the active threads with `index = 2 * s * tid`, but those addresses jump by `2s` words, so lanes collide in the same bank (31.03).

The fix is **sequential addressing**, the scheme in the picture: start with $s$ = half the block and halve it each step; thread `t` is active if `t < s` and adds `a[t + s]`.

```cpp
for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();                       // every step reads what the last step wrote
}
```

Active threads form a contiguous run from 0, so whole warps drop out together, and neighbours read neighbouring words, so banks never conflict. Counting both effects for a 256-thread block:

~~~dg-fig 31.04-addressing.svg
Counted from the index formulas for a 256-thread block, not timed. Left: warps with at least one active thread per step: 47 warp-steps for interleaved addressing, 12 for sequential. Right: strided interleaved indexing causes up to 8-way bank conflicts; sequential addressing none.
~~~

In Harris's measurements this switch doubled the bandwidth again, from 4.9 to 9.7 GB/s.

## Step 3: give every thread real work before the tree

Half the threads are still idle from the first tree step, and with one element per thread the block spends most of its time in tree steps and barriers rather than loads. Two changes fix this, and they matter most:

- **First add during load**: each thread loads two elements and adds them before writing to shared memory. Same data, half the blocks, no idle first step.
- **Many elements per thread**: a grid-stride loop (31.01) in which each thread adds many elements into a register before the tree starts.

The loop appears in the full kernel below. Afterwards each thread holds the sum of hundreds of elements, and the tree is a short tail on a long, coalesced streaming loop. Launch enough blocks to fill every SM a few times, and let each thread loop. The stride must be the whole grid: if thread 0 summed elements 0-999 and thread 1 summed 1000-1999, every warp load would touch 32 sectors instead of one 128-byte line.

## Step 4: finish the last 32 values with warp shuffles

Once the tree is down to 32 values, they all live in one warp. Instead of going through shared memory, threads of a warp can read each other's registers directly with a **warp shuffle**. `__shfl_down_sync(mask, v, off)` returns the value of `v` held by the lane `off` positions higher. Five shuffles reduce 32 values:

~~~dg-fig 31.04-shuffle.svg
Simulated in NumPy: lane i starts with i + 1. At offset 16, lanes 0-15 add lanes 16-31; then offsets 8, 4, 2, 1. Lane 0 ends with 528 = 1 + ... + 32. Grey lanes still compute, but their results are unused.
~~~

The standard building blocks look like this:

```cpp
__device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;                                   // lane 0 holds the warp's total
}
__device__ float block_sum(float v) {           // blockDim.x a multiple of 32, <= 1024
    __shared__ float warp_part[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);                            // 1) every warp reduces its own 32 values
    if (lane == 0) warp_part[w] = v;            // 2) one partial per warp into shared memory
    __syncthreads();
    v = (threadIdx.x < blockDim.x / 32) ? warp_part[lane] : 0.f;
    return w == 0 ? warp_sum(v) : 0.f;          // 3) warp 0 reduces the per-warp partials
}
```

A 1024-thread block needs only one `__syncthreads()`: no shared-memory tree at all. The `_sync` suffix and the mask `0xffffffff` (all 32 lanes) are not decoration. Since Volta, lanes of a warp are not guaranteed to run in lockstep, so Harris's kernel 5, which relied on implicit warp synchrony through a `volatile` shared array, is a race on current GPUs. Use shuffles or `__syncwarp()`; cooperative groups (`cg::reduce`) wrap the same operations.

## Step 5: across the grid

Blocks cannot wait for each other inside a launch, so a full sum ends in one of two ways. **Two passes**: each block writes `partials[blockIdx.x]`, and a second single-block launch reduces them. The launch boundary is a grid-wide barrier and the addition order is fixed, so the result is **deterministic**: same input, same bits. **Atomics**: each block ends with `atomicAdd(out, block_total)`. One launch, but blocks finish in a varying order, and with floating point that changes the answer:

~~~dg-fig 31.04-precision.svg
CPU simulations in NumPy. Left: summing uniform FP16 values; a running FP16 sum stalls at 2048 (94% too small at 65,536 values), an FP16 tree stays within 0.1%, an FP32 accumulator within 2 in a million. Right: 1024 block partials of 2^20 FP32 values added in 2000 random orders, as atomics would: 56 distinct results over about 57 ulps. A fixed order gives one value.
~~~

That is why a bitwise test of an atomic reduction fails intermittently: fine for a metric, not for a reproducibility test. Never do one atomic per *thread* into one address, though: atomics to a single address are serialised in the L2 cache, so millions of them run at the speed of one memory location (31.05 covers atomics in depth).

The left panel is a separate trap: the **accumulator** type. An FP16 number has 11 significant bits, so once the running sum reaches 2048 the gap between representable values is 2, and adding a number below 1 rounds back to 2048. Keep inputs in FP16 or BF16 if you like, but accumulate in FP32. The tree order also helps: pairwise error grows like $\log n$, not $n$.

## Reductions in code

The kernel that combines Steps 3-5 is short:

```cpp
__global__ void sum_kernel(const float* __restrict__ x, float* out, size_t n) {
    float v = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x)
        v += x[i];                              // many elements per thread, coalesced
    v = block_sum(v);                           // shuffles + one shared-memory step
    if (threadIdx.x == 0) out[blockIdx.x] = v;  // one partial per block
}
// pass 1: sum_kernel<<<num_blocks, 256>>>(d_x, d_partials, n);
// pass 2: sum_kernel<<<1, 1024>>>(d_partials, d_result, num_blocks);  // same kernel, one block
```

The CUDA above is not run here (no GPU); the same algorithm emulated in NumPy checks the logic:

```python
import numpy as np

def block_reduce(sdata):                 # sequential addressing over one block's shared memory
    s = sdata.copy(); half = len(s) // 2
    while half > 0:
        s[:half] += s[half:2 * half]     # threads tid < half: s[tid] += s[tid + half]
        half //= 2
    return s[0]

def two_pass_sum(x, blocks=64, threads=256):
    stride = blocks * threads
    pad = np.zeros(-len(x) % stride, x.dtype)
    regs = np.concatenate([x, pad]).reshape(-1, stride).sum(0, dtype=np.float32)  # grid-stride loop
    partials = np.array([block_reduce(regs[b*threads:(b+1)*threads]) for b in range(blocks)], np.float32)
    return block_reduce(partials)        # second launch: one block, 64 partials

print(block_reduce(np.array([3, 1, 7, 0, 4, 1, 6, 3])))   # 25
x = np.random.default_rng(0).random(1_000_000, dtype=np.float32)
print(two_pass_sum(x), x.astype(np.float64).sum())         # 500377.28  500377.2696207166
```

For a plain sum use `torch.sum` or `cub::DeviceReduce::Sum`; write your own when the reduction is **fused** into a softmax, layer norm or loss so the data is read once.

## Step 6: prefix sums (scans)

A **prefix sum**, or **scan**, gives every position the sum of everything up to it. The **inclusive** scan of $[3, 1, 7, 0, 4]$ is $[3, 4, 11, 11, 15]$. The **exclusive** scan excludes the element itself: shift right and start at 0, giving $[0, 3, 4, 11, 11]$.

Scans look sequential but parallelise well, and inference code uses them constantly. Sequence lengths $[5, 3, 7, 2]$ packed into one buffer start at the exclusive scan $[0, 5, 8, 15]$ (**ragged-batch offsets**). Keep flags $[1, 0, 1, 1, 0, 1]$ scan to $[0, 1, 1, 2, 3, 3]$, each kept element's output slot (**stream compaction**, the core of radix sort). Top-p sampling cuts the sorted vocabulary where the cumulative probability crosses $p$.

**Hillis-Steele** is the simplest parallel scan. At step $d = 0, 1, 2, \dots$ every element adds the element $2^d$ places to its left (if there is one). On our eight values: after offset 1, $[3, 4, 8, 7, 4, 5, 7, 9]$; after offset 2, $[3, 4, 11, 11, 12, 12, 11, 14]$; after offset 4, $[3, 4, 11, 11, 15, 16, 22, 25]$, the inclusive scan. Three steps, but $7 + 6 + 4 = 17$ additions against 7 for a loop: it is not **work-efficient**.

**Blelloch's** scan fixes the work in two sweeps. The **up-sweep** is exactly a tree reduction, done in place so the partial sums stay in the array. The **down-sweep** sets the last slot to 0 and walks back down the tree: at each node, the left child receives the parent's value and the right child receives the parent's value plus the old left child.

~~~dg-fig 31.04-scan.svg
Left: Blelloch's scan on the eight values; shaded cells were written in that step. Right: additions per element, counted by simulation: Hillis-Steele grows like log n (19 at a million), Blelloch stays under 2, a loop under 1.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**Blelloch scan by hand on 3 1 7 0 4 1 6 3**

Up-sweep, stride 1 (odd slots add their left neighbour): $1+3 = 4$, $0+7 = 7$, $1+4 = 5$, $3+6 = 9$.

Stride 2: slot 3 gets $7 + 4 = 11$, slot 7 gets $9 + 5 = 14$. Stride 4: slot 7 gets $14 + 11 = 25$, the total.

Set the last slot to 0: $[3, 4, 7, 11, 4, 5, 6, 0]$.

Down-sweep, stride 4 (pair slots 3 and 7): slot 3 takes 0, slot 7 takes $0 + 11 = 11$: $[3, 4, 7, 0, 4, 5, 6, 11]$.

Stride 2 (pairs 1-3 and 5-7): slot 1 takes 0, slot 3 takes $0 + 4 = 4$; slot 5 takes 11, slot 7 takes $11 + 5 = 16$: $[3, 0, 7, 4, 4, 11, 6, 16]$.

Stride 1 (pairs 0-1, 2-3, 4-5, 6-7): $(3, 0) \to (0, 3)$, $(7, 4) \to (4, 11)$, $(4, 11) \to (11, 15)$, $(6, 16) \to (16, 22)$.

Result $[0, 3, 4, 11, 11, 15, 16, 22]$, the exclusive scan, with 14 additions against Hillis-Steele's 17 (at $n = 2^{20}$: 2.1 million against 19.9 million).

</div>

In general Hillis-Steele does $n \log_2 n - (n - 1)$ additions in $\log_2 n$ steps; Blelloch does $2(n - 1)$ in $2 \log_2 n$ steps.

Within a warp, scans use `__shfl_up_sync`. For arrays larger than a block, the classic recipe scans each block's chunk, scans the block totals, then adds each block's offset back, moving the data several times. Merrill and Garland's **decoupled look-back** does it in one pass: each block publishes its total with a status flag and reads its predecessors' flags to find its offset, running at about memory-copy speed. `cub::DeviceScan::InclusiveSum` and `torch.cumsum` already do this; write your own only to learn or to fuse.

## Checking a reduction or scan

- **Reference**: compare with a float64 CPU sum using a relative tolerance (a few times $10^{-6}$ for FP32), never bitwise unless the order is fixed.
- **Odd sizes**: $n = 0, 1, 31, 33$ and non-multiples of the block expose off-by-one bugs.
- **Bandwidth**: bytes read divided by kernel time should approach the spec-sheet value; if not, fix the loop before the tree.
- **Determinism**: hash the outputs of many runs; one hash means a fixed order.
- **Scans**: the last inclusive value equals the total; exclusive equals inclusive minus input.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Wrong total, varies between runs | missing `__syncthreads()` between tree steps | barrier after every shared-memory step
| Hangs or deadlocks | `__syncthreads()` inside `if (tid < s)` | keep barriers outside the branch, reached by all threads
| Wrong on Volta and newer, fine on old GPUs | implicit warp synchrony with `volatile` shared memory | use `__shfl_down_sync` or `__syncwarp()`
| Very slow, single-digit % of bandwidth | one element per thread, tree dominates | grid-stride loop with many elements per thread
| Last bits differ between runs | atomics or nondeterministic block order with floats | two-pass reduction with fixed order
| Sum of many FP16 values far too small | FP16 accumulator stalls at 2048 | accumulate in FP32
| Negative variance | $E[x^2] - E[x]^2$ cancels when the mean is large | Welford or (count, mean, M2) merging
| Hillis-Steele scan gives garbage | reading and writing one array in one step | double-buffer

For the variance row: one million FP32 values with mean 10,000 and variance about 1 gave $E[x^2] - E[x]^2 = -8.0$ in our NumPy run, while subtracting the mean first gave 0.998, matching float64. Both terms are near $10^8$, where FP32 cannot resolve a difference of 1.

## Going deeper

**Work, steps and Brent's bound.** With $P$ processors a tree reduction takes $O(n / P + \log n)$ time. One thread per element costs (processors times time) $O(n \log n)$; $O(n / \log n)$ threads each summing $O(\log n)$ elements first bring it back to $O(n)$. This is Brent's argument for many elements per thread, which Harris calls algorithm cascading.

**Combining more than sums.** Anything with an associative combine step can be tree-reduced. Mean and variance merge as triples: for two groups $(n_A, \mu_A, M_A)$ and $(n_B, \mu_B, M_B)$, where $M$ is the sum of squared deviations from the group mean, let $\delta = \mu_B - \mu_A$ and $n = n_A + n_B$; then $\mu = \mu_A + \delta\, n_B / n$ and $M = M_A + M_B + \delta^2 n_A n_B / n$. This is Chan's parallel form of Welford's update, a common choice in layer-norm kernels. Argmax reduces (value, index) pairs; online softmax reduces (max, sum) pairs.

**Scans of any associative operator.** Scanning $(a, b)$ pairs that represent $h \mapsto a h + b$ computes a linear recurrence in parallel, which is how state-space models such as Mamba parallelise across a sequence.

**Traffic of multi-block scans.** Scan-then-propagate moves about $4n$ elements, reduce-then-scan about $3n$, decoupled look-back $2n$, the same as a copy.

## Key terms

| Term | Meaning
| --- | --- |
| Reduction | Combining many values into one with an associative operation (+, max, argmax)
| Tree reduction | Combining pairs in $\log_2 n$ steps, $n - 1$ operations in total
| Sequential addressing | Thread `t` adds `a[t + s]`, $s$ halving; contiguous, conflict-free
| Warp shuffle | `__shfl_*_sync`: reading another lane's register directly
| Two-pass reduction | Per-block partials, then one block reduces them; deterministic
| Accumulator | The variable holding the running sum; use at least FP32
| Scan (prefix sum) | Each output combines all inputs up to it (inclusive) or before it (exclusive)
| Work-efficient | No more total operations than the sequential algorithm, up to a constant

<div class="take">
<div class="lab">Key takeaway</div>

a reduction is memory-bound, so make it stream: many elements per thread in a coalesced grid-stride loop, warp shuffles for the tail, FP32 accumulators, and a fixed-order second pass when results must be reproducible. Scans reuse the tree; know Blelloch's version and call a library in practice.

</div>
