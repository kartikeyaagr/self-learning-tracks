---
id: "31.07"
course: "inference-eng-course"
module: "31"
slug: "occupancy-and-performance"
title: "Occupancy and performance"
keyIdeas: ["GPUs hide memory latency by switching between resident warps; **occupancy** is active warps per SM divided by the maximum (64 on A100 and H100). Little's law says how much work must be in flight: roughly latency divided by the work each warp does per load.", "Occupancy is capped by whichever resource runs out first: threads (2048 per SM), blocks (32 per SM), **registers** (65,536 per SM) or **shared memory** per SM. At 64 registers per thread, a 256-thread block uses 16,384 registers, so only 4 blocks fit: 50% occupancy. The limits are staircases, so one extra register can cost a whole block.", "More occupancy is not always faster: kernels with lots of **instruction-level parallelism** (several independent loads in flight per thread, register tiling) often peak at 25 to 50% occupancy, and forcing registers down can cause slow spills.", "The **roofline** bounds attainable throughput by $\\min(\\text{peak FLOP/s},\\; I \\times \\text{bandwidth})$ with arithmetic intensity $I$ in FLOP/byte; the ridge point (about 10 FLOP/byte for FP32 on A100, roughly 300 for BF16 Tensor Cores on H100) separates memory-bound from compute-bound kernels.", "Work in a loop: compute the kernel's intensity, measure achieved bandwidth or FLOP/s against the relevant roof in Nsight Compute, and only then tune occupancy, layout or fusion. A kernel at 85-90% of its roof is done; only moving fewer bytes or using Tensor Cores can make it faster."]
question: "A custom attention-score kernel uses 128 registers per thread with 256-thread blocks and no shared memory. Nsight Compute reports 25% occupancy, 35% of peak memory bandwidth and 20% of peak compute, with warps mostly stalled waiting on global loads. Explain the occupancy figure and give two different ways to improve performance."
answer: "At 128 registers per thread, one warp needs $128 \\times 32 = 4096$ registers and a 256-thread block (8 warps) needs 32,768, so only 2 blocks (16 warps) fit within an SM's 65,536 registers. Threads would allow 8 blocks and block slots 32, so registers are the binding limit: 16 of 64 warps is 25% occupancy. With neither memory (35%) nor compute (20%) near its roof and warps stalled on loads, the kernel is latency-bound: there is not enough work in flight to cover HBM latency, by Little's law. Raising occupancy is therefore a valid lever here, unlike for a kernel already near peak bandwidth. One fix is to increase thread-level parallelism: cap registers with `__launch_bounds__(256, 4)` or restructure to use fewer (64 per thread gives 4 blocks, 50%), and check the `-Xptxas -v` output to make sure this does not cause spill stores, which would add local-memory traffic. The other is to increase instruction-level parallelism and reuse without raising occupancy: issue several independent loads per thread before using any of them (unrolling, vectorised 16-byte loads), stage tiles in shared memory with double buffering (or `cp.async`) so loads for the next tile overlap computation on the current one, and reuse each loaded value across several outputs per thread, which also raises arithmetic intensity. Either way, re-measure against the roofline after each change: success means memory or compute throughput climbing toward its roof, not occupancy as such."
sources: [{"title": "CUDA C++ Best Practices Guide: Occupancy", "url": "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy", "note": "How registers, shared memory and block size limit residency, and the occupancy APIs."}, {"title": "Volkov, Better Performance at Lower Occupancy (GTC 2010)", "url": "https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf", "note": "Slides showing instruction-level parallelism hiding latency at low occupancy."}, {"title": "Williams, Waterman and Patterson, Roofline (UC Berkeley tech report, 2008)", "url": "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf", "note": "The original roofline model: arithmetic intensity, ridge points and ceilings."}, {"title": "Horace He, Making Deep Learning Go Brrrr From First Principles", "url": "https://horace.io/brrr_intro.html", "note": "Compute-, memory- and overhead-bound regimes in PyTorch with worked numbers."}]
practice: ["gpu-occupancy-calculator", "analyze-prefill-and-decode-with-roofline-model", "measure-kernel-timeline-utilization"]
tags: ["occupancy", "roofline"]
tools: ["CUDA", "Nsight Compute"]
---

## What you will learn

- Why a GPU needs many loads in flight at once, and how switching between resident warps hides memory latency.
- What **occupancy** is and how to compute it by hand from threads, registers and shared memory per block.
- Why higher occupancy is not always faster: **instruction-level parallelism** can hide latency with fewer warps.
- How the **roofline** turns two spec-sheet numbers into a ceiling for any kernel, and how to tell memory-bound from compute-bound.
- A repeatable tuning loop: decide which roof applies, measure how close you are, and only then touch occupancy.

*Before you start:* you should know how threads, blocks and warps map onto SMs (31.01), what coalesced loads are (31.02), and how tiling raises arithmetic intensity (31.03, which drew a first A100 roofline). The idea of compute-bound versus memory-bound comes from "Compute-Bound vs Memory-Bound & the Roofline Model" earlier in the course.

Lessons 31.02 to 31.06 each fixed one cause of slowness, ending with fused ML primitives. This last lesson asks two questions about any kernel: *how many warps can the hardware keep busy*, and *what is the fastest it could possibly run*? The answers tell you which earlier fix to reach for.

## The idea in one picture

A load from HBM takes several hundred cycles to come back. A GPU does not wait for it: each SM keeps many warps **resident** (loaded and ready to run) and, every cycle, its schedulers issue an instruction from whichever warp has its inputs ready. A warp waiting on a load is simply skipped.

To see why this matters, take a toy SM with one issue slot. Each warp loops: issue one load, wait 24 cycles for it, then issue 5 compute instructions that use the loaded value. One loop is 6 instructions spread over 30 cycles.

| Resident warps | Instructions issued per 30-cycle loop | Issue slot busy (steady state)
| --- | --- | --- |
| 1 | 6 | 6 / 30 = 20%
| 3 | 18 | 18 / 30 = 60%
| 5 | 30 | 30 / 30 = 100%

~~~dg-fig 31.07-timeline.svg
A toy scheduler simulated cycle by cycle. With one warp the issue slot (green row) is idle while the load is outstanding. With five warps, each one's wait is filled with the others' work; after the first 24-cycle warm-up the slot is busy every cycle.
~~~

In the bottom panel, warp 0 loads at cycle 0 and idles until cycle 24 while warps 1 to 4 issue their loads. From then on some warp is always ready. Nothing got faster; the waiting just overlaps.

**Occupancy** measures how many warps an SM is holding compared with the most it could hold:

$$\text{occupancy} = \frac{\text{active warps per SM}}{\text{maximum warps per SM}}$$

On A100 and H100 the maximum is 64 warps (2048 threads), so 5 resident warps would be 5/64, about 8%.

## Step 1: how many warps you need (Little's law)

The table above is an instance of **Little's law**: the amount of work in flight equals throughput times latency. If each warp has one load outstanding at a time, you need roughly

$$N_{\text{warps}} \approx \frac{L + W}{W}$$

warps to keep the issue slot full, where $L$ is the load latency in cycles and $W$ is the number of instructions a warp issues per load. Number check: $L = 24$, $W = 6$ gives $(24 + 6)/6 = 5$, matching the table.

With a more realistic latency of 400 cycles and 9 instructions per load, the same formula asks for $409/9 \approx 45$ warps, which is 70% occupancy. The same law applies to bandwidth. A spec-sheet H100 SXM moves 3.35 TB/s over 132 SMs, about 25.4 bytes per nanosecond per SM. If a load takes 500 ns (an assumed round number, for illustration), each SM must have about $25.4 \times 500 \approx 12{,}700$ bytes in flight to keep HBM busy. A warp doing one 16-byte vector load per lane has 512 bytes in flight, so that is about 25 warps, each with one load outstanding. With plain 4-byte loads, it would take four times as many.

## Step 2: what limits residency

Active warps depend on how many *blocks* fit on an SM at once. A block is placed whole or not at all, and four resources are checked; the one that runs out first sets the answer.

| Resource per SM (A100) | Limit | A block of 256 threads using... | Blocks that fit
| --- | --- | --- | --- |
| Threads | 2048 | 256 threads | 8
| Blocks | 32 | 1 block | 32
| Registers | 65,536 | 64 regs/thread = 16,384 | 4
| Shared memory | ~164 KB usable | 48 KB | 3

The binding limit is the minimum: here shared memory allows 3 blocks, so 24 warps are active, 37.5% occupancy. Drop to 32 registers per thread and 16 KB of shared memory, and threads become the limit at 8 blocks: 64 warps, 100%.

Two details make hand calculations slightly off if you ignore them. Registers are allocated per warp in chunks of 256, so with $R$ registers per thread (as `nvcc -Xptxas -v` reports) a warp uses $\lceil 32R/256 \rceil \times 256$ registers. And each block reserves a little extra shared memory (1 KB on A100), so a 48 KB block really costs 49 KB. The CUDA occupancy calculator, the runtime API and Nsight Compute all apply these rules for you.

~~~dg-fig 31.07-limits.svg
Occupancy of 256-thread blocks on an A100, computed from the limits above. Each curve is a staircase: occupancy only moves when one whole block more or fewer fits, so a one-register change can drop it by a full step.
~~~

The staircase shape is the practical lesson. Going from 64 to 65 registers per thread does not cost 1.5%; it costs a whole block. A warp then needs $65 \times 32 = 2080$ registers, rounded up to 2304, and $65{,}536 / 2304 = 28$ warps, which is only 3 blocks of 8 warps: occupancy falls from 50% to 37.5%.

<div class="ex">
<div class="lab">Worked example</div>

**occupancy by hand, then after a compiler change**

A kernel launches 256-thread blocks (8 warps each), uses 40 registers per thread and 20 KB of shared memory per block, on an A100.

Threads: 64 warps / 8 warps per block = 8 blocks. Block slots: 32 blocks.

Registers: one warp needs $40 \times 32 = 1280$ registers, already a multiple of 256. The SM holds $65{,}536 / 1280 = 51.2$, so 51 warps, which is $\lfloor 51/8 \rfloor = 6$ blocks.

Shared memory: each block costs 20 + 1 = 21 KB. $164 / 21 = 7.8$, so 7 blocks.

The minimum of (8, 32, 6, 7) is 6 blocks, set by registers: $6 \times 8 = 48$ warps, and $48/64 = 75\%$ occupancy.

Now a code change makes the compiler use 48 registers. A warp needs $48 \times 32 = 1536$; $65{,}536 / 1536 = 42.7$, so 42 warps, $\lfloor 42/8 \rfloor = 5$ blocks, 40 warps, $40/64 = 62.5\%$. Nothing else changed; registers alone cost one block.

</div>

## Step 3: occupancy in code

In CUDA, ask the runtime rather than computing by hand, and read the compiler's resource report.

```cpp
int blocks_per_sm = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, my_kernel, 256, smem_bytes);
int min_grid, best_block;
cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block, my_kernel, smem_bytes, 0);
// nvcc -Xptxas -v  prints registers and shared memory per kernel
// __launch_bounds__(256, 2) asks the compiler to fit 2 blocks of 256 per SM
```

To build intuition without a GPU, the same rules fit in a few lines of Python. This version follows the A100 (compute capability 8.0) limits listed above:

```python
import math

def occupancy(threads, regs, smem_kb, sm_smem_kb=164):
    # blocks per SM and occupancy on an A100-class SM (CC 8.0 rules)
    warps = math.ceil(threads / 32)
    regs_per_warp = math.ceil(regs * 32 / 256) * 256        # allocated per warp, in chunks of 256
    limits = {
        "threads": 64 // warps,                               # 64 warps = 2048 threads
        "blocks": 32,
        "registers": (65536 // regs_per_warp) // warps,
        "smem": int(sm_smem_kb // (smem_kb + 1)) if smem_kb else 32,  # +1 KB reserved per block
    }
    blocks = min(limits.values())
    return blocks, blocks * warps / 64, min(limits, key=limits.get)

print(occupancy(256, 64, 48))   # (3, 0.375, 'smem')
print(occupancy(256, 40, 20))   # (6, 0.75, 'registers')
print(occupancy(256, 48, 20))   # (5, 0.625, 'registers')
print(occupancy(256, 65, 0))    # (3, 0.375, 'registers')  one register past 64
print(occupancy(128, 64, 0))    # (8, 0.5, 'registers')
```

The last line shows that smaller blocks do not help when registers are the limit. For H100, pass `sm_smem_kb=228`; the other limits are the same.

Forcing fewer registers (`-maxrregcount`, `__launch_bounds__`) raises occupancy but can cause **spills** to local memory, which lives in global memory behind the caches. Spills often cost more than the occupancy gains; check the `-Xptxas -v` output for spill stores.

## Step 4: occupancy is a means, not a goal

Volkov's GTC 2010 talk, "Better Performance at Lower Occupancy", made the point that has held ever since. Latency can be hidden in two ways:

- **Thread-level parallelism (TLP):** many warps, each with one thing outstanding. This is what occupancy counts.
- **Instruction-level parallelism (ILP):** each warp issues several *independent* loads or math operations before it needs any of the results.

Little's law does not care which one supplies the work in flight. If each warp issues $k$ independent loads before waiting, the warps needed drop to about $(L + kW)/(kW)$.

~~~dg-fig 31.07-ilp.svg
Toy simulation (one issue slot, 400-cycle load latency, 9 instructions per load). Dotted lines mark the predicted knee. One load per warp needs about 45 warps to saturate; four independent loads per warp saturate with about 12, under 20% occupancy.
~~~

The simulation numbers: with 16 resident warps (25% occupancy), the issue slot is busy 35% of cycles with one load per warp, 67% with two, and 97% with four. Same occupancy, nearly three times the useful work.

This is why fast kernels often run at modest occupancy. A matmul thread that computes an 8 x 8 output tile in registers issues many independent FMAs and loads; it uses many registers, so occupancy falls, and it is faster anyway. Fast GEMM and attention kernels commonly run at 12 to 50% occupancy.

Low occupancy is worth fixing when a kernel is **latency-bound**. Nsight Compute shows it this way: warps mostly stalled on memory dependencies, few eligible warps per cycle, and neither bandwidth nor compute anywhere near peak.

## Step 5: the roofline gives the target

Occupancy tells you whether the SM *can* keep busy. The roofline tells you how fast the kernel *could* go if it did. It bounds a kernel's throughput from the device's peak compute rate $P$ (FLOP/s), its peak memory bandwidth $\beta$ (bytes/s), and the kernel's **arithmetic intensity** $I$ = FLOPs performed / bytes moved to and from DRAM:

$$\text{attainable FLOP/s} = \min(P,\; I \cdot \beta)$$

The **ridge point** $P/\beta$ is the intensity where the two roofs meet. Below it, the kernel is memory-bound; above it, compute-bound. Using spec-sheet values: A100 80 GB has a FP32 ridge near 9.6 and a BF16 Tensor Core ridge near 153 FLOP/byte; H100 SXM (3.35 TB/s, 67 TFLOPS FP32, 989 TFLOPS dense BF16) has ridges near 20 and 295.

~~~dg-fig 31.07-roofline.svg
Roofline for an H100 SXM drawn from spec-sheet peaks, not measurements. Elementwise, normalisation and batch-1 decode kernels sit on the sloped bandwidth roof. A large BF16 GEMM sits under the flat Tensor Core roof, far above the FP32 CUDA-core roof.
~~~

| Kernel | Intensity (FLOP/B) | Bound by
| --- | --- | --- |
| Vector add, FP32 | 1 FLOP / 12 B = 0.08 | memory
| Softmax or LayerNorm, BF16 | ~2-5 | memory
| Tiled FP32 matmul, 32 x 32 tiles (global traffic) | 8 | memory, near the A100 FP32 ridge
| Decode-time GEMV, BF16 weights, batch 1 | ~1 | memory
| Decode GEMV, batch $B$ (weights dominate traffic) | ~$B$ | memory until $B$ nears the ridge
| Square GEMM, BF16, N = 4096 | $2N^3 / (3 \cdot 2N^2) = N/3 \approx 1365$ | compute

The roofline gives each kernel a finish line. A memory-bound kernel is done when it reaches 80 to 90% of $\beta$. Making its arithmetic faster is pointless; the only remaining options are moving fewer bytes (fusion, lower precision, avoiding re-reads). A compute-bound kernel is done when it approaches $P$ for the instruction type it uses, which for FP16/BF16 means Tensor Cores. An FP32 CUDA-core matmul on an H100 can never exceed about 7% of the BF16 Tensor Core peak (67 / 989), however well it is tiled.

## The roofline in code

A best-case time is just the larger of "time to move the bytes" and "time to do the math":

```python
SPECS = {  # spec-sheet peaks: (HBM bytes/s, FP32 FLOP/s, BF16 Tensor Core dense FLOP/s)
    "A100-80GB": (2.039e12, 19.5e12, 312e12),
    "H100-SXM":  (3.35e12, 67e12, 989e12),
}

def roofline(flops, bytes_moved, gpu="H100-SXM", tensor_cores=True):
    bw, p32, ptc = SPECS[gpu]
    peak = ptc if tensor_cores else p32
    t_mem, t_math = bytes_moved / bw, flops / peak
    bound = "memory" if t_mem > t_math else "compute"
    return flops / bytes_moved, max(t_mem, t_math) * 1e6, bound   # intensity, best-case us, regime

n = 4096
print(roofline(2 * n * n, 2 * n * n))                 # decode GEMV, batch 1, BF16 weights
# (1.0, 10.02, 'memory')
print(roofline(2 * n**3, 3 * 2 * n * n))              # square BF16 GEMM, A, B, C each read/written once
# (1365.33, 138.97, 'compute')
rows, d = 8192, 4096
print(roofline(4 * rows * d, 2 * 2 * rows * d))       # RMSNorm: read + write BF16, ~4 FLOP/element
# (1.0, 40.06, 'memory')
```

The GEMV line is the core of LLM decode: moving the 32 MiB weight matrix takes about 10 µs, the math alone 0.03 µs. That 300x gap is why batching, which reuses each weight for $B$ tokens, drives decode throughput.

## Reading the result: is the kernel done?

Put the measurement next to the ceiling before changing anything.

<div class="ex">
<div class="lab">Worked example</div>

**a kernel that is already done**

Suppose Nsight Compute reports that an RMSNorm kernel on an 8192 x 4096 BF16 tensor takes 46 µs on an H100 SXM, at 50% occupancy, and flags the occupancy as low.

Bytes moved: it reads $8192 \times 4096 \times 2 = 64$ MiB and writes another 64 MiB, so 128 MiB, which is 134.2 million bytes.

Achieved bandwidth: $134.2 \times 10^6 / 46 \times 10^{-6} \approx 2.92 \times 10^{12}$ B/s = 2.9 TB/s.

Against the roof: $2.92 / 3.35 \approx 87\%$ of peak bandwidth. The roofline code above agrees: best case 40.06 µs.

Verdict: raising occupancy will not help, because there is almost no bandwidth left to gain. Only fusing it with the preceding residual add (31.06) or the following matmul's prologue would make the layer faster, because that removes a whole read and write of the tensor.

</div>

~~~dg-fig 31.07-diagnose.svg
Model curves from the Little's-law formula. A latency-bound kernel gains from occupancy until latency is covered, and gains sooner with more ILP. A kernel already at 87% of peak bandwidth stays flat: occupancy is not what limits it.
~~~

Nsight Compute's "Speed of Light" section reports exactly these two percentages. If one is high (above about 70-80%), you are near a roof: change the algorithm, not the launch configuration. If both are low with warps stalled on memory, add work in flight, through more warps or more independent loads per warp.

## A tuning loop

~~~dg-flow Classify a kernel with the roofline before tuning it.
Count FLOPs and bytes -(intensity I)-> Compare with ridge P/β -(I below ridge)-> *Memory-bound: cut bytes -(I above ridge)-> Compute-bound: use Tensor Cores
~~~

1. Profile end to end first (Nsight Systems) and pick the kernels that dominate time.
2. For each, compute intensity by hand and read Nsight Compute's "Speed of Light" section: achieved percentage of memory and compute peaks.
3. If neither is high, read the stall reasons and occupancy: uncoalesced access (sectors per request, 31.02), bank conflicts (31.03), too few eligible warps, or a grid too small to fill all SMs.
4. Change one thing, re-measure, keep a log.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Chasing 100% occupancy | registers capped, spill stores appear, kernel gets slower | accept lower occupancy if ILP and reuse are high; check `-Xptxas -v` for spills
| Tuning a kernel already at its roof | weeks of tweaks, under 5% gain | compare achieved bandwidth or FLOP/s to the spec roof first
| Ignoring the register staircase | a small edit drops occupancy by a whole step | watch the register count in each build; use `__launch_bounds__` to pin it
| Too-small grid | low SM utilisation even at high occupancy | launch at least several blocks per SM, or split work further
| FP32 math where Tensor Cores apply | compute-bound kernel stuck below about 7% of the BF16 peak | use BF16/FP16 Tensor Core paths (WMMA, CUTLASS, cuBLAS)

## Going deeper

**Why registers win over occupancy.** Registers are the fastest storage on the chip and the 65,536 per SM are shared by every resident thread. Spending them on register tiles (one thread computing many outputs) multiplies data reuse: an $r \times r$ tile per thread performs $r^2$ FMAs per $2r$ values loaded. Spending them on more threads only adds latency hiding. Once latency is hidden, more threads buy nothing, while reuse keeps paying.

**Little's law per resource.** The law applies to each pipeline separately: HBM needs bytes in flight, Tensor Cores need independent MMAs in flight. Asynchronous copies (`cp.async` on A100, TMA on H100) and warp specialisation keep more work in flight without more threads.

**Refinements of the roofline.** Real devices have several bandwidth roofs (L2, shared memory) and several compute roofs; the hierarchical roofline plots intensity against each memory level. A kernel below every roof is limited by something the model ignores: latency, launch overhead, synchronisation or a small grid.

**Dynamic limits.** Shared memory per SM is a carveout shared with L1 cache (up to 164 KB on A100, 228 KB on H100), and more than 48 KB per block needs a `cudaFuncSetAttribute` opt-in.

## Key terms

| Term | Meaning
| --- | --- |
| Resident warp | A warp whose block is loaded on an SM and that the schedulers may pick each cycle
| Occupancy | Active warps per SM divided by the maximum (64 on A100 and H100)
| Latency hiding | Overlapping one warp's wait with other ready work
| Little's law | Work in flight = throughput × latency; sets how much parallelism is needed
| ILP | Independent instructions or loads within one warp that can be in flight together
| Register spill | Values the compiler cannot keep in registers, moved to slow local memory
| Arithmetic intensity | FLOPs performed per byte moved to and from DRAM
| Roofline / ridge point | $\min(P, I\beta)$; the intensity $P/\beta$ where the memory roof meets the compute roof

<div class="take">
<div class="lab">Key takeaway</div>

compute occupancy from registers, shared memory and block size to know what limits residency, but judge a kernel by where it sits under the roofline: push memory-bound kernels towards peak bandwidth by moving fewer bytes, and compute-bound ones towards the Tensor Core roof.

</div>
