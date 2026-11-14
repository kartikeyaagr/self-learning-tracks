---
id: "31.01"
course: "inference-eng-course"
module: "31"
slug: "threads-blocks-and-grids"
title: "Threads, blocks and grids"
keyIdeas: ["A CUDA kernel is one function run by many threads. Threads are grouped into **blocks** (up to 1024 threads, scheduled together on one SM, able to share memory and synchronise) and blocks form a **grid**; the hardware executes threads in **warps** of 32.", "Each thread computes its own work item from built-in indices: `i = blockIdx.x * blockDim.x + threadIdx.x` in 1-D, and a (row, col) pair in 2-D with `threadIdx.x` mapped to the contiguous dimension so each warp reads neighbouring addresses.", "The grid is sized by ceiling division, `(n + B - 1) / B` blocks, so the last block usually has spare threads and **every kernel needs a bounds check**.", "Block sizes should be multiples of 32; 128 or 256 is a good start. An SM holds at most 2048 threads and 32 blocks on A100/H100, so very small or awkward block sizes leave it partly empty.", "A **grid-stride loop** decouples grid size from problem size: launch a few blocks per SM and let each thread walk the array in steps of `gridDim.x * blockDim.x`.", "Launches are asynchronous and do not report errors by themselves; check `cudaGetLastError()` after the launch, synchronise before reading results, and time with `cudaEvent` pairs."]
question: "A colleague's 2-D kernel uses `int row = blockIdx.x * blockDim.x + threadIdx.x; int col = blockIdx.y * blockDim.y + threadIdx.y;` with `dim3 block(32, 8)` on a row-major 4096 x 4096 float matrix. Results are correct but the kernel is about 5x slower than expected. Explain why, and what else could go wrong for other matrix shapes."
answer: "A warp is formed from consecutive linear thread ids, with `threadIdx.x` varying fastest, so with `block(32, 8)` each warp is 32 threads that share `threadIdx.y` and have `threadIdx.x` = 0..31. With `x` mapped to rows, those 32 threads handle 32 different rows of the same column. In row-major storage those elements are 4096 floats (16 KiB) apart, so each warp load touches 32 separate memory segments instead of one contiguous 128-byte span, and most of every memory transaction is wasted. The arithmetic is identical, which is why the results are correct. Swapping the mapping so that `threadIdx.x` indexes columns (`col = blockIdx.x * blockDim.x + threadIdx.x`, `row = blockIdx.y * blockDim.y + threadIdx.y`) makes each warp read contiguous memory, and the grid must be swapped to match: `dim3 grid((cols + 31) / 32, (rows + 7) / 8)`. For other shapes, the grid must be sized per axis with ceiling division and the bounds check must test both `row < rows` and `col < cols`. With non-square matrices, mixing up which grid dimension covers rows and which covers columns either leaves part of the matrix unprocessed or launches threads outside it, and a check on only one axis then writes out of bounds. Very tall matrices can also exceed the 65535-block limit on `grid.y` if rows are mapped to y; the fix is a grid-stride loop over rows, or putting the long axis on `x`, whose limit is 2^31 - 1. Finally, for matrices with more than about 2^31 elements, `row * cols + col` overflows a 32-bit int, so the offset should be computed in 64-bit."
sources: [{"title": "CUDA C++ Programming Guide: Programming Model", "url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html", "note": "Chapter on the thread hierarchy, kernels and the execution configuration; the reference for every limit quoted here."}, {"title": "NVIDIA blog: An Even Easier Introduction to CUDA", "url": "https://developer.nvidia.com/blog/even-easier-introduction-cuda/", "note": "First kernel end to end: memory, launch, timing, and the effect of adding blocks."}, {"title": "NVIDIA blog: CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", "url": "https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/", "note": "Short, with the scalability and debugging arguments for grid-stride loops."}]
practice: ["vector-addition", "matrix-addition", "vector-subtraction", "relu"]
tags: ["thread hierarchy", "launch configuration"]
tools: ["CUDA"]
---

## What you will learn

- What a CUDA kernel is, and how its threads are arranged into a **grid** of **blocks**.
- How each thread finds its element from built-in indices, in 1-D and 2-D, and why `threadIdx.x` goes on the contiguous axis.
- How to size a launch with ceiling division, and why every kernel needs a bounds check.
- How to choose a block size from the hardware limits.
- How grid-stride loops work, and how to catch errors from asynchronous launches.

*Before you start:* you should know what a GPU streaming multiprocessor (SM) and a warp are ("GPU Threads, Warps, Blocks & Memory Access"), and have seen Triton kernels ("Triton & Kernel Engineering"). Reading C is enough.

## Where this module starts

Earlier modules worked at the Triton level and through profilers ("Profiling & GPU Performance"). Triton hides individual threads: the compiler decides how threads, shared memory and vector loads are used. This module goes one level down, to CUDA C++, where you make those decisions yourself.

That is the level at which cuBLAS, FlashAttention, vLLM's custom kernels and most PyTorch extensions are written. The hardware ideas from "GPU Threads, Warps, Blocks & Memory Access" now become code.

This lesson is about *which thread does which work*; memory access (31.02), shared memory (31.03), reductions (31.04), atomics (31.05), ML primitives (31.06) and occupancy (31.07) follow.

## The idea in one picture

Suppose you want to add two arrays of 10 numbers, `c[i] = a[i] + b[i]`. On a CPU you write a loop. In CUDA you write the *body* of the loop once, as a function called a **kernel**, and ask the GPU to run many copies of it at the same time. Each running copy is a **thread**.

Threads come in groups called **blocks**, all of the same size. Say we pick blocks of 4 threads. Ten elements need $\lceil 10/4 \rceil = 3$ blocks, which gives 12 threads. The whole set of blocks for one launch is the **grid**.

| Block (`blockIdx.x`) | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Thread in block (`threadIdx.x`) | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3
| Global index `i` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11
| Has an element? | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | no

~~~dg-fig 31.01-index.svg
Three blocks of four threads cover a ten-element array. Each thread computes its global index from its block number and its position in the block; the two hatched threads at the end have no element and must do nothing.
~~~

Read the table left to right: a thread's global index is "how many threads come before my block" plus "my position inside the block", so `i = blockIdx.x * 4 + threadIdx.x`. Thread 2 of block 1 gets $1 \times 4 + 2 = 6$ and adds `a[6] + b[6]`. Threads 10 and 11 exist because blocks come in whole units, so they must check `i < 10` and skip.

## Step 1: the execution hierarchy

There are four levels, from the unit you write to the unit the hardware schedules.

~~~dg-layers The CUDA execution hierarchy, from the unit you write to the unit the hardware schedules.
Grid | all blocks of one launch; blocks run in any order, on any SM
*Block | up to 1024 threads on one SM; shared memory and `__syncthreads()`
Warp | 32 consecutive threads of a block, issued together (SIMT)
Thread | its own registers and indices; runs the kernel body once
~~~

- A **warp** is 32 consecutive threads of a block; the SM issues one instruction for all 32 at once (SIMT). If threads of a warp take different sides of an `if`, the warp runs both paths with the inactive lanes switched off: **divergence**.
- A **block** runs entirely on one SM, so its threads can share fast on-chip memory and wait for each other with `__syncthreads()`.
- Blocks of a **grid** are independent: any order, any SM, no waiting on each other within a launch.

You write the kernel from one thread's point of view. The hardware assigns each block to a **streaming multiprocessor** (SM); an A100 has 108 SMs and an H100 SXM has 132 (spec sheet values). A block stays on its SM until all its threads finish. Because blocks are independent, the same kernel runs unchanged on a small GPU (blocks run in more rounds) and on a 132-SM part; the cost is that global synchronisation needs a new kernel launch.

## Step 2: the four built-in variables

Each thread can read four built-in values, each with `.x`, `.y`, `.z`.

| Variable | Meaning | Range in one dimension
| --- | --- | --- |
| `threadIdx` | position of this thread inside its block | 0 to `blockDim - 1`
| `blockIdx` | position of this block inside the grid | 0 to `gridDim - 1`
| `blockDim` | number of threads per block (the same for every block) | set at launch
| `gridDim` | number of blocks in the grid | set at launch

The global 1-D index is

$$i = \texttt{blockIdx.x} \times \texttt{blockDim.x} + \texttt{threadIdx.x}.$$

The first term counts all threads in earlier blocks; the second is the offset inside this block. Number check: with `blockDim.x = 256`, thread 17 of block 3 gets $3 \times 256 + 17 = 785$.

The full vector-add kernel and its launch look like this:

```cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];          // bounds check
}
// host side
int B = 256;                                // threads per block
int G = (n + B - 1) / B;                    // blocks: ceiling division
add<<<G, B>>>(d_a, d_b, d_c, n);
cudaError_t e = cudaGetLastError();         // launch-configuration errors
cudaDeviceSynchronize();                    // wait; surfaces execution errors
```

`__global__` marks a GPU function launched from the CPU; `<<<G, B>>>` is the **launch configuration**: grid size, then block size.

## Step 3: size the grid, then guard every access

You choose the block size $B$; the grid size $G$ follows from the problem size $n$. You need at least $n$ threads, so

$$G = \left\lceil \frac{n}{B} \right\rceil = \frac{n + B - 1}{B} \quad \text{(integer division)}.$$

Adding $B - 1$ before the integer division rounds up. Number check: $n = 10$, $B = 4$ gives $(10 + 3)/4 = 13/4 = 3$ blocks, matching the picture. Plain `n / B` would give 2 blocks and silently skip elements 8 and 9.

Rounding up means the grid usually has $G \cdot B - n$ spare threads, all in the last block. Those threads compute an index past the end of the array, so the `if (i < n)` test is not optional.

<div class="ex">
<div class="lab">Worked example</div>

**counting idle threads**

Take $n = 1{,}000{,}000$ elements and $B = 256$ threads per block.

Blocks: $(1{,}000{,}000 + 255) / 256 = 1{,}000{,}255 / 256 = 3906.8$, which integer division truncates to $3907$. Check: $3906 \times 256 = 999{,}936$ is too few, $3907 \times 256 = 1{,}000{,}192$ is enough.

Threads launched: $3907 \times 256 = 1{,}000{,}192$. Spare threads: $1{,}000{,}192 - 1{,}000{,}000 = 192$, all in block 3906, whose threads cover indices $999{,}936$ to $1{,}000{,}191$.

Of block 3906's 256 threads, $1{,}000{,}000 - 999{,}936 = 64$ do work and 192 fail the bounds check. That is 2 full warps busy and 6 warps that exit immediately.

The waste, $192 / 1{,}000{,}192 \approx 0.02\%$, is negligible. The danger is the writes: without the check those 192 threads write past the end of `c`, which may silently corrupt a neighbouring allocation rather than crash; `compute-sanitizer` reports it.

</div>

## Step 4: two dimensions for matrices

Blocks and grids can be 2-D (or 3-D) using the `dim3` type. For a row-major matrix with `cols` columns, element (row, col) lives at offset `row * cols + col`. Map `x` to columns and `y` to rows:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;   // fastest-varying -> contiguous
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < rows && col < cols) C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
// launch: dim3 block(32, 8); dim3 grid((cols + 31) / 32, (rows + 7) / 8);
```

Each axis gets its own ceiling division, and the bounds check tests both axes.

Which threads form a warp in a 2-D block? CUDA flattens the block with `x` varying fastest: the linear thread id is `threadIdx.x + threadIdx.y * blockDim.x` (plus a `z` term for 3-D), and warp $w$ holds linear ids $32w$ to $32w + 31$. With `block(32, 8)`, every warp is exactly one row of the block: 32 threads with the same `row` and consecutive `col`. So with `x` on columns a warp reads 32 neighbouring floats (128 bytes); swap the two lines and it reads 32 rows of one column, `cols` floats apart.

~~~dg-fig 31.01-mapping.svg
Byte offsets read by one warp in a row-major float matrix with 4096 columns. With x on columns the 32 reads fall in one 128-byte segment; with x on rows they are 16 KiB apart and hit 32 separate segments. Counted from the addresses, not measured on a GPU. Same answer, 32 times the segments; 31.02 calls this coalescing.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one thread in a 2-D launch**

Matrix: 1000 rows by 3000 columns, `block(32, 8)`. Grid: `x` gets $(3000 + 31)/32 = 94$ blocks (covering 3008 columns) and `y` gets $(1000 + 7)/8 = 125$ blocks (exactly 1000 rows).

Total threads: $94 \times 125 \times 256 = 3{,}008{,}000$ for $3{,}000{,}000$ elements, so 8000 are idle: the 8 extra columns of every one of the 1000 rows.

Take the thread with `blockIdx = (2, 1)` and `threadIdx = (5, 3)`. Then $\text{col} = 2 \times 32 + 5 = 69$ and $\text{row} = 1 \times 8 + 3 = 11$. It handles offset $11 \times 3000 + 69 = 33{,}069$.

Its linear id in the block is $5 + 3 \times 32 = 101$, so it is lane $101 \bmod 32 = 5$ of warp $\lfloor 101/32 \rfloor = 3$. Warp 3 is block row 3, which covers matrix row 11, columns 64 to 95: contiguous memory.

</div>

## Step 5: choosing a block size

The block size is yours to pick, within hardware limits. These are the ones that matter on recent data-centre GPUs (A100 and H100; check the CUDA Programming Guide's table for other parts).

| Constraint | Value on recent data-centre GPUs | Consequence
| --- | --- | --- |
| Threads per block | at most 1024 | `dim3(32, 32)` is the largest square block; `blockDim.z` is at most 64
| Warp size | 32 | use multiples of 32, or the last warp is partly idle
| Threads per SM | 2048 | blocks of 256 allow 8 resident blocks per SM
| Blocks per SM | 32 | very small blocks (32 threads) cap at 1024 threads per SM
| Grid size (x) | up to $2^{31}-1$ blocks | 1-D grids are effectively unlimited; y and z are limited to 65535

An SM holds at most 2048 threads *and* 32 blocks, so it fits $\min(\lfloor 2048 / B \rfloor, 32)$ blocks of size $B$.

~~~dg-fig 31.01-resident.svg
Resident threads per SM for every multiple of 32, counting only the thread and block limits above. Tiny blocks hit the 32-block cap; sizes that do not divide 2048 (640, 768) leave slots empty. Registers and shared memory can lower these further (31.07).
~~~

Some readings from the figure:

- $B = 32$: 32 blocks fit, but that is only $32 \times 32 = 1024$ threads, half the SM.
- $B = 256$: $\lfloor 2048/256 \rfloor = 8$ blocks, 2048 threads, a full SM.
- $B = 768$: only $\lfloor 2048/768 \rfloor = 2$ blocks fit, which is 1536 threads; a third block would need 2304.
- $B = 100$ needs 4 warps (128 lanes), so 22% of lanes always idle.

Start with 128 or 256 threads per block for elementwise kernels, then tune. Registers and shared memory per block also limit **occupancy** (31.07). Consumer parts of compute capability 8.6 and 8.9 allow only 1536 threads per SM.

## Step 6: grid-stride loops

"One thread per element" ties the launch to the input size. A **grid-stride loop** instead launches a fixed number of threads, typically a small multiple of the SM count, and has each thread process elements `i, i + stride, i + 2*stride, ...` where the stride is the total number of threads in the grid:

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = alpha * x[i] + y[i];
```

~~~dg-fig 31.01-stride.svg
A grid-stride loop with 2 blocks of 4 threads (stride 8) over 20 elements. Each pass of the loop covers 8 consecutive elements, so neighbouring threads still touch neighbouring memory; thread T3 visits 3, 11 and 19.
~~~

Any `n` works with any grid (the loop condition is the bounds check), per-thread setup is amortised over several elements, `<<<1, 1>>>` runs the kernel serially for debugging, and neighbouring threads still touch neighbouring elements in every pass.

For example, $132 \times 4 = 528$ blocks of 256 give 135,168 threads; for $n = 10^6$ each thread does 7 or 8 iterations.

## Simulating a launch in code

The CUDA snippets need an NVIDIA GPU. To test indexing rules anywhere, simulate the launch serially in Python (a real GPU runs the threads in parallel, in no fixed order).

```python
import numpy as np

def launch(grid, block, kernel, *args):
    # run kernel once per (block, thread), serially: what <<<grid, block>>> means
    for b in range(grid):
        for t in range(block):
            kernel(b, t, block, grid, *args)

def add(bIdx, tIdx, bDim, gDim, a, b, c, n, stats):
    i = bIdx * bDim + tIdx
    if i < n:
        c[i] = a[i] + b[i]
    else:
        stats["idle"] += 1

n, B = 1000, 256
G = (n + B - 1) // B
a, b, c = np.arange(n, dtype=np.float32), np.ones(n, np.float32), np.zeros(n, np.float32)
stats = {"idle": 0}
launch(G, B, add, a, b, c, n, stats)
print(G, G * B, stats["idle"], np.array_equal(c, a + b))   # 4 1024 24 True

def saxpy_gs(bIdx, tIdx, bDim, gDim, alpha, x, y, n, visits):
    i = bIdx * bDim + tIdx
    while i < n:                       # grid-stride loop
        y[i] = alpha * x[i] + y[i]
        visits[i] += 1
        i += gDim * bDim
x, y = np.ones(n, np.float32), np.zeros(n, np.float32)
visits = np.zeros(n, int)
launch(2, 128, saxpy_gs, 2.0, x, y, n, visits)
print(visits.min(), visits.max(), y[:3])               # 1 1 [2. 2. 2.]
```

The first launch has 24 idle threads; the grid-stride launch uses only 256 threads yet visits every element exactly once. Assert "exactly once" whenever you change an indexing scheme.

## Errors and timing

Kernel launches return immediately. A bad configuration (1025 threads per block, too much shared memory) means the kernel never starts, and `cudaGetLastError()` reports it. A fault during execution (an illegal address) appears at the *next* synchronising call, often far from the culprit. In development, wrap every runtime call in a checking macro and synchronise after each launch; `CUDA_LAUNCH_BLOCKING=1` and `compute-sanitizer` help.

For timing, use `cudaEvent` pairs recorded on the same stream and read with `cudaEventElapsedTime`. CPU timers around an asynchronous launch only measure the launch overhead (a few microseconds), not the kernel.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| No bounds check | wrong results or corruption only for some sizes; `compute-sanitizer` reports invalid writes | `if (i < n)` or a grid-stride loop
| `G = n / B` instead of ceiling division | last few elements never written (often stay 0) | `(n + B - 1) / B`
| `threadIdx.x` mapped to rows of a row-major matrix | correct results, several times slower | put `x` on the contiguous (column) axis
| Only `row < rows` checked in 2-D | out-of-bounds writes when `cols` is not a multiple of `blockDim.x` | check both axes
| Block of 1025 threads, or too much shared memory | kernel silently does nothing | `cudaGetLastError()` after every launch
| Timing with a CPU clock and no sync | kernel "takes" 5 microseconds whatever its size | `cudaEvent` timing, or synchronise before stopping the clock

## Going deeper

**Waves and the tail effect.** Blocks are dispatched in **waves**: if each SM holds 8 blocks of 256 threads, an H100 with 132 SMs runs $132 \times 8 = 1056$ blocks at a time. A grid of 1096 blocks needs two waves, and the second has only 40 blocks.

~~~dg-fig 31.01-waves.svg
An idealised model in which every block takes the same time: the fraction of SM slots doing useful work across all waves. Just past a full wave the average drops to about half. Computed from the spec-sheet SM count, not measured.
~~~

Real blocks finish at different times, so the effect is softer, but it matters for big-tile kernels such as GEMM, and is one more argument for grids sized to the hardware.

**Index overflow.** `blockIdx.x * blockDim.x` is 32-bit arithmetic and `int i` holds at most $2^{31} - 1$; a 50k by 50k matrix has 2.5 billion elements. Cast before multiplying: `size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;`.

**Warps are not lockstep any more.** Since Volta each thread has its own program counter, so code that assumed lockstep lanes can break. Use `__syncwarp()` or the `*_sync` intrinsics whenever lanes exchange data.

**Beyond blocks.** Hopper adds **thread block clusters** (blocks co-scheduled on neighbouring SMs that can read each other's shared memory), and cooperative launches allow a grid-wide barrier if the whole grid fits at once. This module does not use them.

**How Triton maps onto this.** A Triton program instance corresponds to one CUDA block, `tl.program_id(0)` to `blockIdx.x`, and `num_warps` to `blockDim.x / 32`. The `offsets = pid * BLOCK + tl.arange(0, BLOCK)` line with a `mask` is exactly the global index plus bounds check from Step 2, written for a whole block at once.

## Key terms

| Term | Meaning
| --- | --- |
| Kernel | A `__global__` function run by every thread of a launch
| Thread | One running copy of the kernel, with its own registers and indices
| Warp | 32 consecutive threads of a block issued together by the SM
| Block | Up to 1024 threads on one SM that can share memory and synchronise
| Grid | All blocks of one launch; blocks are independent and run in any order
| Launch configuration | The `<<<grid, block>>>` sizes, chosen by the host code
| Bounds check | The `if (i < n)` test that stops spare threads from touching memory
| Grid-stride loop | A loop in which each thread steps through the data by the total thread count
| Wave | The set of blocks that fit on all SMs at once

<div class="take">
<div class="lab">Key takeaway</div>

write the kernel for one thread, derive its element from block and thread indices with the contiguous axis on `threadIdx.x`, size the grid by ceiling division, guard every access, and check errors explicitly because launches are asynchronous.

</div>
