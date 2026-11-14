---
id: "31.01"
course: "inference-eng-course"
module: "31"
slug: "threads-blocks-and-grids"
title: "Threads, blocks and grids"
keyIdeas: ["A CUDA kernel is one function run by many threads. Threads are grouped into **blocks** (up to 1024 threads, scheduled together on one SM, able to share memory and synchronise) and blocks form a **grid**; the hardware executes threads in **warps** of 32.", "Each thread computes its own work item from built-in indices: `i = blockIdx.x * blockDim.x + threadIdx.x` in 1-D, and a (row, col) pair in 2-D with `threadIdx.x` mapped to the contiguous dimension.", "The grid is sized by ceiling division, `(n + B - 1) / B` blocks, so the last block usually has spare threads and **every kernel needs a bounds check**.", "A **grid-stride loop** decouples grid size from problem size: launch a few blocks per SM and let each thread walk the array in steps of `gridDim.x * blockDim.x`.", "Launches are asynchronous and do not report errors by themselves; check `cudaGetLastError()` after the launch and synchronise before reading results or timing."]
question: "A colleague's 2-D kernel uses `int row = blockIdx.x * blockDim.x + threadIdx.x; int col = blockIdx.y * blockDim.y + threadIdx.y;` with `dim3 block(32, 8)` on a row-major 4096 x 4096 float matrix. Results are correct but the kernel is about 5x slower than expected. Explain why, and what else could go wrong for other matrix shapes."
answer: "With `x` mapped to rows, the 32 threads of a warp (consecutive `threadIdx.x`) handle 32 different rows of the same column. In row-major storage those elements are 4096 floats apart, so each warp load touches 32 separate memory segments instead of one contiguous 128-byte span, wasting most of every memory transaction. Swapping the mapping so that `threadIdx.x` indexes columns makes each warp read contiguous memory. For other shapes, the grid must also be sized per axis with ceiling division and the bounds check must test both `row < rows` and `col < cols`; with non-square matrices, mixing up which grid dimension covers rows and which covers columns either leaves part of the matrix unprocessed or launches threads outside it. Very tall matrices can also exceed the 65535-block limit on `grid.y` if rows are mapped to y."
sources: [{"title": "CUDA C++ Programming Guide: Programming Model", "url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html", "note": "Chapter on the thread hierarchy, kernels and the execution configuration; the reference for every limit quoted here."}, {"title": "NVIDIA blog: An Even Easier Introduction to CUDA", "url": "https://developer.nvidia.com/blog/even-easier-introduction-cuda/", "note": "First kernel end to end: memory, launch, timing, and the effect of adding blocks."}, {"title": "NVIDIA blog: CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", "url": "https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/", "note": "Short, with the scalability and debugging arguments for grid-stride loops."}]
practice: ["vector-addition", "matrix-addition", "vector-subtraction", "relu"]
tags: ["thread hierarchy", "launch configuration"]
tools: ["CUDA"]
---

## Where this module sits

Earlier modules of this course worked at the Triton level ("Triton & Kernel Engineering") and through profilers ("Profiling & GPU Performance"). Triton hides individual threads: one program instance processes a block of elements and the compiler decides how threads, shared memory and vector loads are used. This module goes one level down, to CUDA C++, where you control each of those decisions. That is the level at which cuBLAS, FlashAttention, vLLM's custom kernels and most PyTorch extensions are written, and reading it makes the behaviour of the higher-level tools much less mysterious. The hardware ideas (warps, SMs, memory hierarchy) were introduced in "GPU Threads, Warps, Blocks & Memory Access"; here they become code.

## The execution hierarchy

~~~dg-layers The CUDA execution hierarchy, from the unit you write to the unit the hardware schedules.
Grid | all blocks of one launch; blocks run in any order, on any SM
*Block | up to 1024 threads on one SM; shared memory and `__syncthreads()`
Warp | 32 consecutive threads of a block, issued together (SIMT)
Thread | its own registers and indices; runs the kernel body once
~~~

You write the kernel from one thread's point of view. The hardware assigns each block to a streaming multiprocessor (an A100 has 108 SMs, an H100 SXM 132), and the block stays there until it finishes. Inside the SM, threads are issued in warps of 32 that execute the same instruction; if threads of a warp take different branches, the warp runs both paths with the inactive threads masked off (divergence). Blocks cannot synchronise with each other within a launch and may run in any order, which is what lets the same kernel scale from a laptop GPU to a data-centre part.

## Indexing and launch configuration

Every thread has `threadIdx` (position within its block), `blockIdx` (position of the block in the grid) and the sizes `blockDim` and `gridDim`, each with `.x`, `.y`, `.z` components.

```cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];          // bounds check
}
// host
int B = 256;
int G = (n + B - 1) / B;                    // ceiling division
add<<<G, B>>>(d_a, d_b, d_c, n);
cudaError_t e = cudaGetLastError();         // launch-configuration errors
cudaDeviceSynchronize();                    // wait; surfaces execution errors
```

For a matrix stored row-major with `cols` columns, a 2-D launch maps `x` to columns and `y` to rows:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;   // fastest-varying -> contiguous
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < rows && col < cols) C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
// launch: dim3 block(32, 8); dim3 grid((cols + 31) / 32, (rows + 7) / 8);
```

Putting `threadIdx.x` on the contiguous axis matters: consecutive threads of a warp then touch consecutive addresses, which is what makes memory access efficient (next lesson). Swapping the two lines gives the same answer and several times less bandwidth.

## Choosing block size

| Constraint | Value on recent data-centre GPUs | Consequence
| --- | --- | --- |
| Threads per block | at most 1024 | `dim3(32, 32)` is the largest square block
| Warp size | 32 | use block sizes that are multiples of 32, or the last warp is partly idle
| Threads per SM | 2048 | blocks of 256 allow 8 resident blocks per SM
| Blocks per SM | 32 | very small blocks (32 threads) cap at 1024 threads per SM
| Grid size (x) | up to $2^{31}-1$ blocks | 1-D grids are effectively unlimited; y and z are limited to 65535

Starting points: 128 or 256 threads per block for elementwise kernels, then tune. The block size also determines how many registers and how much shared memory each block uses, which sets occupancy (31.07).

## Grid-stride loops

A grid with one thread per element ties the launch to the input size. A grid-stride loop launches a fixed number of threads, typically a small multiple of the SM count, and has each thread process elements `i, i + stride, i + 2*stride, ...`:

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = alpha * x[i] + y[i];
```

Benefits: any `n` works with any grid; per-thread setup (loading constants, computing pointers) is amortised over several elements; and the kernel is easy to debug with `<<<1, 1>>>`, which runs it serially. Consecutive threads still touch consecutive elements in every iteration, so memory access stays coalesced.

<div class="ex">
<div class="lab">Worked example</div>

**counting idle threads**

$n = 1{,}000{,}000$ elements with 256-thread blocks needs $\lceil 10^6/256 \rceil = 3907$ blocks, or 1,000,192 threads. The last 192 threads fail the bounds check and do nothing. Without the check they write past the end of `c`, which may silently corrupt a neighbouring allocation rather than crash. Running under `compute-sanitizer` reports such out-of-bounds accesses.

</div>

## Errors and timing

Kernel launches return immediately. A bad configuration (1025 threads per block, too much shared memory) is reported by `cudaGetLastError()`; a fault during execution (illegal address) appears at the next synchronising call, often far from the culprit. In development, wrap every runtime call in a checking macro and synchronise after each launch. For timing, use `cudaEvent` pairs recorded on the same stream, not CPU timers around an asynchronous launch, which only measure launch overhead (a few microseconds).

<div class="take">
<div class="lab">Key takeaway</div>

write the kernel for one thread, derive its element from block and thread indices with the contiguous axis on `threadIdx.x`, size the grid by ceiling division, guard every access, and check errors explicitly because launches are asynchronous.

</div>
