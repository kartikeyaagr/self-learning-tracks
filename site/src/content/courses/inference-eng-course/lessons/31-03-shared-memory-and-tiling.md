---
id: "31.03"
course: "inference-eng-course"
module: "31"
slug: "shared-memory-and-tiling"
title: "Shared memory and tiling"
keyIdeas: ["**Shared memory** is on-chip memory per SM (up to ~164 KB on A100, ~228 KB on H100), allocated per block and visible to all its threads; it is a programmer-managed cache with far lower latency than HBM.", "**Tiling** loads a block of data once from global memory into shared memory and reuses it many times. In a $T \\times T$ tiled matmul each global element is reused $T$ times, raising arithmetic intensity from 0.25 to $T/4$ FLOP per byte (FP32).", "The tiled loop needs **two** `__syncthreads()` per phase: one after loading (so no thread reads a tile before it is complete) and one after computing (so no thread overwrites a tile others still read).", "Shared memory has 32 **banks** of 4-byte words; bank $= (\\text{address}/4) \\bmod 32$. Threads of a warp hitting different addresses in the same bank serialise; padding a $32 \\times 32$ tile to $32 \\times 33$ removes the classic column-access conflict.", "Tiling in shared memory is the first step; fast GEMMs add **register tiling** (each thread computes a small block of outputs) and Tensor Cores, which is why you call cuBLAS rather than writing your own."]
question: "You tile a matrix transpose with 32 x 32 blocks: each thread reads `in[y][x]` into `tile[ty][tx]`, synchronises, then writes `out[x'][y'] = tile[tx][ty]` with the output indices swapped so the global write is coalesced. It is faster than the naive transpose but still well below copy bandwidth. Why, and what one-line change fixes it?"
answer: "The global reads and writes are now coalesced, but the shared-memory read `tile[tx][ty]` is a column access: the 32 threads of a warp have consecutive `tx`, so they read 32 words that are each a full row (32 floats) apart. All 32 words map to the same bank, so the warp's access is replayed 32 times, serialising every shared-memory read. Declaring the tile as `__shared__ float tile[32][33]` pads each row by one word, shifting successive rows by one bank, so a column read touches 32 different banks and completes in a single pass. With the padding, the transpose typically runs close to the speed of a simple device-to-device copy."
sources: [{"title": "NVIDIA blog: Using Shared Memory in CUDA C/C++", "url": "https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/", "note": "Shared memory declaration, synchronisation and bank conflicts with a worked example."}, {"title": "NVIDIA blog: An Efficient Matrix Transpose in CUDA C/C++", "url": "https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/", "note": "The coalesced transpose and the one-column padding trick, with measured bandwidths."}, {"title": "Simon Boehm, How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance", "url": "https://siboehm.com/articles/22/CUDA-MMM", "note": "Step-by-step from naive to tiled to register-tiled kernels, each step benchmarked."}]
practice: ["tiled-matrix-multiplication", "shared-memory-bank-conflict-analyzer", "matrix-multiplication", "tiled-transpose"]
tags: ["shared memory", "tiling", "bank conflicts"]
tools: ["CUDA"]
---

## The reuse problem

Matrix multiply $C = AB$ with $A$ of size $M \times K$ and $B$ of size $K \times N$ does $2MNK$ floating-point operations on only $MK + KN$ inputs. Every element of $A$ is used $N$ times and every element of $B$ $M$ times. The question is where that reuse happens. In the naive kernel, each thread computes one $C_{ij}$ by reading row $i$ of $A$ and column $j$ of $B$ straight from global memory:

```cpp
float acc = 0.f;
for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
C[row * N + col] = acc;
```

Each iteration does 2 FLOPs and loads 8 bytes: 0.25 FLOP/byte. On an A100 (19.5 TFLOPS FP32, about 2 TB/s), that intensity caps the kernel near 0.5 TFLOPS if every load went to HBM. Caches recover some of it, but the naive kernel typically reaches a few percent of cuBLAS.

## Tiled matmul

Split $C$ into $T \times T$ tiles, one per block of $T \times T$ threads. To compute its tile, the block walks along $K$ in steps of $T$; in each phase it cooperatively loads one $T \times T$ tile of $A$ and one of $B$ into shared memory, then every thread does $T$ multiply-adds from shared memory.

~~~dg-flow One phase of the tiled loop; the block repeats it K/T times.
Load A and B tiles (coalesced) -(__syncthreads)-> *Multiply from shared memory -(__syncthreads)-> Next phase
~~~

```cpp
#define T 32
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[T][T], Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y, col = blockIdx.x * T + threadIdx.x;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[threadIdx.y][threadIdx.x] = (row < M && k0 + threadIdx.x < K) ? A[row * K + k0 + threadIdx.x] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (k0 + threadIdx.y < K && col < N) ? B[(k0 + threadIdx.y) * N + col] : 0.f;
        __syncthreads();                       // tile fully loaded
        for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                       // done reading before overwrite
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```

Per phase, the block loads $2T^2$ floats ($8T^2$ bytes) and performs $2T^3$ FLOPs, so global intensity is $T/4$ FLOP/byte: 8 for $T = 32$, a 32x improvement. Both loads are coalesced, because `threadIdx.x` walks along a row of each tile. Out-of-range elements are filled with zeros so that matrices whose sizes are not multiples of $T$ work unchanged.

The two barriers are not optional. Remove the first and a fast warp reads tile entries that a slow warp has not yet written. Remove the second and a fast warp starts loading the next phase's tile over values a slow warp is still multiplying. Both bugs give wrong answers only sometimes, which makes them painful to find; `compute-sanitizer --tool racecheck` detects them. A barrier must also be reached by all threads of the block: never place `__syncthreads()` inside a branch that only some threads take.

## Bank conflicts

Shared memory is divided into 32 banks; successive 4-byte words go to successive banks. A warp's shared-memory access completes in one pass if its 32 threads hit 32 different banks, or the same word (broadcast). If $k$ threads hit different words in one bank, the access is replayed $k$ times.

| Access by a warp (float tile `s[32][32]`) | Banks touched | Cost
| --- | --- | --- |
| Row: `s[r][tid]` | 32 distinct | 1 pass
| Same element: `s[r][k]` for all threads | 1 word, broadcast | 1 pass
| Column: `s[tid][c]` | all in bank `c` | 32 passes (32-way conflict)
| Column on padded `s[32][33]`: `s[tid][c]` | 32 distinct | 1 pass

In the tiled matmul above, a warp is one row of the block (fixed `threadIdx.y`, varying `threadIdx.x`). `As[threadIdx.y][k]` is the same word for all 32 threads, a broadcast; `Bs[k][threadIdx.x]` is a row access. No conflicts. The matrix transpose is where conflicts bite: read a tile row-wise, write it out column-wise, and the column read from shared memory is a 32-way conflict unless the tile is declared `[32][33]`. The extra column shifts each row by one bank.

<div class="ex">
<div class="lab">Worked example</div>

**how far tiling gets you**

On a 4096 x 4096 FP32 matmul, the naive kernel reaches roughly 1% of cuBLAS throughput on a recent GPU, and shared-memory tiling gets to roughly 10%. The rest comes from register tiling (each thread computing a 4 x 4 or 8 x 8 output block so values loaded from shared memory are reused in registers), wider vectorised loads, double buffering that overlaps the next tile's load with the current tile's math, and, for FP16/BF16, Tensor Core instructions. Simon Boehm's step-by-step write-up measures each stage on a real GPU.

</div>

## Limits and trade-offs

Shared memory per block defaults to 48 KB for static arrays; larger allocations need dynamic shared memory and an explicit opt-in attribute. Every byte a block uses reduces how many blocks fit on an SM, which lowers occupancy (31.07). Two 32 x 32 FP32 tiles use 8 KB, small. A double-buffered 128 x 32 BF16 tiling uses much more and is designed with occupancy in mind.

<div class="take">
<div class="lab">Key takeaway</div>

tile so that each byte fetched from global memory is reused many times from shared memory, fence each phase with two barriers, and lay out shared arrays so a warp never hits one bank with different addresses.

</div>
