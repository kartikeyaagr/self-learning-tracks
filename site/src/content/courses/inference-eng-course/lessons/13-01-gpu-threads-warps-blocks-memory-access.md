---
id: "13.01"
course: "inference-eng-course"
module: "13"
slug: "gpu-threads-warps-blocks-memory-access"
title: "GPU Threads, Warps, Blocks & Memory Access"
keyIdeas: ["A kernel runs a grid of thread blocks; each block runs on one SM; threads execute in warps of 32.", "Threads in a block share fast shared memory and can synchronize.", "Coalesced access: adjacent threads touch adjacent addresses, combined into few transactions.", "Occupancy: enough active warps per SM to hide memory latency.", "Bank conflicts in shared memory and branch divergence within a warp slow kernels."]
question: "Why does it matter that adjacent threads in a warp read adjacent memory addresses?"
answer: "The hardware combines a warp's memory requests into as few memory transactions as possible. If 32 threads read 32 consecutive 4-byte values, that is one or a few 128-byte transactions (coalesced). If they read scattered addresses, the warp needs many transactions, wasting bandwidth, which is the main limit for memory-bound inference kernels."
sources: [{"title": "CUDA C++ Best Practices Guide", "url": "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/", "note": "Coalescing and memory optimization."}, {"title": "Modal GPU Glossary", "url": "https://modal.com/gpu-glossary", "note": "Threads, warps, blocks and SMs."}]
tags: ["cuda", "gpu"]
---

## The execution hierarchy

Every GPU kernel runs as a hierarchy of threads. Knowing it is the first step to writing kernels or reading why one is slow.

~~~dg-layers The CUDA execution hierarchy, from a whole launch down to one thread.
Grid | all thread blocks of one kernel launch
*Thread block | runs on one SM, shares that SM's shared memory, can synchronize
Warp | 32 threads executing in lockstep
Thread | its own registers and program counter
~~~

- A **grid** can contain thousands of thread blocks. The hardware schedules them onto SMs as space allows.
- A **thread block** of, say, 128 to 512 threads stays on one SM for its whole life. Its threads can share data through shared memory and wait for each other at barriers.
- A **warp** is the unit the SM actually schedules. Its 32 threads execute the same instruction at the same time.
- Each **thread** has private registers, the fastest storage on the GPU.

## Coalesced memory access

When the 32 threads of a warp load from global memory, the hardware combines their requests into as few memory transactions as possible. If thread i reads address base + i × 4 bytes, the 32 reads form one contiguous 128-byte span, served by a single wide transaction. This is **coalescing**.

If the threads read scattered addresses, or addresses with a large stride, the hardware needs many separate transactions, and most of the bytes in each one are wasted.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of a strided read**

A [4096, 4096] BF16 matrix is stored row by row. A kernel needs to sum each column.

If each thread handles one column and the warp's 32 threads read 32 neighbouring columns of the same row, their addresses are contiguous: 64 bytes, fully coalesced.

If instead each thread handles one row and reads down its row's elements while the warp's threads are on 32 different rows, each load touches 32 addresses 8 KB apart: 32 separate transactions, and bandwidth drops by roughly an order of magnitude.

Same math, same data. Only the mapping of threads to data changed.

</div>

## Shared memory banks

Shared memory is divided into 32 banks. Each bank can serve one 4-byte word per cycle. If several threads in a warp access different addresses in the same bank, the accesses are serialized: a **bank conflict**. A common example is reading a column of a tile stored row-major with a row length that is a multiple of 32 words. Padding each row by one element, or swizzling the layout, spreads the accesses across banks.

## Hiding latency with occupancy

A load from HBM takes several hundred cycles. An SM hides that wait by switching to another warp that is ready to run. The more warps resident on an SM, the more waiting it can hide. The number of resident warps, relative to the maximum, is called **occupancy**.

Occupancy is limited by the resources each thread block uses:

- **Registers**: an SM has 64K 32-bit registers. A kernel using 128 registers per thread can keep at most 512 threads, 16 warps, resident.
- **Shared memory**: a block that uses 100 KB of shared memory leaves room for only two such blocks per SM.
- **Block size and count limits** set by the hardware.

High occupancy is not always necessary. Well-tuned matmul kernels run at low occupancy and hide latency by issuing many independent loads from each warp. But for simple memory-bound kernels, low occupancy usually means poor bandwidth.

## Divergence

If threads in one warp take different branches of an `if`, the warp runs both branches one after the other, with some threads switched off each time. Branches that depend on the thread index within a warp can halve throughput or worse. Arrange work so that threads in a warp take the same path, for example by handling boundary conditions with masks rather than branches.

| Problem | Symptom in a profiler | Typical fix |
| --- | --- | --- |
| Uncoalesced loads | low DRAM efficiency, many transactions per request | change the data layout or the thread-to-data mapping |
| Bank conflicts | shared memory stalls | pad or swizzle the tile layout |
| Low occupancy | many stalls waiting for memory | fewer registers per thread, smaller blocks |
| Divergence | low warp execution efficiency | restructure branches, use masks |
| Too few blocks | many idle SMs | split the work more finely, such as split-KV |

## Deep dive: coalescing in sectors

Global memory is served in 32-byte sectors. When the 32 threads of a warp each read one 2-byte BF16 value, the hardware fetches every sector that any thread touches:

⟦M708⟧

Adjacent threads reading adjacent values touch 2 sectors: 100%. With a stride of ⟦m710⟧ elements, the warp touches ⟦m711⟧ sectors.

~~~dg-raw
%%chart Load efficiency of one warp reading BF16 values at a stride.
x = 1, 2, 4, 8, 16, 32
logx = 1
xlab = Stride between threads (elements)
ylab = Useful share of fetched bytes (%)
Efficiency: 100, 50, 25, 12.5, 6.25, 6.25
%%
~~~

## Deep dive: what limits occupancy

An SM holds at most 64 warps (2,048 threads) on an H100. The real limit is whichever resource runs out first:

⟦M709⟧

with ⟦m712⟧ registers per thread. A kernel using 128 registers per thread can keep only ⟦m713⟧ warps, 25% occupancy. That is fine for a matmul that hides latency with large tiles, and bad for a memory-bound kernel that needs many loads in flight (12.04).

## Deep dive: shared memory bank conflicts

Shared memory has 32 banks, each 4 bytes wide. Thread ⟦m714⟧ reading 4-byte word ⟦m715⟧ uses bank ⟦m716⟧. If ⟦m717⟧ threads in a warp hit different words in the same bank, the access is split into ⟦m718⟧ passes. Reading a column of a 32 × 32 FP32 tile row-major puts every thread in one bank: a 32-way conflict, 32 times slower. Padding each row to 33 words moves consecutive rows to different banks and removes it.

<div class="take">
<div class="lab">Key takeaway</div>

GPUs reward many threads reading contiguous memory. Get coalescing and enough parallelism right first, then look at bank conflicts, occupancy and divergence.

</div>
