---
id: "13.02"
course: "inference-eng-course"
module: "13"
slug: "triton-programming-model"
title: "Triton Programming Model"
keyIdeas: ["Triton is a Python-embedded language for writing GPU kernels on blocks of data.", "Each program instance handles one tile; you index tiles with tl.program_id.", "tl.load / tl.store with masks handle bounds; tl.dot uses tensor cores.", "The compiler handles thread layout, coalescing and shared memory.", "Autotuning picks block sizes and warps for your shapes."]
question: "How does Triton's programming model differ from CUDA's, and what does that save you?"
answer: "In CUDA you write code per thread and manage thread indexing, shared memory and synchronization yourself. In Triton you write code per program instance operating on blocks (tiles) of data with tensor-like operations, and the compiler handles thread mapping, coalescing, shared memory and much of the tiling detail. You keep control of tile sizes and data layout while writing far less low-level code."
sources: [{"title": "Triton tutorials", "url": "https://triton-lang.org/main/getting-started/tutorials/index.html", "note": "Official step-by-step kernels."}, {"title": "Triton: an intermediate language and compiler for tiled neural network computations", "url": "https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf", "note": "The original paper."}]
tags: ["triton"]
---

## Why Triton

Writing CUDA C++ gives full control, but you must manage every thread, every shared memory buffer and every synchronization. Triton raises the level of abstraction: you write code that operates on blocks of data, and the compiler decides how to map them onto threads, registers and shared memory. For many inference kernels, Triton reaches 80 to 100% of hand-written CUDA performance with far less code. torch.compile generates Triton kernels, and vLLM and SGLang use Triton for several operations.

## The programming model

A Triton kernel is a Python function decorated with `@triton.jit`. You launch a grid of **program instances**. Each instance processes one block of data using vector operations on whole blocks at once.

```
import triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                     # which block am I?
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # the indices I handle
    mask = offs < n                            # guard the last block
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out
```

Key points:

- `tl.program_id(0)` plays the role of a block index.
- `tl.arange(0, BLOCK)` creates a vector of offsets. Every operation on it is a vector operation.
- `mask` handles the last, partial block, instead of an `if` per thread.
- `BLOCK` is a `tl.constexpr`, known at compile time, so the compiler can specialize the code.

## CUDA and Triton compared

| Aspect | CUDA | Triton |
| --- | --- | --- |
| Unit of code | one thread | one block of data |
| Shared memory | allocated and synchronized by you | managed by the compiler |
| Coalescing | your responsibility | mostly automatic for contiguous blocks |
| Tensor cores | intrinsics or libraries such as CUTLASS | `tl.dot` on blocks |
| Peak performance | the highest ceiling | close for many kernels |
| Iteration speed | slow compile, more code | fast, Python-based |

Triton is weaker where you need fine control: warp-specialized pipelines, very specific memory layouts, or the newest hardware features before the compiler supports them. FlashAttention-3 and the fastest FP8 matmuls are still written in CUDA or with CUTLASS.

## Autotuning

The best block size and number of warps depend on the input shape and GPU. Triton can try several configurations and keep the fastest:

```
@triton.autotune(
    configs=[triton.Config({"BLOCK": 512}, num_warps=4),
             triton.Config({"BLOCK": 1024}, num_warps=8),
             triton.Config({"BLOCK": 2048}, num_warps=8)],
    key=["n"],
)
@triton.jit
def add_kernel(...): ...
```

On the first call for each distinct value of the `key` arguments, Triton benchmarks every configuration and caches the winner. In a server, this means the first request of each new shape is slow. Engines either warm up all expected shapes at startup or ship pre-tuned configurations.

<div class="ex">
<div class="lab">Worked example</div>

**what autotuning finds**

A fused RMSNorm kernel for hidden size 4,096 was tested with four configurations at 1 and 4,096 rows.

At 1 row, the best was a single program with 4 warps: about 4 µs, dominated by launch cost.

At 4,096 rows, the best was 8 warps per program with one row per program: about 25 µs, about 85% of peak bandwidth. The 4-warp setting reached only 70%.

No single configuration was best for both. That is why the key includes the shape.

</div>

## How Triton compiles

The `@triton.jit` function is compiled on first use to an intermediate representation, optimized (including choices about shared memory, vector widths and software pipelining) and then lowered to PTX for NVIDIA or to the equivalent for AMD. Compiled kernels are cached on disk, so later runs start faster. The same Triton source can often run on both NVIDIA and AMD GPUs.

## Deep dive: how a Triton grid maps to the GPU

A Triton kernel runs one program per block of work. For a vector of ⟦m721⟧ elements in blocks of size ⟦m722⟧:

⟦M719⟧

Each program is compiled into one CUDA thread block of `num_warps` × 32 threads. A larger ⟦m723⟧ means fewer programs, more registers per program and more work per thread. Autotuning searches the product of choices:

⟦M720⟧

Four block sizes, three warp counts and three stage counts is 36 compilations, each benchmarked. That is why autotuning results are cached per shape.

<div class="take">
<div class="lab">Key takeaway</div>

Triton lets you write tiled GPU kernels in Python, operating on blocks instead of threads, and leaves coalescing and shared memory to the compiler. With autotuning per shape, it is the most practical path into kernel work.

</div>
