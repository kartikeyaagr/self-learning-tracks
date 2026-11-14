---
id: "15.02"
course: "inference-eng-course"
module: "15"
slug: "torch-compile-kernel-fusion"
title: "torch.compile & Kernel Fusion"
keyIdeas: ["torch.compile captures graphs with TorchDynamo and generates kernels with TorchInductor.", "Inductor fuses pointwise and reduction ops into Triton kernels.", "Matmuls go to cuBLAS or generated templates; modes like max-autotune search for faster ones.", "First call compiles (slow); later calls reuse compiled code.", "Engines like vLLM integrate torch.compile for parts of the model."]
question: "What does torch.compile do to a model's forward pass at a high level?"
answer: "TorchDynamo captures the Python code into an FX graph of tensor operations (falling back to Python where it cannot), AOTAutograd prepares the graph, and TorchInductor lowers it to optimized code, fusing elementwise and reduction ops into generated Triton kernels and calling libraries like cuBLAS for matmuls. The result runs fewer, larger kernels with less Python overhead."
sources: [{"title": "PyTorch: Introduction to torch.compile", "url": "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html", "note": "Usage and concepts."}, {"title": "PyTorch 2 paper (Dynamo and Inductor)", "url": "https://dl.acm.org/doi/10.1145/3620665.3640366", "note": "Design of the compiler stack."}]
tags: ["compile"]
---

## What torch.compile does

`torch.compile` turns a PyTorch model into optimized code without changing the model's source. It captures the model's operations as a graph, optimizes the graph, and generates fused kernels, mostly in Triton.

~~~dg-flow The torch.compile pipeline. Press Play to follow a model through it.
Python model -(TorchDynamo)-> FX graph -(AOTAutograd)-> *TorchInductor -> Triton kernels and library calls
~~~

- **TorchDynamo** watches Python execute and records the tensor operations into an FX graph. It inserts guards: checks that must hold for the compiled code to be reused, such as input shapes and dtypes.
- **AOTAutograd** traces through PyTorch's internals to a lower-level graph of primitive operations. For inference it mostly passes the graph through.
- **TorchInductor** is the compiler backend. It decides which operations to fuse, generates Triton kernels for them, and calls libraries such as cuBLAS for large matmuls.

## Using it

```
model = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=False)
```

The main modes:

- **default**: fuses operations with moderate compile time.
- **reduce-overhead**: adds CUDA graphs on top, to cut launch overhead.
- **max-autotune**: benchmarks several kernel configurations, including Triton matmul templates, and keeps the fastest. Much longer compile time.

Engines usually call torch.compile on the model body themselves and manage CUDA graphs separately, which is why `-no-cudagraphs` variants exist.

## What fusion gives

Chains of element-wise and reduction operations become single kernels:

| Chain | Eager kernels | Compiled kernels |
| --- | --- | --- |
| residual add + RMSNorm + gain | about 5 | 1 |
| SiLU(gate) × up | 3 | 1 |
| RoPE on Q and K | several | 1 or 2 |
| dtype casts around operations | 1 each | usually absorbed |

Each fusion removes a write and a read of an intermediate tensor in HBM, and removes launches.

<div class="ex">
<div class="lab">Worked example</div>

**the effect of compiling a decode step**

An 8B model at batch 8, eager: about 650 kernels and 9.1 ms per step.

With torch.compile (no CUDA graphs): about 300 kernels and 7.4 ms per step, about 19% faster. Most of the gain is from fewer launches; some is from fused element-wise work.

With compile plus CUDA graphs: 5.9 ms, close to the weight-read limit. Fusion and graphs together get most of the way to the hardware bound.

</div>

## What it does not do

- It does not rewrite large matmuls, which already run in cuBLAS or CUTLASS. Those are typically most of the step's GPU time.
- It does not replace specialized kernels such as FlashAttention or paged attention. Engines register those as custom operators that the compiler treats as opaque calls.
- It does not change the algorithm. Quantization, batching and caching are separate decisions.

## The costs

- **Compile time**: from seconds to minutes at startup, more with max-autotune. Engines cache compiled artifacts on disk to avoid repeating this.
- **Recompiles**: when an input shape or a code path changes in a way the guards did not allow, Dynamo compiles again, which can cause a multi-second stall in the middle of serving. The next lesson covers how to avoid this.
- **Debuggability**: errors can come from generated code. `TORCH_LOGS=output_code` prints the generated kernels, and `TORCH_LOGS=recompiles` reports why recompiles happen.

## Deep dive: what one fusion saves

Fusing a chain of ⟦m805⟧ element-wise ops over a tensor of ⟦m806⟧ bytes replaces ⟦m807⟧ reads and ⟦m808⟧ writes with one of each, and ⟦m809⟧ launches with one:

⟦M804⟧

<div class="ex">
<div class="lab">Worked example</div>

**the SiLU-multiply in the MLP at batch 1**

Unfused: SiLU on the gate output, then multiply by the up output. Two kernels on 14,336-element vectors (28 KB each).

Bytes saved: about 57 KB, 0.02 µs at 3 TB/s. Launches saved: one, about 4 µs.

At decode sizes the saving is almost all launch overhead. At prefill sizes (⟦m810⟧, 235 MB per tensor) the bytes matter: about 160 µs saved per layer.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

torch.compile captures the model as a graph and generates fused kernels, cutting kernel count and Python overhead. It leaves large matmuls and attention to specialized libraries, and it costs compile time and the risk of recompiles.

</div>
