---
id: "00.02"
course: "inference-eng-course"
module: "00"
slug: "vectors-matrices-tensor-shapes"
title: "Vectors, Matrices & Tensor Shapes"
keyIdeas: ["A tensor is an n-dimensional array. In LLMs the common activation shape is [batch, sequence, hidden].", "Matrix multiply [M, K] x [K, N] gives [M, N] and costs 2MKN FLOPs.", "Linear layers in transformers are matmuls where M is the number of tokens in flight.", "Attention adds a head dimension: [batch, heads, seq, head_dim].", "Most bugs and many performance surprises are shape bugs. Print shapes early."]
question: "In a transformer layer with hidden size 4096, a batch of 8 sequences each 512 tokens long enters a linear layer with weight shape [4096, 11008]. What are the input and output shapes, and how many FLOPs does the matmul take?"
answer: "Input is [8, 512, 4096], output is [8, 512, 11008]. FLOPs are 2 x 8 x 512 x 4096 x 11008, about 370 GFLOPs."
sources: [{"title": "PyTorch: Tensors tutorial", "url": "https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html", "note": "Tensor creation, shapes and operations."}, {"title": "Mathematics for Machine Learning (book)", "url": "https://mml-book.github.io/", "note": "Linear algebra background, free PDF."}]
tags: ["math", "tensors"]
---

## Why shapes matter

Every bug in inference code eventually shows up as a shape problem, and every performance question eventually becomes "what are the shapes of the matrix multiplies". Being fluent with tensor shapes is the single most useful math skill for this work.

## Vectors, matrices and tensors

- A **vector** is a list of numbers, shape `[d]`. A token's hidden state in Llama 3 8B is a vector of 4,096 numbers.
- A **matrix** is a 2D grid, shape `[rows, cols]`. A weight matrix that maps 4,096 inputs to 14,336 outputs has shape `[4096, 14336]`.
- A **tensor** is the general n-dimensional version. Activations in a batch usually have shape `[batch, seq, hidden]`.

## Matrix multiply, the workhorse

Multiplying `[M, K]` by `[K, N]` gives `[M, N]`. The inner dimension K must match. It costs **2 × M × K × N** FLOPs and reads at least M×K + K×N inputs and writes M×N outputs.

In an LLM layer, M is the number of tokens being processed this step, K and N come from the weights:

- During **prefill** of a 2,000-token prompt, M = 2,000. The weight matrix is reused 2,000 times per read. Lots of FLOPs per byte.
- During **decode** with batch 8, M = 8. Each weight is read and used only 8 times. Few FLOPs per byte.

<div class="ex">
<div class="lab">Worked example</div>

**one MLP projection in Llama 3 8B**

Weight W_up has shape [4096, 14336] = 58.7M parameters = 117 MB in BF16.

Decode, batch 8: input [8, 4096] × W_up → [8, 14336]. FLOPs = 2 × 8 × 4096 × 14336 ≈ 0.94 GFLOP. Bytes ≈ 117 MB. Intensity ≈ 8 FLOP per byte. Memory-bound.

Prefill, 2,000 tokens: input [2000, 4096] × W_up. FLOPs ≈ 235 GFLOP for the same 117 MB. Intensity ≈ 2,000 FLOP per byte. Compute-bound.

</div>

## Shapes through one transformer layer

For Llama 3 8B (hidden 4,096, 32 query heads, 8 KV heads, head dim 128, MLP width 14,336), with T tokens in flight:

| Step | Input shape | Weight shape | Output shape |
| --- | --- | --- | --- |
| Q projection | [T, 4096] | [4096, 4096] | [T, 32, 128] after reshape |
| K and V projections | [T, 4096] | [4096, 1024] each | [T, 8, 128] each |
| Attention | Q [32, T, 128], K and V [8, S, 128] | none | [T, 32, 128] |
| Output projection | [T, 4096] | [4096, 4096] | [T, 4096] |
| MLP gate and up | [T, 4096] | [4096, 14336] each | [T, 14336] each |
| MLP down | [T, 14336] | [14336, 4096] | [T, 4096] |

S is the total context length the attention looks back over, which for decode is everything generated so far. The K and V projections are narrow (1,024 instead of 4,096) because Llama 3 uses grouped-query attention: 8 KV heads shared by 32 query heads. That makes the KV cache 4× smaller, which you will see again in the memory lessons.

## Reshape, transpose, view

Heads are a reshape. A `[T, 4096]` tensor becomes `[T, 32, 128]`, then is transposed to `[32, T, 128]` so each head's attention runs as an independent batched matrix multiply. A **view** reinterprets memory without copying. A **transpose** often makes the tensor non-contiguous, and some kernels then need a copy (`.contiguous()`), which costs memory bandwidth. Unnecessary copies are a classic hidden cost.

## Broadcasting

Adding a bias of shape `[4096]` to activations `[T, 4096]` works because the bias is broadcast across the T dimension. Broadcasting is convenient but can silently produce the wrong shape when dimensions happen to match by accident. Assert shapes in your own code.

## Deep dive: counting a matmul exactly

For ⟦m49⟧ with ⟦m50⟧ and ⟦m51⟧, each of the ⟦m52⟧ outputs is a dot product of length ⟦m53⟧: ⟦m54⟧ multiplies and ⟦m55⟧ adds, which we round to ⟦m56⟧. So

⟦M48⟧

where ⟦m57⟧ is bytes per element. The minimum bytes assume each matrix is read or written exactly once. Real kernels approach it by tiling, which later modules cover.

~~~dg-raw
%%grid How a 4×4 output is computed in 2×2 tiles: each tile of C needs one row-band of A and one column-band of B.
cols = c0, c1, c2, c3
key A = tile 1 | c1
key B = tile 2 | c2
key C = tile 3 | go
key D = tile 4 | c3
r0 | A A B B
r1 | A A B B
r2 | C C D D
r3 | C C D D
%%
~~~

## Deep dive: the shapes of one Llama 3 8B layer

With ⟦m58⟧ tokens in the batch, hidden size ⟦m59⟧, 32 query heads, 8 KV heads of size 128, and MLP width 14,336:

| Operation | Shapes | FLOPs per token |
| --- | --- | --- |
| Q projection | ⟦m60⟧ | 33.6M |
| K and V projections | ⟦m61⟧ each | 16.8M |
| Attention output | ⟦m62⟧ | 33.6M |
| Gate and up | ⟦m63⟧ each | 234.9M |
| Down | ⟦m64⟧ | 117.4M |
| Total linear, per layer | | 436M |

Many engines fuse Q, K and V into one ⟦m65⟧ weight, and gate and up into one ⟦m66⟧ weight. Fewer, larger matmuls run closer to peak.

~~~dg-raw
%%bars Where the per-layer weight bytes go (BF16, MB).
unit = MB
Q | 33.6 | c1
K | 8.4 | c1
V | 8.4 | c1
O | 33.6 | c1
Gate | 117.4 | c3
Up | 117.4 | c3
Down | 117.4 | c3
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the same projection at prefill and decode**

Down projection, prefill of 2,048 tokens: ⟦m67⟧ FLOPs; bytes ⟦m68⟧. Intensity ≈ 1,250 FLOP/byte: compute bound.

Same projection, decode at batch 8: ⟦m69⟧ FLOPs; bytes ≈ 118 MB, almost all weight. Intensity ≈ 8: memory bound, and the GPU's math units sit mostly idle.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

an `[M, K] × [K, N]` matmul costs 2MKN FLOPs. In inference, M is "tokens this step", which is why prefill and decode behave so differently on the same weights.

</div>
