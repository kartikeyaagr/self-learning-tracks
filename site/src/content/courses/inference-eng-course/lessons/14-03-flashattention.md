---
id: "14.03"
course: "inference-eng-course"
module: "14"
slug: "flashattention"
title: "FlashAttention"
keyIdeas: ["FlashAttention fuses attention into one kernel using tiling and online softmax.", "It never stores the score matrix in HBM, cutting memory traffic sharply.", "Speedups of several times for long sequences; memory linear in length.", "FA2: better work partitioning and parallelism over sequence length.", "FA3: Hopper features (TMA, warp specialization, FP8)."]
question: "FlashAttention does more FLOPs than standard attention in the backward pass, yet it is faster. Why, and what changed in FlashAttention-2 and 3?"
answer: "It is faster because attention is limited by HBM traffic, not FLOPs; FlashAttention keeps tiles in on-chip memory and never writes the score matrix, so it moves far fewer bytes, even if it recomputes some values. FlashAttention-2 improved parallelism and work partitioning across warps and along the sequence. FlashAttention-3 targets Hopper, using asynchronous data movement (TMA), warp specialization and FP8 support to raise tensor core utilization."
sources: [{"title": "FlashAttention", "url": "https://arxiv.org/abs/2205.14135", "note": "Original IO-aware attention."}, {"title": "FlashAttention-2", "url": "https://arxiv.org/abs/2307.08691", "note": "Better parallelism and partitioning."}, {"title": "FlashAttention-3", "url": "https://arxiv.org/abs/2407.08608", "note": "Hopper-specific optimizations."}]
tags: ["attention", "flashattention"]
---

## What FlashAttention is

FlashAttention is a single fused kernel that computes softmax(Q Kᵀ) V for blocks of queries using the tiled online softmax from the previous lesson. It streams K and V tiles from HBM into shared memory, keeps scores and running statistics on chip, and writes only the final output. It was introduced in 2022 and has become the standard attention kernel for training and for inference prefill.

## Why it wins

| Property | Standard attention | FlashAttention |
| --- | --- | --- |
| S × S matrix written to HBM | yes, at least twice | never |
| Memory use | O(S²) | O(S) |
| Kernels per attention call | several | one |
| Bottleneck at long S | memory traffic | compute |

The key insight is that FlashAttention does not reduce the math. It even does slightly more, because of the rescaling. It is faster because it moves far fewer bytes. At long sequence lengths it turns attention from memory bound into compute bound, which is where GPUs are strongest.

## IO-awareness

The FlashAttention paper framed this as **IO-aware** design: count reads and writes between levels of the memory hierarchy, not only FLOPs, and design the algorithm to minimize the slow ones. For a sequence of length S and head size d, with SRAM (shared memory) of size M, FlashAttention needs about S² d² ÷ M accesses to HBM, against about S d + S² for standard attention. For typical d and M, that is many times fewer.

## Versions

- **FlashAttention (v1)**: the original fused kernel, parallelized over batch and heads.
- **FlashAttention-2**: also parallelizes over the sequence length, so long sequences with small batch still fill the GPU; reduces non-matmul work; and divides work between warps better. Roughly 2 times faster than v1, reaching 50 to 70% of peak FLOPs on A100.
- **FlashAttention-3**: designed for Hopper (H100). It uses the Tensor Memory Accelerator (TMA) for asynchronous loads, overlaps softmax computation with matmuls by assigning them to different warp groups, and adds FP8 support with techniques to limit quantization error. Reaches around 75% of peak BF16 FLOPs on H100.

<div class="ex">
<div class="lab">Worked example</div>

**prefill attention time at long context**

A single 32,768-token prefill for an 8B model on an H100, attention only, all 32 layers.

FLOPs (causal): about 4 × 32 layers × 4,096 × 32,768² ÷ 2 ≈ 281 TFLOP.

At 70% of the 990 TFLOP/s peak, that takes about 0.41 s. Standard attention would need to move several terabytes of score matrices and would not fit in memory for all heads at once.

The rest of the prefill (matmuls, about 524 TFLOP) takes about 0.75 s. At 32k context, attention is already more than a third of prefill time, even with FlashAttention.

</div>

## In inference engines

Engines use FlashAttention, or kernels built on the same ideas, for:

- **Prefill**, including chunked prefill where each chunk attends to cached keys from earlier chunks.
- **Variable-length batches**, using a "varlen" interface where all sequences are concatenated with start offsets, so no padding is needed.
- **Paged KV**, where K and V tiles are gathered through block tables.

For pure decode, with one query per sequence, the best tiling is different: a 128-query tile has nothing to fill it. Decode uses specialized variants, covered in the next lesson.

## Using it

In PyTorch, `torch.nn.functional.scaled_dot_product_attention` dispatches to a FlashAttention-style kernel when inputs allow. The `flash-attn` package exposes the kernels directly, including `flash_attn_varlen_func` for ragged batches and paged KV. Engines select an attention backend at startup and report it in the log.

## Deep dive: IO complexity

With SRAM of ⟦m780⟧ elements per SM, head dimension ⟦m781⟧ and sequence length ⟦m782⟧, the FlashAttention paper shows the number of HBM accesses is

⟦M779⟧

For ⟦m783⟧ and ⟦m784⟧ around 100,000 elements, ⟦m785⟧: several times fewer accesses, and none of them for an ⟦m786⟧ matrix. The paper also proves no exact attention algorithm can do asymptotically better for all ⟦m787⟧ in a range.

~~~dg-raw
%%bars Reported share of peak BF16 throughput for fused attention in the forward pass (from the FlashAttention papers, best cases).
unit = % of peak
FlashAttention-2 on A100 | 73 | c3
FlashAttention-3 on H100 | 75 | c1
%%
~~~

FlashAttention-3 reaches this on H100 by overlapping the softmax of one tile with the matmuls of the next (warp specialization), and by using the Hopper asynchronous copy and matrix units. Exact speeds depend on head dimension, sequence length and masking.

<div class="take">
<div class="lab">Key takeaway</div>

FlashAttention is fast because it moves fewer bytes, not because it does less math. It fuses attention into one IO-aware kernel, turns long-context attention compute bound, and is the standard prefill attention kernel.

</div>
