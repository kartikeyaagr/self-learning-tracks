---
id: "26.02"
course: "inference-eng-course"
module: "26"
slug: "multi-lora-serving"
title: "Multi-LoRA Serving"
keyIdeas: ["LoRA adds low-rank matrices A (d x r) and B (r x d) to selected layers.", "Multi-LoRA serving shares base weights across all adapters.", "Batched kernels apply different adapters to different requests in one step.", "Adapter memory is small; keep many on GPU, page others from CPU.", "Overhead grows with rank and number of distinct adapters per batch."]
question: "How can one engine batch requests for 50 different LoRA adapters in the same step?"
answer: "The base model weights are shared; each request carries an adapter id. For each linear layer, the base matmul runs on the whole batch, then a batched low-rank kernel (like SGMV or BGMV) applies each request's own A and B matrices to its tokens and adds the result. Adapters live in GPU memory (or are swapped in from CPU) in a pool, so many can be active at once."
sources: [{"title": "S-LoRA", "url": "https://arxiv.org/abs/2311.03285", "note": "Serving thousands of LoRA adapters."}, {"title": "Punica", "url": "https://arxiv.org/abs/2310.18547", "note": "SGMV kernel for multi-tenant LoRA."}, {"title": "LoRA paper", "url": "https://arxiv.org/abs/2106.09685", "note": "Low-rank adaptation."}]
tags: ["lora"]
---

## LoRA in one line

```
y = x W + (x A) B × scale     # A: d × r, B: r × d, r small (8 to 64)
```

W is the frozen base weight. A and B are the adapter's two small matrices. Because r is small, the adapter has far fewer parameters than W.

<div class="ex">
<div class="lab">Worked example</div>

**adapter size**

One 4096 × 4096 projection has about 16.8M parameters. A rank-16 adapter on it has 2 × 4096 × 16 = 131k parameters, under 1% of the original.

Applied to all seven projections in all 32 layers of an 8B model, a rank-16 adapter has about 42M parameters, about 84 MB in BF16.

A GPU with 10 GB spare can hold over 100 such adapters next to the base model.

</div>

## Batching across adapters

Different requests in one batch can use different adapters. The base matmul is shared; only the small adapter matmuls differ.

~~~dg-flow One linear layer with mixed adapters.
Batch tokens -> Base GEMM (shared W) -> *Segmented LoRA kernel (per-request A and B) -> Add -> Output
~~~

Kernels such as Punica's SGMV or S-LoRA's handle the per-request adapters in one launch, grouping tokens by adapter.

## Memory tiers

- Hundreds of adapters fit in GPU memory beside the base model.
- Thousands can live in CPU memory and be loaded on demand in a few milliseconds each.
- The rest can stay on disk or object storage.

## Performance

| Factor | Effect |
| --- | --- |
| Rank | higher rank means more FLOPs and memory per adapter |
| Distinct adapters per batch | more segments, slight overhead |
| Adapter swaps | latency on first use of an adapter not in GPU memory |

In practice, serving many LoRAs adds roughly 10 to 30% to step time compared with the base model alone, depending on rank and how many adapters share a batch.

## Cache interaction

Prefix cache keys must include the adapter id. The same prompt produces different KV under different adapters, so a cache hit across adapters would give wrong results.

## Deep dive: the cost of a LoRA adapter, per token

An adapter adds ⟦m1102⟧ with ⟦m1103⟧, ⟦m1104⟧. Applying it unmerged costs, per token and per adapted matrix:

⟦M1101⟧

For ⟦m1105⟧ and ⟦m1106⟧, that is 0.8%. The FLOPs are negligible. The real cost is bytes: at decode, each distinct adapter in the batch must be read from memory, and the batched kernels (gather by adapter index) are less efficient than one dense matmul.

~~~dg-raw
%%chart Decode step time against distinct adapters in a batch of 64 (8B model, rank 16 on all linear layers, illustrative).
x = 1, 4, 16, 64
logx = 1
xlab = Distinct adapters in the batch
ylab = Step time (ms)
Step time: 10.0, 10.2, 10.9, 13.1
Base model alone*: 9.8, 9.8, 9.8, 9.8
%%
~~~

Each rank-16 adapter on all seven linear layers of 32 layers is about 80 MB in BF16. Sixty-four different adapters add about 5 GB of reads per step, which accounts for most of the rise.

<div class="take">
<div class="lab">Key takeaway</div>

multi-LoRA serving shares the base model and applies per-request adapters with batched low-rank kernels. Hundreds of fine-tunes can share one GPU pool.

</div>
