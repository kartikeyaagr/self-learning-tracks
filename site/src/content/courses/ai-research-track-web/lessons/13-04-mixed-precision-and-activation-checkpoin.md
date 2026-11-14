---
id: "13.04"
course: "ai-research-track-web"
module: "13"
slug: "mixed-precision-and-activation-checkpointing"
title: "Mixed precision and activation checkpointing"
keyIdeas: ["Low-precision formats trade mantissa bits (precision) against exponent bits (range). bf16 keeps fp32's range with 8 significant bits; fp16 has more precision but overflows above 65504; fp8 comes as E4M3 (max 448) and E5M2 (max 57344).", "Matmuls run in 16 or 8 bits, but the optimiser updates fp32 master weights. A small update added to a bf16 weight is often rounded away entirely, so pure-bf16 training silently stalls late in a schedule.", "fp16 needs loss scaling to keep small gradients from underflowing; bf16 does not. fp8 needs scaling factors per tensor or per block, because its range is too narrow for raw activations and gradients.", "Activation checkpointing stores only a few tensors per layer and recomputes the rest in backward. Full recomputation costs about one extra forward pass (training FLOPs go from about $6N$ to $8N$ per token).", "Selective recomputation recomputes only the parts that are memory-heavy but cheap to compute, chiefly the attention score matrices, which removes the quadratic activation term for a few percent of extra compute."]
question: "To save memory a teammate stores weights and Adam moments in bf16, with no fp32 master copy. The learning rate decays to $3 \\times 10^{-5}$ by the end of training and typical weights have magnitude around 0.02. Adam's per-step updates are roughly the learning rate in size. What happens to the updates, and what would you change? Separately, the same run uses full activation checkpointing; by how much does that inflate the FLOPs per token, and what would you try first instead?"
answer: "A bf16 weight of about 0.02 lies in the binade $[2^{-6}, 2^{-5})$ = [0.0156, 0.0313), where consecutive bf16 values are $2^{-6} \\times 2^{-7} = 2^{-13} \\approx 1.22 \\times 10^{-4}$ apart. An update of $3 \\times 10^{-5}$ is below half that gap ($6.1 \\times 10^{-5}$), so it is rounded away: the weight does not move, and late training stalls while the loss curve flattens for no visible reason. Larger updates are quantised to multiples of the gap, so even those that survive are distorted (an update of $10^{-4}$ becomes a step of $1.22 \\times 10^{-4}$). The fix is fp32 master weights (adds 4 bytes per parameter, usually recovered by sharding them with ZeRO), or if memory forbids it, stochastic rounding or Kahan-compensated updates in bf16. Keep the Adam second moment in fp32 too, since it is used in a square root and a division. On recomputation: full checkpointing adds one forward pass, about $2N$ on top of $6N$, so 33 percent more FLOPs per token. Try selective recomputation of the attention core, or simply FlashAttention, which removes the quadratic activation term at a few percent cost, plus sequence parallelism if tensor parallelism is in use; fall back to full checkpointing only for layers that still do not fit."
sources: [{"title": "Micikevicius et al.: Mixed Precision Training (2017)", "url": "https://arxiv.org/abs/1710.03740", "note": "The three ingredients: fp32 master weights, loss scaling and fp32 accumulation, with the evidence for each."}, {"title": "Micikevicius et al.: FP8 Formats for Deep Learning (2022)", "url": "https://arxiv.org/abs/2209.05433", "note": "The E4M3 and E5M2 definitions, their ranges and how scaling is applied in training."}, {"title": "Korthikanti et al.: Reducing Activation Recomputation in Large Transformer Models (2022)", "url": "https://arxiv.org/abs/2205.05198", "note": "Selective recomputation and sequence parallelism, with the memory and overhead measurements."}, {"title": "Chen et al.: Training Deep Nets with Sublinear Memory Cost (2016)", "url": "https://arxiv.org/abs/1604.06174", "note": "The original gradient-checkpointing analysis and the square-root schedule."}]
practice: ["mixed-precision-training-memory-accountant", "mixed-precision-with-loss-scaling", "rematerialized-mlp-block", "block-wise-fp8-quantization"]
tags: ["mixed precision", "fp8", "activation checkpointing"]
tools: ["PyTorch", "Transformer Engine", "Megatron-LM"]
---

## Why precision is a budget

Tensor cores deliver roughly twice the throughput in bf16 that they do in tf32, and fp8 roughly doubles it again; halving the bytes per value also halves memory traffic and activation storage. The question is which tensors can afford fewer bits. The PyTorch course covered `autocast` and `GradScaler` mechanics; this lesson is about the numerics behind them at scale and about trading compute for memory with recomputation.

| format | exponent / mantissa bits | largest finite | typical role
| --- | --- | --- | --- |
| fp32 | 8 / 23 | about $3.4 \times 10^{38}$ | master weights, optimiser moments, reductions
| bf16 | 8 / 7 | about $3.4 \times 10^{38}$ | matmul inputs, activations, gradients
| fp16 | 5 / 10 | 65504 | the same, on older GPUs, with loss scaling
| fp8 E4M3 | 4 / 3 | 448 | forward weights and activations
| fp8 E5M2 | 5 / 2 | 57344 | gradients, which need range more than precision

## Master weights

Mixed-precision training (Micikevicius et al.) keeps an fp32 copy of every weight. Each step casts it to 16 bits for forward and backward, then applies the optimiser update to the fp32 copy. The reason is rounding. bf16 has 8 significant bits, so the gap between representable numbers near a value $w$ is about $w \times 2^{-7}$. Any update smaller than half that gap is rounded away: near 1.0 the gap is $2^{-7} \approx 0.0078$, so adding $10^{-4}$ changes nothing. Adam's updates are roughly the learning rate in size, and late in a cosine or decay schedule they fall well below the gap for most weights. The run does not crash; it quietly stops learning.

~~~dg-flow One mixed-precision step: low precision where it is fast, fp32 where it accumulates.
*fp32 master weights -(cast)-> bf16 weights -> Forward and backward in bf16 -> bf16 gradients -(reduce, upcast)-> fp32 Adam update -> *fp32 master weights
~~~

Reductions follow the same logic: softmax, norms, losses and gradient all-reduces over many ranks accumulate many small terms and are done in fp32. Alternatives to fp32 master weights exist (stochastic rounding, Kahan-compensated summation in bf16), but fp32 masters are the default because they are simple and robust; they are 4 of the 16 bytes per parameter from lesson 1.

## Loss scaling for fp16

fp16's smallest normal number is about $6.1 \times 10^{-5}$, and many activation gradients are smaller. Multiplying the loss by a scale $S$ multiplies every gradient by $S$, lifting them into range; dividing by $S$ before the update undoes it. Dynamic scaling starts with a large $S$, halves it and skips the step when an inf or NaN appears, and grows it after a run of clean steps. bf16 shares fp32's exponent range, so it needs no scaling, which is why it replaced fp16 for LLM training on hardware that supports it.

## fp8

fp8's range is too small to hold raw tensors, so every fp8 tensor carries a scale factor: the value stored is $x/s$ with $s$ chosen from the tensor's absolute maximum so that the largest entry lands near the format's maximum. Common recipes use E4M3 for weights and activations and E5M2 for gradients. NVIDIA's Transformer Engine uses delayed scaling, which picks each tensor's scale from a history of recent maxima so no extra pass over the data is needed. Per-tensor scales are fragile when a few outlier channels dominate the maximum, so recent large runs use finer-grained scaling, one factor per small tile or block (DeepSeek-V3 is a published example). Even then the fp8 region is limited to the big matmuls: embeddings, the output head, norms, attention softmax and the optimiser stay in bf16 or fp32. Expect loss curves to track bf16 closely when the recipe is right, and to diverge after many thousands of steps when it is not, so compare against a bf16 baseline on a long enough run.

## Activation checkpointing

The other big memory consumer is activations (lesson 1: about $34sbh$ bytes per layer with fused attention, more without). Checkpointing saves only the input of each layer, $2sbh$ bytes in bf16, discards the rest after forward, and recomputes the layer's forward just before its backward. Peak activation memory becomes $L \times 2sbh$ plus one layer's full activations.

The price is one extra forward pass. Forward is about $2N$ of the $6N$ FLOPs per token, so full recomputation raises the cost to about $8N$: 33 percent more compute for the same tokens. Model FLOPs utilisation does not count the recomputed work, so MFU drops even though the hardware is just as busy. Checkpointing every $k$-th layer instead gives intermediate trade-offs; Chen et al. showed that checkpointing about every $\sqrt{L}$ layers gives $O(\sqrt{L})$ memory for one extra forward.

Selective recomputation (Korthikanti et al.) observes that the activations are not equally expensive. The attention scores, softmax output and dropout mask make up the quadratic $5as^2b$ term but cost few FLOPs to recompute, while the matmul outputs are cheap to store relative to recomputing them. Recomputing only the attention core removes the quadratic term at a few percent extra compute. FlashAttention does exactly this by construction: it never stores the score matrix and recomputes it tile by tile in backward. Combined with tensor and sequence parallelism, which divide the remaining activations by $t$, selective recomputation is usually enough, and full checkpointing is kept for when memory is still short.

<div class="ex">
<div class="lab">Worked example</div>

**two failures the budget predicts**

Rounding: a weight of 1.0 receiving 1,000 Adam updates of $+10^{-4}$ should end at 1.1. Accumulated in fp32 it does. Accumulated in bf16, each $1 + 10^{-4}$ rounds back to 1.0 (the gap at 1.0 is $2^{-7} = 0.0078$, half of it 0.0039), so after 1,000 steps the weight is still exactly 1.0.

Memory: the 6.7B model from lesson 1 ($L = 32$, $h = 4096$, $s = 4096$, micro-batch 1, 32 heads) needs about 104 GB of activations with unfused attention. Selective recomputation (or FlashAttention) removes the $5as/h = 160$ term, leaving $34 \times 4096 \times 4096 \times 32 = 18.3$ GB. Full checkpointing keeps only layer inputs: $2 \times 4096 \times 4096 = 33.6$ MB per layer, 1.07 GB for 32 layers, plus one layer being recomputed (0.57 GB with fused attention), about 1.6 GB, for 33 percent more compute.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

compute in bf16 or fp8 with scaling, accumulate and update in fp32, and buy activation memory with selective recomputation first and full checkpointing only when still needed.

</div>
