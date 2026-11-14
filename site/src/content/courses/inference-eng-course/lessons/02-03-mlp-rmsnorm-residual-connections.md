---
id: "02.03"
course: "inference-eng-course"
module: "02"
slug: "mlp-rmsnorm-residual-connections"
title: "MLP, RMSNorm & Residual Connections"
keyIdeas: ["The MLP transforms each token independently: up-project, nonlinearity, down-project.", "SwiGLU: down(silu(gate(x)) * up(x)). Three weight matrices.", "RMSNorm scales by root-mean-square with a learned gain. Cheap but a separate kernel unless fused.", "Residual connections add each sublayer's output to the running hidden state.", "Small elementwise ops are cheap in FLOPs but cost memory traffic and kernel launches; fusion helps."]
question: "Why is the MLP typically the largest compute cost in decode at short context, and how does SwiGLU change the weight count compared with a plain two-matrix MLP?"
answer: "The MLP processes every token through large matrices (hidden to about 2.7 to 4 times hidden and back), and at short context attention over the cache is cheap, so the MLP's matmuls dominate both FLOPs and weight bytes. SwiGLU uses three matrices (gate, up, down) instead of two, and shrinks the intermediate width to about 8/3 of hidden to keep total parameters similar."
sources: [{"title": "GLU Variants Improve Transformer", "url": "https://arxiv.org/abs/2002.05202", "note": "SwiGLU and related gated MLPs."}, {"title": "Root Mean Square Layer Normalization", "url": "https://arxiv.org/abs/1910.07467", "note": "RMSNorm paper."}]
tags: ["transformer"]
---

## What happens after attention

Attention moves information between tokens. The rest of each layer processes every token independently: a normalization, an MLP and residual additions. These parts hold most of the weights and most of the FLOPs, and some of them cost far more time than their math suggests.

## The gated MLP

Older transformers used a two-matrix MLP: expand, apply an activation, project back. Modern LLMs use a gated variant called **SwiGLU**:

```
mlp(x) = W_down( silu(W_gate · x) * (W_up · x) )
```

- `W_gate` and `W_up` both map from d to f, the intermediate size.
- `silu(z) = z × sigmoid(z)` is a smooth activation.
- The element-wise product lets the gate branch switch features on and off.
- `W_down` maps from f back to d.

With d = 4,096 and f = 14,336, as in Llama 3 8B, the three matrices hold 3 × 4,096 × 14,336 ≈ 176M parameters per layer, about 80% of the layer.

<div class="ex">
<div class="lab">Worked example</div>

**MLP cost for one token**

FLOPs: three matmuls at 2 × 4,096 × 14,336 each, which is about 352 MFLOP per layer, or 11.3 GFLOP across 32 layers.

Bytes at batch 1: 176M parameters × 2 bytes = 352 MB per layer, or 11.3 GB in total. The MLP alone is about 70% of all the weight bytes read in a decode step.

Doubling the batch to 2 doubles the FLOPs but reads the same 11.3 GB. That is why batching is almost free for the MLP until the batch gets large.

</div>

## RMSNorm

Normalization keeps the scale of the hidden state stable as it passes through many layers. Modern LLMs use RMSNorm:

```
rmsnorm(x) = x / sqrt(mean(x²) + eps) × g
```

It divides by the root mean square and multiplies by a learned gain vector g. Compared with LayerNorm, it skips subtracting the mean and adding a bias. The math is cheap: a few operations per element.

The cost is not the math. A standalone norm kernel must read the whole hidden state from GPU memory and write the result back. For a batch of 256 tokens with d = 4,096 in BF16, that is 2 MB in and 2 MB out, plus a kernel launch of a few microseconds. Run twice per layer across 32 layers, these small costs add up.

## Residual connections

Each sublayer adds its output to the running hidden state:

```
h = h + attention(rmsnorm(h))
h = h + mlp(rmsnorm(h))
```

This is called **pre-norm**: normalize before the sublayer, then add. Residual paths let gradients flow during training and let each layer make small changes to a shared stream.

For inference, a residual add is another memory round trip: read h, read the sublayer output, write the sum. Then the next norm reads the sum again.

## Fusion: removing the memory passes

Because norms, activations and adds are memory bound, engines fuse them with their neighbours so the data is read once:

- **Fused add + RMSNorm**: one kernel computes `h = h + a` and `x = rmsnorm(h)`, reading and writing h once instead of three times.
- **Fused SiLU and multiply**: compute `silu(gate) * up` in one pass. Some engines also merge `W_gate` and `W_up` into a single wider matmul so both run as one kernel.
- **Norm folded into the next matmul**: in some quantized paths, the norm gain is folded into the weights.

| Op | Weights | FLOPs per token | Memory traffic per token (BF16) |
| --- | --- | --- | --- |
| Gate + up matmul | 2df | 4df | weights, shared across the batch |
| SiLU × up | 0 | about 3f | read 2f, write f |
| Down matmul | df | 2df | weights, shared across the batch |
| RMSNorm | d | about 4d | read d, write d |
| Residual add | 0 | d | read 2d, write d |

The weight rows are shared across the batch, so their cost per token falls as the batch grows. The element-wise rows are per token, so their cost does not fall. At large batch they become a noticeable share of step time unless fused.

## Activation memory

The MLP's intermediate tensors are the largest activations in a forward pass. For a prefill chunk of 8,192 tokens, `gate` and `up` each take 8,192 × 14,336 × 2 bytes ≈ 235 MB. Engines size their prefill chunks partly to keep these temporaries within budget, which is one reason chunked prefill exists.

## Deep dive: the gated MLP in one line

⟦M247⟧

⟦m249⟧ is element-wise multiplication. The gate decides, per dimension, how much of the "up" signal passes. Compared with an ungated two-matrix MLP of the same width, it has 1.5 times the parameters, which is why gated models use a narrower width (⟦m250⟧ rounded, 14,336 for ⟦m251⟧).

## Deep dive: RMSNorm

⟦M248⟧

⟦m252⟧ is a learned scale. Unlike LayerNorm, it does not subtract the mean, which saves one reduction. It must be computed in FP32 even when ⟦m253⟧ is BF16: summing 4,096 squares in BF16 loses most of the precision.

## Deep dive: what fusion saves, in bytes

Unfused, "add residual, then RMSNorm" reads and writes the hidden state several times. Count bytes for ⟦m254⟧ tokens of width ⟦m255⟧ in BF16:

| Step | Reads | Writes |
| --- | --- | --- |
| residual add | 2 × 2Td | 2Td |
| compute sum of squares | 2Td | small |
| normalize and scale | 2Td | 2Td |
| Total unfused | 8Td | 4Td |
| Fused kernel | 4Td | 4Td |

The fused kernel moves ⟦m256⟧ bytes instead of ⟦m257⟧, a third less. For decode these ops are tiny next to the weights, but for prefill at large ⟦m258⟧ the saving is real, and the bigger gain is fewer kernel launches.

~~~dg-raw
%%bars Memory passes over the hidden state for "residual add + RMSNorm" (relative bytes moved).
unit = × Td bytes
Unfused (3 kernels) | 12 | c1
Fused (1 kernel) | 8 | go
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

the MLP holds most of the weights and FLOPs, and its cost per token falls with batch size. Norms, activations and residual adds are cheap math but cost memory passes and launches, which is why engines fuse them.

</div>
