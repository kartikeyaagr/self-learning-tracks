---
id: "13.01"
course: "ai-research-track-web"
module: "13"
slug: "compute-and-memory-accounting"
title: "Compute and memory accounting"
keyIdeas: ["A dense decoder has about $12Ld^2$ parameters in its blocks ($4d^2$ for attention projections, $8d^2$ for an MLP with a $4d$ hidden layer) plus $Vd$ per embedding matrix. Count before you design anything else.", "Training costs about $6N$ FLOPs per token: $2N$ for the forward pass and $4N$ for the backward pass. Attention's score and value products add about $12LTd$ per token, which relative to $6N$ is roughly $T/(6d)$.", "Mixed-precision training with Adam needs about 16 bytes per parameter before activations: bf16 weights and gradients (2 + 2) plus fp32 master weights and two fp32 moments (4 + 4 + 4).", "Activation memory grows with micro-batch size, sequence length, hidden size and depth. Without a fused attention kernel the stored score matrices add a term quadratic in sequence length, and at long context activations run out before weights do.", "Convert FLOPs to time with achieved throughput, not peak: GPU-seconds = total FLOPs / (peak FLOP/s x MFU). Model FLOPs utilisation of 35 to 50 percent is typical for well-tuned dense training."]
question: "You are asked to cost a run: a dense decoder with 48 layers, hidden size 6144, vocabulary 50,000 (untied embeddings), context 2048, trained on 600B tokens on 256 H100s (989 TFLOP/s bf16 peak) at 45 percent MFU. Estimate the parameter count, the training FLOPs with and without the attention term, the wall-clock time, and the model-state memory. Can any single GPU hold the full optimiser state?"
answer: "Parameters: blocks $12 \\times 48 \\times 6144^2 = 21.7$B, embeddings $2 \\times 50{,}000 \\times 6144 = 0.61$B, so $N \\approx 22.4$B. Training FLOPs: $6 \\times 22.4\\text{B} \\times 600\\text{B} = 8.05 \\times 10^{22}$. The attention term is $12 \\times 48 \\times 2048 \\times 6144 \\times 600\\text{B} = 4.35 \\times 10^{21}$, about 5.4 percent extra, consistent with $T/(6d) = 2048/36{,}864 = 5.6$ percent; short contexts make attention cheap relative to the weights. Total about $8.48 \\times 10^{22}$ FLOPs. Throughput: $256 \\times 989 \\times 10^{12} \\times 0.45 = 1.14 \\times 10^{17}$ FLOP/s, so about $7.4 \\times 10^5$ seconds, roughly 8.6 days, before restarts and evaluation overhead. Model state at 16 bytes per parameter: $16 \\times 22.4\\text{B} = 358$ GB. No single 80 GB GPU can hold it; even before activations you need the state split at least five ways, and in practice sharded across all 256 GPUs (about 1.4 GB each under full sharding) or split by tensor and pipeline parallelism. The first planning decision is therefore how to partition state, not how big a batch to use."
sources: [{"title": "Kaplan et al.: Scaling Laws for Neural Language Models (2020)", "url": "https://arxiv.org/abs/2001.08361", "note": "Section 2.1 and Table 1: the per-operation parameter and FLOP counts behind the 6N rule."}, {"title": "Korthikanti et al.: Reducing Activation Recomputation in Large Transformer Models (2022)", "url": "https://arxiv.org/abs/2205.05198", "note": "Section 4: the per-layer activation memory derivation, term by term."}, {"title": "How to Scale Your Model (JAX scaling book)", "url": "https://jax-ml.github.io/scaling-book/", "note": "The chapter on Transformer maths: parameter, FLOP and memory counting with worked numbers."}, {"title": "Rajbhandari et al.: ZeRO (2019)", "url": "https://arxiv.org/abs/1910.02054", "note": "Section 3.1: where the 16 bytes per parameter of mixed-precision Adam come from."}]
practice: ["transformer-training-flop-estimator", "mixed-precision-training-memory-accountant", "training-compute-budget", "gradient-accumulation-equivalence"]
tags: ["FLOPs", "memory", "MFU"]
tools: ["PyTorch", "Python"]
---

## Why count before you launch

A large training run is a budget negotiation between three quantities: FLOPs (how long it takes), memory (whether it fits) and bandwidth (whether the GPUs wait on each other). The first two can be estimated to within ten or twenty percent with arithmetic you can do on paper, and that estimate decides the model shape, the number of GPUs, which parallelism and sharding you need, and whether the schedule is days or months. Every later lesson in this module starts from these numbers.

## Counting parameters

For a GPT-style block with hidden size $d$, the attention projections $W_Q, W_K, W_V, W_O$ are each $d \times d$, and the MLP has an up-projection $d \times 4d$ and a down-projection $4d \times d$. Norm weights and biases are $O(d)$ and negligible.

| component | parameters per layer | note
| --- | --- | --- |
| attention Q, K, V, O | $4d^2$ | grouped-query attention shrinks K and V
| MLP (4d hidden) | $8d^2$ | SwiGLU with hidden $8d/3$ and three matrices is also $8d^2$
| norms, biases | $O(d)$ | ignore for budgeting
| embeddings | $Vd$ | doubled if input and output are untied

So $N \approx 12Ld^2 + 2Vd$ for an untied model. With $L = 32$, $d = 4096$ and $V = 32{,}000$ the blocks hold 6.44B parameters and the two embedding matrices 0.26B: a 6.7B model. Mixture-of-experts models need two counts, total parameters (memory) and active parameters per token (compute).

## FLOPs per token: the 6N rule

A matmul of a $1 \times k$ activation with a $k \times n$ weight costs $2kn$ FLOPs (a multiply and an add per weight). Summed over all weights, one token's forward pass costs about $2N$. The backward pass computes two products per weight matrix, the gradient with respect to the input and the gradient with respect to the weight, so it costs about $4N$. Training therefore costs

$$C \approx 6ND$$

for $D$ training tokens. This ignores attention's own products. Per layer and per token, $QK^\top$ and the multiplication by $V$ each cost about $2Td$ FLOPs for context length $T$, so $4LTd$ forward and $12LTd$ for training. Relative to $6N \approx 72Ld^2$ that is $T/(6d)$: about 17 percent at $T = d = 4096$, and growing linearly with context. Causal masking lets fused kernels skip roughly half of that work. Reports differ on whether they count the attention term at all, so check before comparing utilisation figures.

## Memory: the 16-byte rule

The persistent state of mixed-precision training with Adam, per parameter:

~~~dg-layers Bytes per parameter in mixed-precision Adam training (before activations).
bf16 weights | 2 bytes, used in forward and backward
bf16 gradients | 2 bytes, produced by backward
*fp32 master weights | 4 bytes, what the optimiser actually updates
fp32 Adam first moment | 4 bytes
fp32 Adam second moment | 4 bytes
~~~

That is 16 bytes, so a 7B model carries about 107 GB of state: it does not fit on one 80 GB GPU no matter how small the batch is. This is the motivation for the sharding in the next lesson. Some frameworks keep gradients in fp32 (18 to 20 bytes); 8-bit optimisers or bf16 moments push it lower at some risk to stability.

## Activations

Backward needs the intermediate tensors from forward. For a GPT-style layer with 16-bit activations, sequence length $s$, micro-batch $b$, hidden $h$ and $a$ heads, Korthikanti et al. estimate roughly

$$s\,b\,h\left(34 + \frac{5as}{h}\right) \text{ bytes per layer.}$$

The $34sbh$ part is linear in sequence length (layer inputs, projections, MLP intermediates, dropout masks). The $5as^2b$ part is the attention scores, softmax output and dropout mask, quadratic in $s$. FlashAttention never materialises those, so with it the per-layer figure drops to about $34sbh$. The constants assume the original GPT block; SwiGLU and no dropout shift them, so treat the formula as an approximation that tells you which term dominates.

Do not forget the logits: $b \times s \times V$ values, often upcast to fp32 for the loss. At $s = 4096$ and $V = 128{,}000$ that is about 2.1 GB per sequence, which is why fused or chunked cross-entropy kernels exist.

## From FLOPs to wall-clock time

Model FLOPs utilisation (MFU) is the FLOPs the model mathematically requires per second divided by the hardware's peak. Recomputed activations do not count toward it (that is hardware FLOPs utilisation, HFU). Wall-clock time is

$$t = \frac{C}{n_{\text{GPU}} \times \text{peak} \times \text{MFU}}.$$

An H100 SXM's dense bf16 peak is about 989 TFLOP/s. Well-tuned dense training reaches roughly 35 to 50 percent MFU; small models, long sequences or heavy communication do worse.

~~~dg-flow From model shape to a training schedule.
Shape (L, d, V) -> Parameters N -(x 6 x D)-> *Training FLOPs C -(peak x MFU x GPUs)-> Wall-clock days
Parameters N -(x 16 bytes)-> State memory -> Sharding plan
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a 6.7B model on one trillion tokens**

Shape: $L = 32$, $d = 4096$, $V = 32{,}000$, untied embeddings, context 4096. Parameters: $12 \times 32 \times 4096^2 = 6.44$B in blocks plus $2 \times 32{,}000 \times 4096 = 0.26$B in embeddings, so $N = 6.70$B.

Compute: $6 \times 6.70\text{B} \times 10^{12} = 4.02 \times 10^{22}$ FLOPs. Attention adds $12 \times 32 \times 4096 \times 4096 = 6.4$ GFLOPs per token against $6N = 40.2$ GFLOPs, 16 percent more.

Time at 40 percent MFU on H100s: $4.02 \times 10^{22} / (989 \times 10^{12} \times 0.4) = 1.02 \times 10^8$ GPU-seconds, about 28,200 GPU-hours, or 2.3 days on 512 GPUs (ignoring the attention term).

Memory: $16 \times 6.70\text{B} = 107$ GB of state. Activations at micro-batch 1 without a fused attention kernel: $4096 \times 4096 \times (34 + 160) = 3.25$ GB per layer, 104 GB across 32 layers. With FlashAttention: 0.57 GB per layer, 18 GB total. Both state and activations need to be split across GPUs or reduced.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

$N \approx 12Ld^2$, $C \approx 6ND$, 16 bytes per parameter for Adam state, and an activation term that scales with $s \cdot b \cdot h \cdot L$; four lines of arithmetic tell you whether a plan fits and how long it takes.

</div>
