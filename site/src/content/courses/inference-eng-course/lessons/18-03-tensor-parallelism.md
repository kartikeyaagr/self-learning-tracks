---
id: "18.03"
course: "inference-eng-course"
module: "18"
slug: "tensor-parallelism"
title: "Tensor Parallelism"
keyIdeas: ["Tensor parallelism splits each layer's weights across GPUs.", "Attention: split by heads; MLP: column-split up/gate, row-split down.", "Two all-reduces per layer (after attention and after MLP).", "Each GPU holds 1/TP of weights and of KV heads: more memory for KV, faster per-step reads.", "Works best inside a node over NVLink; limited by KV heads (TP should divide KV heads, or replicate them)."]
question: "In Megatron-style tensor parallelism of an MLP, which matrix is split by columns, which by rows, and where is the all-reduce?"
answer: "The first matrices (gate and up projections) are split by columns, so each GPU computes part of the intermediate activations with no communication. The down projection is split by rows, so each GPU produces a partial sum of the full output. One all-reduce sums those partial outputs at the end of the MLP. Attention is split by heads in the same way, with one all-reduce after the output projection."
sources: [{"title": "Megatron-LM", "url": "https://arxiv.org/abs/1909.08053", "note": "Column and row parallel layers."}, {"title": "vLLM: Parallelism and scaling", "url": "https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html", "note": "Tensor parallel in practice."}]
tags: ["tensor-parallel"]
---

## The idea

Tensor parallelism (TP) splits each layer's weight matrices across GPUs, so each GPU stores and computes only part of every layer. All GPUs work on the same tokens at the same time. The method most engines use comes from Megatron-LM, and it is designed to need only two communication steps per layer.

## Splitting the MLP

~~~dg-flow A tensor-parallel MLP on 2 GPUs. Press Play to follow the data.
x, replicated -> *Gate and up: column shards -> SiLU × up, local -> Down: row shards -> All-reduce -> Output
~~~

1. Every GPU has the full input x, shape [tokens, d].
2. The gate and up matrices are split by **columns**: each GPU holds half of the intermediate dimension. Each GPU computes its half of gate and up with no communication.
3. SiLU and the element-wise multiply are local, because each GPU has complete values for its half of the intermediate features.
4. The down matrix is split by **rows**, matching that half. Each GPU computes a partial result of shape [tokens, d].
5. An **all-reduce** sums the partial results, so every GPU has the full output.

## Splitting attention

Attention splits naturally by heads. With 32 query heads and 8 KV heads on 4 GPUs, each GPU takes 8 query heads and their 2 KV heads:

- Each GPU computes its heads' Q, K and V projections (a column split).
- Each GPU runs attention for its heads using its own part of the KV cache.
- The output projection is split by rows, and an all-reduce sums the result.

So each layer needs exactly two all-reduces: one after attention and one after the MLP.

## What TP gives you

| Effect | Why |
| --- | --- |
| Fits bigger models | each GPU holds 1/TP of the weights |
| More KV capacity per replica | each GPU stores only its heads' KV, and has more free memory |
| Lower TPOT | each GPU reads 1/TP of the weights per step |

~~~dg-raw
%%widget tp Change the tensor-parallel degree and interconnect. Watch the step split into compute and communication, and see when NVLink stops helping.
model = Llama 3 70B
b = 5
%%
~~~

## What it costs

- **Two all-reduces per layer per step.** For an 80-layer model, 160 per step.
- **Diminishing returns.** Doubling TP halves each GPU's weight read but not the communication. At high TP on a small model, communication can exceed compute.
- **KV heads limit the degree.** With 8 KV heads, TP up to 8 gives each GPU at least one whole KV head. Beyond 8, KV heads must be replicated, which wastes memory.
- **Smaller matmuls.** Each GPU's matrices are narrower, which can lower kernel efficiency at small batch.

| TP degree | Weight read per GPU | Communication per step | Typical use |
| --- | --- | --- | --- |
| 1 | 100% | none | the model fits on one GPU |
| 2 | 50% | moderate | 30B to 70B models, or lower latency |
| 4 | 25% | more | 70B in BF16 |
| 8 | 12.5% | high | 400B dense models or large MoE |

<div class="ex">
<div class="lab">Worked example</div>

**TPOT for Llama 3 70B**

In BF16, the weights are 140 GB, so at least TP 2 on 80 GB GPUs, and TP 4 leaves room for KV.

At TP 4 on H100s: each GPU reads 35 GB per step, about 11.5 ms at 3 TB/s. 160 all-reduces at about 15 µs each add about 2.4 ms. TPOT at low load is about 14 ms.

At TP 8: each GPU reads 17.5 GB, about 5.8 ms, plus about 2.6 ms of all-reduces. TPOT is about 8.5 ms. Faster, but now communication is almost a third of the step.

</div>

## Deep dive: TPOT under tensor parallelism

Each layer does two all-reduces (after attention and after the MLP). With per-all-reduce latency ⟦m906⟧:

⟦M905⟧

~~~dg-raw
%%chart Predicted batch-1 TPOT for Llama 3 70B in BF16 on H100s against TP degree.
x = 2, 4, 8
xlab = TP degree
ylab = TPOT (ms)
Weight reads: 25, 12.5, 6.25
All-reduce latency (NVLink): 1.9, 2.4, 3.2
Total: 27.9, 15.9, 10.5
%%
~~~

From TP 4 to TP 8, weight reads halve but the all-reduce term grows, so TPOT improves 1.5 times, not 2. Each further doubling helps less, because ⟦m907⟧ does not shrink. That fixed term is why custom low-latency all-reduce kernels (one-shot, in-place over NVLink) matter so much for decode.

<div class="take">
<div class="lab">Key takeaway</div>

tensor parallelism splits every layer across GPUs, with two all-reduces per layer. It fits larger models, adds KV capacity and lowers TPOT, works best inside a server over NVLink, and communication caps how far it scales.

</div>
