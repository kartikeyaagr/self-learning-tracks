---
id: "19.01"
course: "inference-eng-course"
module: "19"
slug: "how-moe-inference-differs-from-dense-models"
title: "How MoE Inference Differs From Dense Models"
keyIdeas: ["MoE layers replace the dense MLP with many expert MLPs and a router.", "Each token uses only top-k experts: compute scales with active parameters.", "Memory scales with total parameters: all experts must be resident.", "At large batch, most experts get some tokens each step, so most expert weights are read.", "MoE shifts the bottleneck from compute toward memory capacity and communication."]
question: "A MoE model has 235B total parameters and 22B active per token. What does each number mean for memory and for decode speed?"
answer: "All 235B parameters must be stored (470 GB in BF16), so memory and GPU count are set by the total. Each token only uses about 22B parameters, so compute per token is like a 22B dense model. At small batch decode, only the selected experts' weights are read, so bandwidth cost per token is also near 22B; at large batch, different tokens select different experts, so most experts are read every step."
sources: [{"title": "Mixtral of Experts", "url": "https://arxiv.org/abs/2401.04088", "note": "A widely used MoE architecture."}, {"title": "Switch Transformers", "url": "https://arxiv.org/abs/2101.03961", "note": "Sparse expert routing at scale."}]
tags: ["moe"]
---

## The architecture

A mixture-of-experts (MoE) model keeps attention dense and replaces the single MLP in each layer with many smaller MLPs called experts. A small router decides which experts each token visits.

~~~dg-flow An MoE layer for one token.
Token hidden state -> *Router (top-k scores) -> Selected experts (MLPs) -> Weighted sum -> Output
~~~

Typical numbers:

- **Experts per layer:** 8 (Mixtral) to 256 (DeepSeek-V3).
- **Experts per token (k):** usually 2 to 8.
- **Shared expert:** some models add one or two experts that every token visits, on top of the routed ones.

## Two parameter counts

| Count | Mixtral 8x7B | DeepSeek-V3 | What it sets |
| --- | --- | --- | --- |
| Total parameters | about 47B | about 671B | memory, and so the number of GPUs |
| Active parameters per token | about 13B | about 37B | FLOPs per token |

A dense 13B model and Mixtral cost about the same FLOPs per token. Mixtral needs more than three times the memory, though: about 94 GB in BF16 against 26 GB.

<div class="ex">
<div class="lab">Worked example</div>

**what one decode step reads**

Take Mixtral (8 experts, k = 2) at batch 1. Each layer reads attention weights plus 2 of the 8 experts. That is about 13B parameters, or 26 GB in BF16, so on an H100 the step costs roughly what a dense 13B model costs.

At batch 64, each token picks 2 experts, so 128 picks are spread over 8 experts. The chance that a given expert gets no tokens is (6/8)^64, which is effectively zero. Every expert is read, so the step reads all 47B parameters (94 GB), the same as a dense 47B model.

The batch-1 advantage is gone at batch 64, but compute is still that of a 13B model. Throughput per GPU is good. Latency at batch 1 is also good. Memory is the price.

</div>

## Why decode looks different

- **Batch 1:** only k experts per layer are read. The model runs fast for its size.
- **Large batch:** tokens spread over all experts, so the bytes read per step approach the full model. Each expert, however, sees only about (batch × k ÷ experts) tokens.
- With 256 experts, k = 8 and a batch of 64, each expert sees about 2 tokens per step. Its matmul is a tiny, memory-bound GEMV. The full weight read buys very little compute.

This is the central tension of MoE serving. You need very large batches across the whole deployment to give each expert enough tokens to be efficient.

## What changes for inference engineers

- **Memory capacity**, not compute, often sets the GPU count.
- **Expert placement and all-to-all communication** become first-class concerns.
- **Load balance** across experts and GPUs sets step time.
- **Grouped GEMM kernels** are needed to run many small, uneven matmuls in one launch.
- **Batch size** matters more than for dense models, because it is divided among experts.

## Deep dive: how much of an MoE layer a batch reads

With ⟦m940⟧ experts, top-⟦m941⟧ routing and ⟦m942⟧ tokens routed roughly at random, the expected share of experts that at least one token uses is

⟦M939⟧

Decode reads the weights of every used expert once, so this share sets the bytes per step.

~~~dg-raw
%%chart Share of expert weights read per decode step against batch size.
x = 1, 4, 8, 16, 32, 64, 128
logx = 1
xlab = Tokens in the step
ylab = Experts touched (%)
8 experts, top-2 (Mixtral style): 25, 68.4, 90, 99, 100, 100, 100
256 experts, top-8 (DeepSeek-V3 style): 3.1, 11.9, 22.4, 39.8, 63.8, 86.9, 98.3
%%
~~~

At batch 1, an MoE model reads only its active parameters and decodes like a much smaller dense model. By batch 16 for Mixtral, or about 128 for a 256-expert model, almost every expert is read every step, and decode bytes approach those of a dense model of the total size. MoE's speed advantage in decode is a small-batch advantage. Its compute advantage (FLOPs per token follow active parameters) holds at every batch size.

<div class="take">
<div class="lab">Key takeaway</div>

MoE separates compute (active parameters) from memory (total parameters). Serving it well is about memory capacity, expert placement, communication and getting enough tokens to each expert.

</div>
