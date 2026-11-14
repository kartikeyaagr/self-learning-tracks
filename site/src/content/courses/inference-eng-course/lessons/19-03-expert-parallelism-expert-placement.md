---
id: "19.03"
course: "inference-eng-course"
module: "19"
slug: "expert-parallelism-expert-placement"
title: "Expert Parallelism & Expert Placement"
keyIdeas: ["Expert parallelism (EP) places different experts on different GPUs.", "Tokens travel to their experts via all-to-all, and back.", "Each GPU stores a fraction of experts, so huge MoE models fit.", "Placement affects balance and traffic; hot experts can be replicated.", "EP is usually combined with DP attention or TP attention."]
question: "With 256 experts per layer over 32 GPUs, how does expert parallelism work, and what does placement decide?"
answer: "Each GPU holds 8 experts per layer. After routing, tokens are sent with all-to-all to the GPUs holding their chosen experts, computed there, and sent back with another all-to-all. Placement decides which experts live on which GPU: grouping often co-activated experts, spreading hot experts, or replicating popular ones changes communication volume and load balance."
sources: [{"title": "DeepSeek-V3 Technical Report", "url": "https://arxiv.org/abs/2412.19437", "note": "Expert parallelism at large scale."}, {"title": "vLLM: Expert parallel deployment", "url": "https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html", "note": "EP configuration."}]
tags: ["moe", "expert-parallel"]
---

## How expert parallelism runs a layer

With expert parallelism (EP), each GPU holds a subset of whole experts. Tokens travel to the GPUs that hold their experts, then travel back.

~~~dg-seq One MoE layer with expert parallelism.
GPU A > GPU A : route local tokens
GPU A > GPU B : dispatch tokens
GPU B > GPU B : run local experts
GPU B > GPU A : combine results
~~~

Both moves are all-to-all: every GPU sends a different slice to every other GPU. Attention is usually run with data parallelism (each GPU handles different requests) or tensor parallelism, separately from the expert layout.

## Memory arithmetic

- DeepSeek-V3 has 256 routed experts in each of 58 MoE layers.
- Spread over 32 GPUs, each GPU holds 8 experts per layer.
- The expert weights alone are well over 600 GB in FP8, so no single 80 GB GPU, or even an 8-GPU server in BF16, holds them all. EP across several servers makes the model fit with room left for KV cache.

## Placement strategies

| Strategy | Effect | Cost |
| --- | --- | --- |
| Round robin | simple and predictable | ignores load |
| Spread hot experts | balances compute | needs statistics |
| Replicate hot experts | evens out load best | extra memory per copy |
| Group co-activated experts | fewer GPUs per token, less traffic | needs co-activation data |

## Tensor parallelism versus expert parallelism for MoE

| | Tensor parallelism | Expert parallelism |
| --- | --- | --- |
| How experts are split | every expert is sliced across all GPUs | each expert lives whole on one GPU |
| Matmul size per GPU | tiny: a slice of a small expert | larger: full expert, more tokens |
| Communication | all-reduce every layer | two all-to-alls every MoE layer |
| Sensitive to imbalance | no | yes |
| Good fit | small MoE (8 experts) on one server | large MoE (64+ experts), many GPUs |

<div class="ex">
<div class="lab">Worked example</div>

**Mixtral on 8 GPUs**

TP = 8 slices each of the 8 experts eight ways. A 14336-wide expert MLP becomes 1792 wide per GPU. At small batch, every GPU does 8 tiny matmuls per layer.

EP = 8 puts one whole expert on each GPU. Each token visits 2 GPUs per layer. With a balanced batch of 256 tokens, each GPU's expert sees about 64 tokens, which makes a reasonable GEMM.

EP wins on matmul efficiency at large batch, but if one expert is popular, its GPU becomes the bottleneck. With only 8 experts there is little room to rebalance, which is why small MoE models often stay on TP.

</div>

## Deep dive: tokens per expert decide matmul efficiency

Each expert's matmul has ⟦m952⟧ rows on average. Its arithmetic intensity is about that number, so

⟦M950⟧

<div class="ex">
<div class="lab">Worked example</div>

**why large-scale MoE serving uses huge batches**

DeepSeek-V3 style: ⟦m953⟧, ⟦m954⟧. At a batch of 256 tokens, each expert sees about 8 tokens: intensity 8, far below the ridge of about 295. The expert matmuls are badly memory-bound.

To reach intensity near 256, a step needs about ⟦m955⟧ tokens. A single GPU cannot hold that much decode batch with its KV, so engines spread experts over many GPUs (wide EP) and gather tokens from many attention replicas into each expert.

</div>

Memory per GPU under expert parallelism with degree ⟦m956⟧:

⟦M951⟧

The shared part (attention, router, embeddings) is replicated; experts are split.

<div class="take">
<div class="lab">Key takeaway</div>

expert parallelism spreads whole experts across GPUs and moves tokens to them. It keeps matmuls larger than tensor parallelism does, but placement then decides balance and traffic.

</div>
