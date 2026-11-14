---
id: "00.06"
course: "inference-eng-course"
module: "00"
slug: "transformer-architecture-refresher"
title: "Transformer Architecture Refresher"
keyIdeas: ["LLMs are decoder-only transformers: embeddings, N identical layers, final norm, LM head.", "Each layer has attention (mix information across tokens) and an MLP (transform each token independently).", "Modern variants use RMSNorm, rotary position embeddings (RoPE), gated MLPs (SwiGLU) and grouped-query attention (GQA).", "MLP weights dominate parameter count; attention cost and KV memory grow with sequence length.", "Residual connections let each layer add to a running hidden state instead of replacing it."]
question: "Sketch one decoder layer of a modern LLM like Llama and say which part dominates parameters and which part grows with context length."
answer: "Input goes through RMSNorm, then self-attention (Q, K, V projections, RoPE on Q and K, attention, output projection), plus a residual add; then RMSNorm, a gated MLP (gate, up, down projections with SiLU), plus a residual add. The MLP holds most parameters (around two thirds). Attention's KV cache and attention compute grow with context length."
sources: [{"title": "The Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/", "note": "Visual walkthrough of the architecture."}, {"title": "Llama 2 paper", "url": "https://arxiv.org/abs/2307.09288", "note": "A concrete modern decoder-only architecture."}, {"title": "nanoGPT", "url": "https://github.com/karpathy/nanoGPT", "note": "Short readable GPT implementation."}]
tags: ["transformer"]
---

## The decoder-only transformer

Every major open LLM (Llama, Mistral, Qwen, DeepSeek, Gemma) is a decoder-only transformer. The structure is a stack of identical blocks between an embedding layer and an output head:

~~~dg-flow The path of one token through the model.
Token id -(lookup)-> Embedding -(×L blocks)-> *Transformer block -(final norm)-> LM head -(softmax)-> Next-token probs
~~~

## Inside one block

Each block has two sub-layers, each wrapped in a residual connection with a normalization before it (pre-norm):

1. `h = x + Attention(RMSNorm(x))`
2. `out = h + MLP(RMSNorm(h))`

The residual connections mean each block adds a correction to a running "residual stream" rather than replacing it. That is why you can skip, prune or quantize some layers and the model still works roughly.

## The components

- **Embedding**: a table of shape `[vocab, hidden]`. Token id 9906 picks row 9906. For Llama 3 8B that table is 128,256 × 4,096 ≈ 525M parameters.
- **RMSNorm**: divides the vector by its root-mean-square and multiplies by a learned scale. Cheaper than LayerNorm (no mean subtraction) and works as well.
- **Attention**: Q, K, V projections, rotary position embeddings (RoPE) applied to Q and K, causal attention, output projection. Uses grouped-query attention (GQA) in most modern models.
- **MLP**: a gated unit (SwiGLU): `down(silu(gate(x)) × up(x))`. It holds about two thirds of each block's parameters.
- **LM head**: a `[hidden, vocab]` matrix producing logits. Sometimes tied to the embedding table.

## Where the parameters are

| Part (Llama 3 8B) | Parameters | Share |
| --- | --- | --- |
| Embedding | 0.53B | 6.5% |
| Attention projections, 32 layers | 1.34B | 16.7% |
| MLP, 32 layers | 5.64B | 70.2% |
| LM head | 0.53B | 6.5% |
| Norms | tiny | 0% |

<div class="ex">
<div class="lab">Worked example</div>

**counting one layer**

Attention: Q 4096×4096 + K 4096×1024 + V 4096×1024 + O 4096×4096 = 41.9M.

MLP: gate 4096×14336 + up 4096×14336 + down 14336×4096 = 176.2M.

Per layer ≈ 218M. Times 32 layers ≈ 6.98B. Plus embedding and head (1.05B) ≈ 8.03B. That is the "8B".

</div>

## RoPE in one paragraph

Attention by itself is order-blind. RoPE encodes position by rotating pairs of dimensions in Q and K by an angle proportional to the token's position. The dot product between a query at position m and a key at position n then depends on m − n, the relative distance. RoPE is applied after the K projection, so cached keys already contain position, which matters when you reuse cached prefixes.

## Grouped-query attention

In multi-head attention (MHA), every query head has its own K and V heads. In GQA, several query heads share one K/V head. Llama 3 8B has 32 query heads and 8 KV heads, so the KV cache is 4× smaller than MHA with negligible quality loss. Multi-query attention (MQA) goes further with one KV head. DeepSeek's multi-head latent attention (MLA) compresses K and V into a small latent vector. All of these exist to shrink the KV cache, because that is what limits concurrency.

## What is the same in prefill and decode

The weights and operations are identical. What changes is how many tokens flow through at once and how much past context attention reads. That difference, not the architecture, drives inference performance.

## Deep dive: counting Llama 3 8B's parameters

For hidden size ⟦m140⟧, ⟦m141⟧ query heads, ⟦m142⟧ KV heads of size ⟦m143⟧, MLP width ⟦m144⟧, ⟦m145⟧ layers and vocabulary ⟦m146⟧:

⟦M136⟧

⟦M137⟧

With ⟦m147⟧, ⟦m148⟧, ⟦m149⟧, ⟦m150⟧, ⟦m151⟧, ⟦m152⟧:

- ⟦m153⟧
- ⟦m154⟧
- 32 layers: ⟦m155⟧
- Embedding and LM head: ⟦m156⟧
- Total: about 8.03 billion. Norm weights add only a few hundred thousand.

~~~dg-raw
%%bars Llama 3 8B parameters by component (billions).
unit = B
MLP (32 layers) | 5.64 | c3
Attention (32 layers) | 1.34 | c1
Embedding | 0.53 | c2
LM head | 0.53 | c2
%%
~~~

Two thirds of the weights are in the MLPs. That is why MLP quantization and MLP kernels matter most for weight bandwidth.

## Deep dive: FLOPs per token, including attention

⟦M138⟧

The first term covers all matmuls with weights, including the LM head (about 7.5B parameters); the embedding lookup is a table read, not a matmul. The second is attention against ⟦m157⟧ earlier tokens. For Llama 3 8B at 8k context: ⟦m158⟧ GFLOPs plus ⟦m159⟧ GFLOPs. Attention adds about 29% at 8k, and grows linearly with context.

## Deep dive: RoPE as a rotation

RoPE splits each query and key into pairs of dimensions and rotates pair ⟦m160⟧ at position ⟦m161⟧ by angle ⟦m162⟧, with ⟦m163⟧ and base ⟦m164⟧ (500,000 for Llama 3):

⟦M139⟧

Because rotations compose, the dot product of a query at position ⟦m165⟧ and a key at position ⟦m166⟧ depends only on ⟦m167⟧. That gives relative position information without any extra parameters, and it is why the KV cache can store keys already rotated: their rotation never changes.

<div class="take">
<div class="lab">Key takeaway</div>

a decoder-only LLM is embedding, L identical pre-norm blocks of attention plus MLP, and an LM head. The MLP holds most of the weights; attention holds most of the per-request memory.

</div>
