---
id: "04.03"
course: "ai-engineering-course"
module: "04"
slug: "transformer-architecture"
title: "Transformer architecture"
keyIdeas: ["Modern LLMs are decoder-only transformers: a stack of identical blocks between an embedding layer and an output layer.", "Each block: normalisation, causal multi-head self-attention, residual add, normalisation, feed-forward network (MLP), residual add.", "Residual connections let each block add a small update to a running residual stream and make deep stacks trainable; pre-norm with RMSNorm is now standard.", "The MLP holds most of the weights: two thirds in a classic block, about 80% in Llama 3 8B with GQA and SwiGLU. Much factual recall seems to live there.", "Modern variants keep the skeleton and swap parts: RoPE, GQA or MLA, QK-norm, mixture-of-experts MLPs, local/global and linear-attention layers."]
question: "Draw (in words) one decoder block of a modern LLM like Llama, naming each sub-layer in order and where the residual connections go."
answer: "Input from the residual stream, then RMSNorm, then causal multi-head self-attention (RoPE applied to queries and keys, usually with grouped-query attention), then add the attention output back to the residual stream. Then RMSNorm again, then the feed-forward MLP (SwiGLU: gate and up projections, elementwise gating, down projection), then add its output back to the stream, which passes to the next block. Residual connections wrap both sub-layers, so each block only adds corrections to the stream rather than replacing it, and the norms sit before each sub-layer (pre-norm). After the last block a final norm and the LM head turn each position into logits over the vocabulary."
sources: [{"title": "Jay Alammar: The Illustrated GPT-2", "url": "https://jalammar.github.io/illustrated-gpt2/", "note": "Decoder-only transformer, block by block."}, {"title": "Andrej Karpathy: nanoGPT", "url": "https://github.com/karpathy/nanoGPT", "note": "About 300 lines of GPT model code you can read in one sitting."}, {"title": "Sebastian Raschka: The Big LLM Architecture Comparison", "url": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison", "note": "How DeepSeek, Llama, Qwen, Gemma and others differ."}]
rubric: ["Orders the sub-layers: norm, attention, residual add, norm, MLP, residual add", "Places a residual connection around both attention and the MLP", "Identifies pre-norm (normalisation before each sub-layer), RMSNorm in Llama", "Mentions causal masking and RoPE in attention", "Mentions the final norm and LM head producing logits"]
practice: ["transformer-decoder-block", "rmsnorm-forward-pass", "parameter-matched-swiglu-block", "qk-norm-vs-logit-softcap", "transformer-training-flop-estimator"]
---

## The overall shape

The 2017 transformer had an encoder and a decoder for translation. GPT-style LLMs keep only the decoder: a stack of $L$ identical blocks with causal self-attention, so each token can only see earlier tokens. Llama 3 8B has 32 blocks; the 405B model has 126.

~~~dg-flow A decoder-only LLM end to end.
Token ids -(embedding)-> Vectors -(L decoder blocks)-> *Final hidden states -(norm + LM head)-> Logits over vocabulary
~~~

## Inside one block

~~~dg-layers One pre-norm decoder block (Llama-style), bottom to top.
Residual stream in | token vectors of size d_model (4,096 in Llama 3 8B)
RMSNorm | normalise before attention
*Causal multi-head self-attention | with RoPE on Q and K; output added back to the stream
RMSNorm | normalise before the MLP
Feed-forward (SwiGLU MLP) | expand to ~3.5x d_model, then project back; added to the stream
Residual stream out | passed to the next block
~~~

Each block computes two residual updates:

$$
h = x + \text{Attention}(\text{RMSNorm}(x)), \qquad y = h + \text{MLP}(\text{RMSNorm}(h))
$$

**The residual stream.** The "+ x" means each sub-layer only adds a correction to a running vector. A 32-layer model is 32 small edits to the same vector, not 32 rewrites, and gradients flow straight through the additions, which is why 100+ layer stacks train at all.

**Attention vs MLP.** Attention moves information between positions, token to token ([Attention](lesson:aieng:04.02)). The MLP processes each position on its own; it holds most of the parameters, and interpretability work suggests much factual recall happens there.

**Normalisation.** The original transformer normalised after each sub-layer (post-norm). Modern models normalise before (pre-norm), which keeps the edits at a stable scale and trains more stably without careful learning-rate tricks. Llama uses RMSNorm, a cheaper LayerNorm without mean-centring.

**The output.** After the last block, a final norm and the LM head project the hidden state at each position to one logit per vocabulary entry. Some models tie the head to the input embedding table to save parameters; Llama 3 8B does not.

## Where the parameters are

For hidden size $d$ and $L$ layers, a classic block has about $4d^2$ attention weights (Q, K, V, output) and $8d^2$ in an MLP that expands to $4d$, so roughly $12Ld^2$ in total plus embeddings, with the MLP at two thirds. Modern models move that share further towards the MLP. Change the shape below and see where the weights live; the defaults are close to Llama 3 8B.

~~~dg-raw
%%widget params Where a transformer's parameters are: attention, MLP and embeddings.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**counting Llama 3 8B, then loading it**

Per layer, with $d = 4{,}096$, 32 query heads and 8 KV heads of dimension 128:

- **Attention**: $W_Q$ and $W_O$ are 4,096 × 4,096 (16.8M each). Grouped-query attention makes $W_K$ and $W_V$ only 4,096 × 1,024 (4.2M each). Total about 42M.
- **MLP (SwiGLU)**: three matrices of 4,096 × 14,336 (gate, up, down). Total about 176M.
- **Per layer**: about 218M, of which the MLP is 81%. Times 32 layers: about 7.0B.
- **Embeddings**: 128,256 × 4,096 = 525M for the input table and another 525M for the untied output head.
- **Total**: about 8.0B.

Loading it: 8B parameters × 2 bytes (bf16) = 16 GB. At 4-bit ([quantization formats](lesson:ieng:11.02)) that drops to about 4.5 GB including scales. A 70B model in bf16 is 140 GB: two 80 GB GPUs before a single token of KV cache. Parameter count tells you the minimum hardware before you run a single request.

</div>

The MLP's share is why mixture-of-experts models put their experts in the MLP ([MoE for app engineers](lesson:aieng:06.06)) and why quantization work focuses on MLP matrices.

## How modern models differ

Raschka's architecture comparison shows the 2017 skeleton survives, with a set of standard tweaks:

| Change | What it does | Examples |
| --- | --- | --- |
| RoPE | Relative positions by rotating Q and K ([Positional embeddings](lesson:aieng:04.01)) | Nearly all open models |
| Grouped-query or latent attention | Fewer or compressed K/V heads to shrink the KV cache ([Multi-head attention](lesson:aieng:05.02)) | Llama 3, Qwen3 (GQA); DeepSeek-V3 (MLA) |
| SwiGLU MLP | Gated activation; three matrices instead of two | Llama, Qwen, Mistral |
| QK-norm | Normalise queries and keys before the dot product for stable training | Qwen3, Gemma 3, OLMo 2 |
| Mixture-of-experts MLP | Many expert MLPs, a few active per token | DeepSeek-V3, Qwen3-MoE, gpt-oss, Llama 4 |
| Local/global attention | Most layers see only a recent window ([The KV cache](lesson:aieng:04.04)) | Gemma 3, gpt-oss |
| Linear-attention layers | Fixed-size state in most layers ([What attention changes](lesson:aieng:04.05)) | Qwen3-Next, Nemotron-H |

## Common mistakes

- **Parameters are not memory.** Weights are only the floor; the KV cache grows with context length and batch size and can exceed the weights.
- **Total vs active parameters.** An MoE model's total count sets memory; its active count sets compute per token. Comparing a 120B MoE with a 120B dense model on either alone misleads.
- **Assuming the two-thirds rule.** With GQA and a 3.5× SwiGLU MLP, the MLP is about 80% of each layer; estimate from the actual config.

<div class="take">
<div class="lab">Key takeaway</div>

An LLM is embeddings, then $L$ blocks of (norm, attention, add, norm, MLP, add), then a projection to logits. Attention mixes across tokens, the MLP transforms each token and holds most of the weights, and the residual stream holds it together.

</div>
