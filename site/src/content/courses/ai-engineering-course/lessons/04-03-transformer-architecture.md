---
id: "04.03"
course: "ai-engineering-course"
module: "04"
slug: "transformer-architecture"
title: "Transformer architecture"
keyIdeas: ["Modern LLMs are decoder-only transformers: a stack of identical blocks between an embedding layer and an output layer.", "Each block: normalisation, self-attention (causal, multi-head), residual add, normalisation, feed-forward network (MLP), residual add.", "Residual connections let each block add a small update to a running 'residual stream' and make deep stacks trainable.", "Pre-norm (normalise before each sub-layer) is now standard; Llama uses RMSNorm.", "The MLP is usually about 2/3 of parameters, often with a SwiGLU activation. It's where much factual knowledge seems to live.", "The final layer projects the last hidden state to vocabulary logits (often sharing weights with the input embedding)."]
question: "Draw (in words) one decoder block of a modern LLM like Llama, naming each sub-layer in order and where the residual connections go."
answer: "Input from the residual stream -> RMSNorm -> causal multi-head self-attention (with RoPE on queries/keys) -> add the output back to the residual stream -> RMSNorm -> feed-forward MLP (SwiGLU) -> add back to the residual stream -> next block. Residual connections wrap both the attention and the MLP sub-layers; after the last block a final norm and the LM head produce logits."
sources: [{"title": "Jay Alammar: The Illustrated GPT-2", "url": "https://jalammar.github.io/illustrated-gpt2/", "note": "Decoder-only transformer, block by block."}, {"title": "Andrej Karpathy: nanoGPT", "url": "https://github.com/karpathy/nanoGPT", "note": "About 300 lines of GPT model code you can read in one sitting."}, {"title": "Sebastian Raschka: The Big LLM Architecture Comparison", "url": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison", "note": "How DeepSeek, Llama, Qwen, Gemma and others differ."}]
---

## The overall shape

The 2017 transformer had an encoder and a decoder for translation. GPT-style LLMs keep only the decoder: a stack of N identical blocks with causal self-attention, so each token can only see earlier tokens. Llama 3 8B has 32 blocks; the 405B model has 126.

~~~dg-flow A decoder-only LLM end to end.
Token ids -(embedding)-> Vectors -(N decoder blocks)-> *Final hidden states -(norm + LM head)-> Logits over vocabulary
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

**Residual stream**
Each sub-layer reads the stream, computes an update, and **adds** it back: x = x + Attention(Norm(x)), then x = x + MLP(Norm(x)). Gradients flow straight through the additions, which is why 100+ layer stacks train at all.

**Attention vs MLP**

- **Attention** moves information between positions (token to token).
- **MLP** processes each position on its own. It holds most of the parameters (about two thirds), and interpretability work suggests much factual recall happens here.

**Normalisation**
The original transformer normalised after each sub-layer (post-norm). Modern models normalise before (pre-norm), which trains more stably. Llama uses RMSNorm, a cheaper LayerNorm without mean-centering.

## Where the parameters are

For a model with hidden size d and L layers, each block has about 4d^2 attention weights (Q, K, V, output) and about 8d^2 in the MLP. So total parameters are roughly 12 L d^2, plus embeddings. For d = 4,096 and L = 32 that's about 6.4B, close to Llama's 8B once embeddings and a wider MLP are counted.

## How modern models differ

Raschka's architecture comparison shows the 2017 skeleton survives, with tweaks:

- RoPE instead of absolute positions.
- Grouped-query attention (fewer K/V heads) to shrink the KV cache.
- SwiGLU activations.
- Mixture-of-experts MLPs (module 6) in DeepSeek, Mixtral, Qwen MoE.
- Sliding-window or mixed local/global attention (Gemma, Mistral).

## Count the parameters yourself

The figure computes parameter counts from the shape of a model. The defaults are close to Llama 3 8B. Change the hidden size, layers, heads or MLP width and see where the weights live.

~~~dg-raw
%%widget params Where a transformer's parameters are: attention, MLP and embeddings.
~~~

Here is the arithmetic for the default, per layer:

- **Attention**: W_q is 4,096 × 4,096 (16.8 M). With 8 KV heads of dim 128, W_k and W_v are 4,096 × 1,024 each (4.2 M each). W_o is 4,096 × 4,096 (16.8 M). Total about 42 M.
- **MLP (SwiGLU)**: three matrices of 4,096 × 14,336: gate, up and down. Total about 176 M.
- **Per layer**: about 218 M. Times 32 layers: about 7.0 B.
- **Embeddings**: 128,256 × 4,096 = 525 M for the input table and another 525 M for the untied output head.
- **Grand total**: about 8.0 B.

The MLP holds about 80% of the per-layer weights. That is why mixture-of-experts (06.02) puts its experts in the MLP, and why quantization papers focus on MLP matrices.

## Pre-norm and residuals, concretely

Each block computes:

```
h = x + Attention(RMSNorm(x))
y = h + MLP(RMSNorm(h))
```

The residual "+ x" means each block only adds a correction to a running vector (the residual stream). A 32-layer model is 32 small edits to the same vector, not 32 rewrites. Normalizing before each sub-layer (pre-norm) keeps those edits at a stable scale, which is what lets very deep stacks train without careful learning-rate tricks.

<div class="ex">
<div class="lab">Worked example</div>

**memory needed just to load the weights**

8 B parameters × 2 bytes (bf16) = 16 GB. At int4 (07.01) that drops to about 4.5 GB including scales.

A 70 B model in bf16 is 140 GB: two 80 GB GPUs before any KV cache. Parameter count tells you the minimum hardware before you run a single request.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

an LLM is embeddings, then N blocks of (norm, attention, add, norm, MLP, add), then a projection to logits. Attention mixes across tokens; the MLP transforms each token; residual adds hold it together.

</div>
