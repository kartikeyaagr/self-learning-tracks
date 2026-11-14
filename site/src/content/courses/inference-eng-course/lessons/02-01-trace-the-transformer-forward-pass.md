---
id: "02.01"
course: "inference-eng-course"
module: "02"
slug: "trace-the-transformer-forward-pass"
title: "Trace the Transformer Forward Pass"
keyIdeas: ["Forward pass: embed tokens, run N layers, normalize, project to vocabulary logits.", "Each layer is attention then MLP, each wrapped in a norm and a residual add.", "The LM head is a big matmul: hidden size x vocabulary (e.g. 4096 x 128k).", "In decode, only the newest token goes through the layers; earlier tokens are represented by cached K and V.", "Hooks or a profiler let you see every op and its time."]
question: "For one decode step of a 32-layer model, list the operations in order from token id to next-token logits."
answer: "Embedding lookup for the new token; then for each of 32 layers: RMSNorm, Q/K/V projections, RoPE on Q and K, append K and V to the cache, attention over all cached keys and values, output projection, residual add, RMSNorm, gate and up projections, SiLU and multiply, down projection, residual add; then a final RMSNorm and the LM head matmul to vocabulary logits."
sources: [{"title": "The Illustrated GPT-2", "url": "https://jalammar.github.io/illustrated-gpt2/", "note": "Visual walkthrough of a decoder forward pass."}, {"title": "Hugging Face Llama modeling code", "url": "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py", "note": "Reference implementation to trace."}]
tags: ["transformer"]
---

## From one id to one probability table

A forward pass takes token ids in and gives logits out. Between the two sit an embedding lookup, a stack of identical layers, a final norm and a projection onto the vocabulary. Knowing this path in detail lets you predict where time and memory go before you profile anything.

~~~dg-flow One forward pass, end to end. Press Play to follow a token through it.
Token id -(lookup)-> Embedding -(N times)-> *Decoder layer -> Final RMSNorm -(LM head)-> Logits
~~~

## The shapes, for Llama 3 8B

Track one token's hidden state through the network. Write B for batch, S for tokens in this pass and d for the hidden size.

| Stage | Output shape | Weights involved |
| --- | --- | --- |
| Embedding lookup | [B, S, 4096] | 128,256 × 4,096 table (525M params) |
| 32 decoder layers | [B, S, 4096] | about 218M params each |
| Final RMSNorm | [B, S, 4096] | 4,096 gains |
| LM head | [B, S, 128256] | 4,096 × 128,256 matrix (525M params) |

The embedding lookup is just an indexed read, so it costs almost nothing. The LM head is a full matmul with a very wide output, so it is one of the most expensive single ops in the model.

## Inside one decoder layer

Every layer applies the same nine steps:

1. `x = rmsnorm(h)`: normalize the residual stream.
2. `q, k, v = x @ Wq, x @ Wk, x @ Wv`: three projections. With GQA, `k` and `v` are narrower than `q`.
3. Apply rotary position embeddings (RoPE) to `q` and `k`, so attention knows the position of each token.
4. Write `k` and `v` into the KV cache for this layer.
5. `a = attention(q, K_cache, V_cache)`, then `a = a @ Wo`.
6. `h = h + a`: the first residual add.
7. `x = rmsnorm(h)`.
8. `m = (silu(x @ Wgate) * (x @ Wup)) @ Wdown`: the SwiGLU MLP.
9. `h = h + m`: the second residual add.

Attention mixes information between positions. The MLP transforms each position on its own. The residual stream `h` carries information from layer to layer.

## Where the parameters live

For one Llama 3 8B layer, with d = 4,096, 32 query heads, 8 KV heads, head size 128 and MLP size 14,336:

- **Wq**: 4,096 × 4,096 = 16.8M
- **Wk** and **Wv**: 4,096 × 1,024 each = 8.4M together
- **Wo**: 4,096 × 4,096 = 16.8M
- **Wgate**, **Wup**, **Wdown**: 3 × 4,096 × 14,336 = 176M

The MLP holds about 80% of each layer's weights. Attention projections hold about 19%. Norms are a rounding error.

## Prefill and decode trace the same path differently

- In **prefill**, S is the prompt length, often hundreds or thousands of tokens. Every matmul is a large matrix-matrix multiply. Attention compares every prompt token to every earlier prompt token. The GPU's tensor cores are busy.
- In **decode**, S is 1 for each sequence. Each matmul multiplies a few rows by a large weight matrix. Attention compares one new query against the cached keys. The GPU spends most of its time reading weights and cache from memory.

The code is identical. Only the shapes change, and the shapes decide whether you are compute bound or memory bound.

<div class="ex">
<div class="lab">Worked example</div>

**FLOPs for one decode token**

A matmul of a [1, d_in] row with a [d_in, d_out] weight costs 2 × d_in × d_out FLOPs.

Per layer: Q, K, V and O projections cost 2 × 42M = 84 MFLOP. The MLP costs 2 × 176M = 352 MFLOP. That is about 436 MFLOP per layer, or 14 GFLOP for 32 layers.

The LM head adds 2 × 525M = 1.05 GFLOP. Total: about 15 GFLOP per token, close to the "2 × parameters" rule (2 × 8B = 16 GFLOP; the embedding table is a lookup, not a matmul).

Attention over a 4,000-token cache adds 2 × 2 × 32 layers × 4,096 × 4,000 ≈ 2.1 GFLOP. Small next to the matmuls at this length, but it grows with every token.

</div>

## Where time goes in practice

Trace a batch-1 decode step with the PyTorch profiler and you will usually see:

| Op group | Share of step time (short context) | Why |
| --- | --- | --- |
| MLP matmuls | about 55 to 60% | largest weights to read |
| Attention projections | about 20 to 25% | smaller weights thanks to GQA |
| LM head | about 5 to 10% | one wide matmul |
| Attention kernel | 3 to 10%, growing with context | reads the KV cache |
| Norms, RoPE, adds, sampling | 5 to 15% | many tiny kernels, launch overhead |

The last row is where batch-1 servers lose a surprising amount of time. Each tiny op needs its own kernel launch, and the GPU idles between launches. CUDA graphs and kernel fusion, covered in Module 15, remove most of it.

## Deep dive: the forward pass as equations

For a sequence of token ids ⟦m223⟧, with embedding matrix ⟦m224⟧:

⟦M219⟧

Each of the ⟦m225⟧ layers applies two residual sub-blocks:

⟦M220⟧

⟦M221⟧

Then a final norm and the LM head produce logits over the vocabulary for the last position:

⟦M222⟧

The hidden state ⟦m226⟧ keeps the same shape ⟦m227⟧ through every layer. Each layer reads and adds to it; this "residual stream" is why layers can be stacked so deep.

~~~dg-raw
%%bars Where the FLOPs go for one decode token in Llama 3 8B at 4k context (GFLOPs).
unit = GFLOP
MLP matmuls (32 layers) | 11.27 | c3
Q, K, V, O matmuls (32 layers) | 2.68 | c1
LM head | 1.05 | c2
Attention scores and values (4k context) | 2.15 | flame
Norms, RoPE, softmax, residual adds | 0.02 | c4
%%
~~~

## Deep dive: the residual stream's size in memory

During decode, the hidden state for one token is just ⟦m228⟧ numbers, 8 KB in BF16. For a batch of 256 it is 2 MB. Activations during decode are tiny compared with weights (16 GB) and KV. During prefill of 8,192 tokens, though, the MLP's intermediate ⟦m229⟧ tensor is 235 MB per layer in BF16, which is why engines cap the number of tokens processed per step.

<div class="take">
<div class="lab">Key takeaway</div>

a forward pass is embed, N layers of attention plus MLP, a final norm and the LM head. The MLP holds most of the weights. Prefill and decode run the same code with different shapes, and the shapes decide the bottleneck.

</div>
