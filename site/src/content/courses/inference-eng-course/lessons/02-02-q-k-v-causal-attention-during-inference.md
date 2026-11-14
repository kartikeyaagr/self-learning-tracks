---
id: "02.02"
course: "inference-eng-course"
module: "02"
slug: "q-k-v-causal-attention-during-inference"
title: "Q, K, V & Causal Attention During Inference"
keyIdeas: ["Query: what this token is looking for. Key: what each token offers. Value: what each token passes on.", "Causal masking means a token only sees itself and earlier tokens.", "Because of the mask, a past token's K and V never change. That makes the KV cache possible.", "Decode attention is one query against many keys: a matrix-vector product, memory bound.", "Multi-query and grouped-query attention share K/V heads to shrink the cache."]
question: "In decode, why do we only compute Q for the new token, but need K and V for every previous token?"
answer: "The new token is the only one asking a question about the past, so it alone needs a query. Its answer is a weighted mix of all earlier tokens' values, weighted by how its query matches their keys, so all past K and V are required. Past tokens' K and V do not change when new tokens arrive because of the causal mask, which is why they can be cached."
sources: [{"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762", "note": "Original attention formulation."}, {"title": "GQA: Training Generalized Multi-Query Transformer Models", "url": "https://arxiv.org/abs/2305.13245", "note": "Sharing K and V heads to cut memory."}]
tags: ["attention"]
---

## Three projections of every token

Attention is how a transformer lets each token use information from earlier tokens. It starts by projecting every token's hidden state three ways:

- **Query (Q)**: what this token is looking for.
- **Key (K)**: a description of what this token offers.
- **Value (V)**: the content this token passes along if someone attends to it.

For one head, the output for token t is a weighted average of the values of tokens 0 to t. The weights come from comparing t's query with each key:

scores = q_t · K / √head_dim
weights = softmax(scores)
output = weights · V

The √head_dim scaling keeps scores in a sensible range so softmax does not saturate.

## Many heads

A model runs many heads in parallel, each with its own projections. Llama 3 8B has 32 query heads of size 128. Different heads learn to track different relationships, such as the previous word, the subject of the sentence or a matching bracket. The head outputs are concatenated and projected back with Wo.

## The causal mask

In a decoder-only model, token t may only look at positions 0 to t. Before softmax, every score for a future position is set to minus infinity, so its weight becomes zero. This is the **causal mask**.

The mask has a consequence that makes efficient inference possible. Token t's key and value depend only on token t and the tokens before it. Adding new tokens later never changes them. So once you compute K and V for a position, you can store them and reuse them for every later step. That stored data is the **KV cache**.

## Attention during decode

At decode step t, only the new token needs a query. It attends over the stored keys and values for positions 0 to t.

~~~dg-flow Decode attention for one head.
New q [1×128] -(dot)-> K cache [t×128] -> Scores [1×t] -(softmax)-> *Weights -(mix)-> V cache [t×128]
~~~

This is a matrix-vector product: one query row against t keys, then t weights against t values. Count the work:

- **FLOPs**: about 4 × t × head_dim per head (two dot-product passes).
- **Bytes**: the whole K and V for this head, 2 × t × head_dim × 2 bytes in BF16.

That is about one FLOP per byte. A GPU like the H100 needs about 300 FLOPs per byte to be compute bound. Decode attention is therefore firmly memory bound, and its time grows linearly with context length.

<div class="ex">
<div class="lab">Worked example</div>

**decode attention cost at long context**

One Llama 3 8B sequence at 32,000 tokens has a KV cache of 32,000 × 128 KB = 4 GB.

Every decode step must read those 4 GB. At 3.35 TB/s that takes about 1.2 ms, on top of about 5 ms to read the 16 GB of weights.

At 128,000 tokens the cache is 16 GB, the same size as the weights, and attention time roughly equals matmul time. Long-context serving is a KV cache problem.

</div>

## Attention during prefill

In prefill, all prompt positions have queries at once. Attention becomes a matrix-matrix product: [S, head_dim] queries against [S, head_dim] keys, giving an S × S score matrix (half of it masked). The FLOPs grow with S², but so does the arithmetic per byte, so prefill attention is compute bound.

Materializing the full S × S matrix would use huge amounts of memory at long prompts: 32,000² scores × 2 bytes = 2 GB per head. FlashAttention avoids this by computing attention in tiles and never writing the full matrix to memory. Module 13 covers how.

## Making K and V smaller

Because decode reads the KV cache every step, shrinking it speeds up decode and fits more users. Models do this by using fewer key and value heads than query heads.

| Variant | KV heads (for 32 query heads) | KV size relative to MHA | Used in |
| --- | --- | --- | --- |
| Multi-head attention (MHA) | 32 | 1 | Llama 2 7B, GPT-2 |
| Grouped-query attention (GQA) | 8 | 1/4 | Llama 3, Mistral, Qwen2.5 |
| Multi-query attention (MQA) | 1 | 1/32 | PaLM, Falcon |
| Multi-head latent attention (MLA) | compressed latent | about 1/10 to 1/30 | DeepSeek V2 and V3 |

In GQA, each group of query heads shares one key head and one value head. With 32 query heads and 8 KV heads, groups of four query heads read the same cached K and V. Quality stays close to MHA while the cache shrinks four times. MQA saves more memory but tends to lose quality, and it cannot be split cleanly across many GPUs with tensor parallelism, since there is only one KV head to split.

~~~dg-raw
%%widget kv Compare Llama 2 7B (no GQA) with Llama 3 8B at the same context to see what GQA saves.
model = Llama 2 7B (no GQA)
gpu = H100 SXM (80 GB)
t = 13
b = 8
%%
~~~

## Deep dive: attention in matrix form

For one head with ⟦m232⟧:

⟦M230⟧

⟦m233⟧ is the causal mask: 0 where key position ⟦m234⟧ query position ⟦m235⟧, and ⟦m236⟧ elsewhere, so those entries become 0 after softmax.

~~~dg-raw
%%grid The causal mask for 6 tokens. Row = query position, column = key position.
cols = k1, k2, k3, k4, k5, k6
key A = allowed (score kept) | c3
key X = masked (−∞) | line
q1 | A X X X X X
q2 | A A X X X X
q3 | A A A X X X
q4 | A A A A X X
q5 | A A A A A X
q6 | A A A A A A
%%
~~~

During decode, only the last row is new. Rows 1 to 5 were already computed in earlier steps, and their K and V are in the cache. So a decode step computes one row of ⟦m237⟧, length ⟦m238⟧, per head.

## Deep dive: grouped-query attention, exactly

With ⟦m239⟧ query heads and ⟦m240⟧ KV heads, each group of ⟦m241⟧ query heads shares one K and V:

⟦M231⟧

~~~dg-raw
%%grid GQA in Llama 3 8B: 32 query heads (columns) map to 8 KV heads (colors). Shown for the first 16.
cols = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
key A = KV head 0 | c1
key B = KV head 1 | c2
key C = KV head 2 | c3
key D = KV head 3 | go
query heads | A A A A B B B B C C C C D D D D
%%
~~~

KV memory falls by ⟦m242⟧, and decode attention reads ⟦m243⟧ times fewer bytes. Compute does not change: every query head still does its own dot products. That is why GQA raises the arithmetic intensity of decode attention from 1 to about ⟦m244⟧.

<div class="ex">
<div class="lab">Worked example</div>

**the bytes read per decode step at 32k context**

Llama 2 7B (no GQA): 512 KB per token × 32,768 = 16 GB of KV, as much as the weights.

Llama 3 8B (GQA, g = 4): 128 KB × 32,768 = 4 GB. Step time at batch 1 falls from about ⟦m245⟧ ms to about ⟦m246⟧ ms, despite the larger model.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

the causal mask means past keys and values never change, so they can be cached. Decode attention is one query reading the whole cache, which is memory bound and grows with context. GQA and MLA shrink the cache and speed up decode.

</div>
