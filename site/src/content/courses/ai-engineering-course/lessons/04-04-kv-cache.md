---
id: "04.04"
course: "ai-engineering-course"
module: "04"
slug: "kv-cache"
title: "KV Cache"
keyIdeas: ["During decode, each new token needs the keys and values of all previous tokens at every layer.", "Recomputing them every step would make generation quadratic. The KV cache stores them once and appends one entry per new token.", "Size per token = 2 (K and V) x layers x KV heads x head dim x bytes per value.", "For long contexts and big batches, the KV cache can use more GPU memory than the weights. It limits batch size and context length.", "Ways to shrink it: grouped-query or multi-query attention (fewer KV heads), quantized caches (fp8/int8), sliding windows, multi-head latent attention.", "Prompt caching reuses the KV cache of a shared prefix (system prompt, documents) across requests."]
question: "Compute the KV cache size for one 8,000-token sequence on a model with 32 layers, 8 KV heads, head dim 128, in fp16."
answer: "Per token: 2 (K and V) x 32 layers x 8 KV heads x 128 head dim x 2 bytes = 131,072 bytes = 128 KB. For 8,000 tokens: 8,000 x 128 KB = 1,024,000 KB, about 1 GB (0.98 GiB) per sequence."
sources: [{"title": "Hugging Face: KV cache strategies", "url": "https://huggingface.co/docs/transformers/kv_cache", "note": "Dynamic, static, offloaded and quantized caches."}, {"title": "Pope et al.: Efficiently Scaling Transformer Inference (2022)", "url": "https://arxiv.org/abs/2211.05102", "note": "Memory and compute analysis of inference, including the KV cache."}, {"title": "Ainslie et al.: GQA (2023)", "url": "https://arxiv.org/abs/2305.13245", "note": "Grouped-query attention, fewer KV heads for a smaller cache."}]
---

## The repeated work

In decoding, the model produces one new token per step. Attention for that token needs a query for the new token, and keys and values for **every** previous token, at every layer. The previous tokens' K and V don't change. Without a cache, step t recomputes K and V for all t tokens, so generating n tokens costs O(n^2) extra work.

## The cache

Store each layer's K and V for every token once. At each decode step, compute q, k, v only for the new token, append k and v to the cache, and attend over the cached keys and values.

~~~dg-flow One decode step with a KV cache.
New token -(project)-> q, k, v -(append k, v)-> *KV cache (all previous tokens) -(attention with q)-> Output -(next layer)-> ...
~~~

Prefill fills the cache for the whole prompt in one pass; decode then reads it every step. This is why decode is memory-bandwidth bound: each step reads all weights and the whole cache for very little compute.

## How big it gets

```
bytes per token = 2 x layers x kv_heads x head_dim x bytes_per_value
```

<div class="ex">
<div class="lab">Worked example</div>

**Llama-3-8B-like model, 8,000 tokens**

2 x 32 layers x 8 KV heads x 128 dims x 2 bytes (fp16) = 131,072 bytes, about 128 KB per token.

8,000 tokens x 128 KB is about 1 GB for one sequence.

A batch of 32 such sequences needs about 32 GB of cache, two times the 16 GB of fp16 weights.

</div>

Pope et al.'s inference analysis shows this pattern: at large batch and long context, the KV cache, not the weights, sets how many requests fit on a GPU.

## Making it smaller

| Technique | How it shrinks the cache |
| --- | --- |
| Multi-query attention (MQA) | All query heads share one K/V head |
| Grouped-query attention (GQA) | Groups of query heads share K/V heads (Llama 3: 32 Q heads, 8 KV heads, 4x smaller) |
| KV quantization | Store K/V in fp8 or int8 instead of fp16 |
| Sliding-window attention | Keep only the last W tokens for some layers |
| Multi-head latent attention (DeepSeek) | Store a compressed latent and expand it on the fly |
| Paged attention (module 6) | Doesn't shrink entries, but removes wasted, fragmented space |

## Prompt caching

Many requests share a prefix: the same system prompt, the same long document. Servers and APIs can keep the KV cache for that prefix and reuse it, skipping its prefill. That cuts time to first token and cost, which is why API providers price cached input tokens much lower.

## Size the cache for your deployment

The KV cache stores the keys and values of every past token, in every layer, so decode doesn't recompute them. Its size is easy to compute and often surprising. Set a model and a workload below.

~~~dg-raw
%%widget kvcache KV cache memory per sequence and for a batch, next to the model weights.
~~~

The formula:

```
bytes per token = 2 (K and V) × layers × kv_heads × head_dim × bytes per value
```

For Llama 3 8B in bf16: 2 × 32 × 8 × 128 × 2 = 131,072 bytes, which is 128 KiB per token.

- One 8,192-token conversation: 1 GiB.
- 32 such conversations at once: 32 GiB, twice the size of the model weights.
- With full multi-head attention (32 KV heads instead of 8) it would be 4× larger. That is why GQA (05.02) became standard.

## Why the cache sets your throughput

Serving cost is mostly decided by how many sequences you can decode at once (the batch). Each decode step reads all the weights once regardless of batch size, so a bigger batch spreads that cost over more users. The batch size is capped by how many KV caches fit in the memory left after the weights. Shrinking the cache is therefore one of the most direct ways to cut cost per token:

| Technique | Cache reduction | Cost |
| --- | --- | --- |
| GQA (8 KV heads instead of 32) | 4× | Built into the model; slight quality change |
| FP8 KV cache | 2× versus bf16 | Small accuracy loss on some tasks |
| Sliding window (07.02) | Bounded by window size | Can't attend beyond the window directly |
| Paged allocation (06.01) | Removes wasted reservation | Engine support (vLLM, SGLang, TensorRT-LLM) |

## Prompt caching from the outside

Hosted APIs reuse the cache across requests that share a prefix. A 10,000-token system prompt plus tool definitions, sent on every call, is processed once and then read from cache. Providers typically bill cached input tokens at a fraction of the normal price. To benefit, keep the stable part of the prompt first and byte-identical; put anything that changes per request (the date, the user's name) after it.

<div class="take">
<div class="lab">Key takeaway</div>

the KV cache trades memory for compute by storing past keys and values. At long context and high batch it dominates GPU memory, which is what GQA, KV quantization and paging attack.

</div>
