---
id: "04.04"
course: "ai-engineering-course"
module: "04"
slug: "kv-cache"
title: "KV Cache"
keyIdeas: ["During decode, each new token needs the keys and values of all previous tokens at every layer; the KV cache stores them once and appends one entry per token.", "Size per token = 2 (K and V) x layers x KV heads x head dim x bytes per value. At long context and high batch it can exceed the weights and caps how many requests fit.", "Fewer or compressed KV heads (GQA, MQA, MLA) and lower-precision caches shrink every entry.", "Sliding-window layers cap the cache at a fixed window; attention sinks keep the first few tokens so windowed attention stays stable.", "Sparse attention (NSA, DeepSeek Sparse Attention) reads only selected tokens per step: it cuts attention compute and memory traffic at long context, not the stored cache."]
question: "Compute the KV cache size for one 8,000-token sequence on a model with 32 layers, 8 KV heads, head dim 128, in fp16."
answer: "Per token: 2 (K and V) x 32 layers x 8 KV heads x 128 head dim x 2 bytes = 131,072 bytes = 128 KiB. For 8,000 tokens: 8,000 x 131,072 bytes = 1.05 GB, about 0.98 GiB per sequence. That is per sequence, so a batch of 32 such conversations needs about 31 GiB of cache, roughly twice the 16 GB of bf16 weights of an 8B model, and the cache rather than the weights decides how many requests fit on the GPU. With full multi-head attention (32 KV heads) it would be four times larger; an fp8 cache would halve it."
sources: [{"title": "Pope et al.: Efficiently Scaling Transformer Inference (2022)", "url": "https://arxiv.org/abs/2211.05102", "note": "Memory and compute analysis of inference, including the KV cache."}, {"title": "Xiao et al.: Efficient Streaming Language Models with Attention Sinks (2023)", "url": "https://arxiv.org/abs/2309.17453", "note": "StreamingLLM: keep a few initial tokens plus a recent window."}, {"title": "Yuan et al.: Native Sparse Attention (DeepSeek, 2025)", "url": "https://arxiv.org/abs/2502.11089", "note": "Trainable, hardware-aligned sparse attention: compress, select, slide."}, {"title": "DeepSeek-V3.2-Exp (GitHub)", "url": "https://github.com/deepseek-ai/DeepSeek-V3.2-Exp", "note": "DeepSeek Sparse Attention: a lightning indexer picks the tokens each query attends to."}]
practice: ["kv-cache-memory-for-mha-mqa-gqa-and-mla", "decode-attention-with-kv-cache", "sliding-window-causal-mask", "hybrid-cache-memory", "attention-with-sink-logits"]
---

## The repeated work

In decoding, the model produces one new token per step. Attention for that token needs its query and the keys and values of **every** previous token, at every layer. Those earlier keys and values never change. Without a cache, step $t$ recomputes K and V for all $t$ tokens, so generating $n$ tokens repeats $O(n^2)$ work.

## The cache

Store each layer's K and V for every token once. At each decode step, compute $q, k, v$ only for the new token, append $k$ and $v$ to the cache, and attend over the cached keys and values.

~~~dg-flow One decode step with a KV cache.
New token -(project)-> q, k, v -(append k, v)-> *KV cache (all previous tokens) -(attention with q)-> Output -(next layer)-> ...
~~~

Prefill fills the cache for the whole prompt in one parallel pass; decode then reads it every step. That is why decode is memory-bandwidth bound: each step reads all the weights and the whole cache to do very little arithmetic.

## How big it gets

$$
\text{bytes per token} = 2 \times \text{layers} \times \text{KV heads} \times \text{head dim} \times \text{bytes per value}
$$

For Llama 3 8B in bf16: $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes, 128 KiB per token. One 8,192-token conversation is 1 GiB; 32 of them at once are 32 GiB, twice the 16 GB of weights. Set a model and a workload below.

~~~dg-raw
%%widget kvcache KV cache memory per sequence and for a batch, next to the model weights.
~~~

**Why this sets your throughput.** Each decode step reads the weights once whatever the batch size, so a bigger batch spreads that cost over more users. The batch is capped by how many caches fit in the memory left after the weights. Pope et al.'s inference analysis shows the pattern: at large batch and long context, the KV cache, not the weights, sets how many requests fit on a GPU. Shrinking the cache is one of the most direct ways to cut cost per token.

## Architecture choices that shrink the cache

Some fixes shrink every entry, some cap how many entries a layer keeps, and some reduce how many entries each step reads.

| Choice | What it changes | Effect | Used in |
| --- | --- | --- | --- |
| Grouped-query / multi-query attention | Query heads share K/V heads | 4× smaller with 32 Q and 8 KV heads; 32× with MQA | Llama 3, Qwen3, Mistral |
| Multi-head latent attention | Cache one compressed latent per token, expand on the fly | Several times smaller than GQA at similar quality | DeepSeek-V3, Kimi K2 |
| Lower-precision cache | Store K/V in fp8 or 4-bit | 2× (fp8) versus bf16 | vLLM, SGLang, TensorRT-LLM option |
| Sliding-window layers | Layer keeps only the last $W$ tokens | Cache per layer capped at $W$ | Mistral 7B, Gemma 3, gpt-oss |
| Attention sinks | Keep the first few tokens as well as the window | Makes windowed attention stable on endless streams | StreamingLLM, gpt-oss |
| Sparse attention | Each query reads only selected tokens or blocks | Less compute and traffic per step; full cache still stored | DeepSeek NSA, DeepSeek-V3.2 |

GQA, MQA and MLA are covered in [Multi-head attention](lesson:aieng:05.02); fp8 and 4-bit caches in [quantization formats](lesson:ieng:11.02).

**Sliding windows.** Each token attends to its previous $W$ tokens, so the cache for that layer becomes a fixed-size ring buffer however long the conversation grows. Information can still travel further, one window per layer: with $W = 4{,}096$ and 32 layers the theoretical reach is about 131k tokens, but a detail relayed through a dozen hops is mostly lost. So current models interleave: Gemma 3 uses five local layers (window 1,024) for every global one, and gpt-oss alternates full layers with 128-token banded layers. The few global layers keep exact long-range recall; the many local ones keep the cache small. Pick a pattern below to see which query-key pairs each one computes, and stack layers to see how far information can travel.

~~~dg-raw
%%widget sparse Sliding-window, dilated, global-token and block-sparse attention patterns.
~~~

**Attention sinks.** Xiao et al. (StreamingLLM, 2023) found that a pure sliding window collapses once the first tokens are evicted. Softmax weights must sum to 1, and models learn to park spare attention on the first few tokens, which act as a sink. Keep about four initial tokens plus the recent window and the model stays stable over millions of tokens of streaming, without retraining. gpt-oss builds the idea in as a learned per-head sink logit. A sink keeps the model fluent; it does not let it recall what fell out of the window.

**Sparse attention.** Native Sparse Attention (DeepSeek, 2025) gives each query three branches: compressed summaries of distant blocks, a few blocks selected by relevance, and a local window, all shaped to run fast on GPUs and trained end to end. DeepSeek Sparse Attention in DeepSeek-V3.2 uses a small, cheap "lightning indexer" to score past tokens and lets each query attend only to its top 2,048, on top of MLA. Both cut attention from $O(n^2)$ towards $O(nk)$ at long context; DeepSeek cut its API prices when it shipped the first DSA model. They still store every token's cache entry, because a later query might select it.

<div class="ex">
<div class="lab">Worked example</div>

**128k tokens, full versus interleaved windows (illustrative mix)**

Per layer, per token, the Llama-3-8B-like model stores $2 \times 8 \times 128 \times 2 = 4{,}096$ bytes.

- **All 32 layers global** at 131,072 tokens: $32 \times 131{,}072 \times 4$ KiB = 16 GiB for one sequence.
- **8 global layers, 24 windowed at 4,096**: global $8 \times 131{,}072 \times 4$ KiB = 4 GiB, plus local $24 \times 4{,}096 \times 4$ KiB = 384 MiB. Total about 4.4 GiB, 3.7× smaller.

On an 80 GB GPU with 16 GB of weights, that is the difference between 3 and 13 concurrent 128k-token sequences. The window layers' share stays at 384 MiB however long the context grows; the global layers still grow linearly, which is why the ratio of local to global layers is the real design knob.

</div>

## Allocation and reuse

How the server lays out cache memory matters as much as its size: paged allocation stores the cache in fixed-size blocks so sequences of unknown length do not reserve or fragment memory ([PagedAttention](lesson:ieng:08.03)). Requests that share a prefix (system prompt, tool definitions, a long document) can reuse its cached K and V instead of prefilling it again; hosted APIs expose this as prompt caching with cheaper cached input tokens, and the mechanics and pricing are in [Cost budgeting](lesson:aieng:12.04).

## Where this breaks

- **Sizing by weights alone.** A deployment that fits the weights can still run out of memory at long context and high batch; size the cache for your real length distribution.
- **Treating a window as memory.** A sliding-window or sink model streams forever, but facts outside the window are gone unless a global layer or retrieval brings them back.
- **Expecting sparse attention to save memory.** NSA and DSA save compute and bandwidth at long context; the stored cache still grows with every token.
- **Quantizing the cache blindly.** fp8 caches are usually safe; 4-bit caches can hurt long-context recall. Check on your own long inputs.

<div class="take">
<div class="lab">Key takeaway</div>

The KV cache trades memory for compute by storing past keys and values, and at long context and high batch it dominates GPU memory. Architectures attack it in three ways: smaller entries (GQA, MLA, fp8), fewer kept entries (sliding windows with sinks), and fewer read entries (sparse attention).

</div>
