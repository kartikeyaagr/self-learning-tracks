---
id: "03.06"
course: "inference-eng-course"
module: "03"
slug: "calculate-kv-memory-context-concurrency-limits"
title: "Calculate KV Memory, Context & Concurrency Limits"
keyIdeas: ["KV bytes per token = 2 x layers x kv_heads x head_dim x bytes per element.", "GQA cuts this by the ratio of query heads to KV heads.", "Concurrency at full context = KV pool / (bytes per token x context length).", "Real sequences are shorter than max; paged allocation lets you plan on average length, not max.", "KV quantization (FP8) halves bytes per token."]
question: "Compute KV cache bytes per token for a model with 32 layers, 8 KV heads, head dim 128, in BF16. How many 8k-token conversations fit in 40 GB?"
answer: "Per token: 2 (K and V) x 32 layers x 8 heads x 128 dim x 2 bytes = 131,072 bytes, about 128 KB. An 8k-token sequence needs about 1 GB. So 40 GB holds about 40 such sequences at full length."
sources: [{"title": "kipply: Transformer Inference Arithmetic", "url": "https://kipp.ly/transformer-inference-arithmetic/", "note": "KV cache formula with examples."}, {"title": "GQA paper", "url": "https://arxiv.org/abs/2305.13245", "note": "Why fewer KV heads shrink the cache."}]
tags: ["memory", "kv-cache"]
---

## The formula

The KV cache stores one key vector and one value vector per token, per layer, per KV head:

KV bytes per token = 2 × layers × KV heads × head size × bytes per element

The 2 counts K and V. Memorize this formula. It answers more capacity questions than any other.

| Model | Layers | KV heads | Head size | BF16 bytes per token |
| --- | --- | --- | --- | --- |
| Llama 3 8B | 32 | 8 | 128 | 128 KB |
| Mistral 7B | 32 | 8 | 128 | 128 KB |
| Llama 2 7B (MHA) | 32 | 32 | 128 | 512 KB |
| Qwen2.5 32B | 64 | 8 | 128 | 256 KB |
| Llama 3 70B | 80 | 8 | 128 | 320 KB |

The effect of GQA is plain in the table. Llama 2 7B and Llama 3 8B are about the same size, but the older model needs four times as much cache per token.

## From bytes to users

Once you know the pool size and the bytes per token:

tokens that fit = KV pool bytes ÷ KV bytes per token
concurrent sequences ≈ tokens that fit ÷ average tokens per sequence

The average tokens per sequence is prompt plus output so far. For capacity planning, use the expected maximum: the prompt plus the full output length, because a request's cache keeps growing until it finishes.

~~~dg-raw
%%widget kv Move context and batch until the bar turns red. That is the concurrency limit for this model and GPU.
model = Llama 3 70B
gpu = H100 SXM (80 GB)
t = 12
b = 16
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**an 8B model on one H100**

KV pool: about 52 GB (from lesson 03.05). Tokens that fit: 52 GB ÷ 128 KB ≈ 406,000.

At 2,000 tokens per conversation: about 200 concurrent requests.

At 8,000 tokens per conversation: about 50.

At 32,000 tokens per conversation: about 12.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**Llama 3 70B on four H100s with tensor parallelism**

Weights: 140 GB split four ways, 35 GB per GPU.

Per GPU: 72 GB target − 35 GB weights − about 4 GB for activations, graphs and context = 33 GB of pool. Across four GPUs: 132 GB.

Tokens that fit: 132 GB ÷ 320 KB ≈ 412,000. At 4,000 tokens per request, about 100 concurrent requests.

With tensor parallelism each GPU holds the cache for its share of the KV heads, so the 320 KB per token is split across the four GPUs as 80 KB each.

</div>

## Context length is a per-request limit too

The pool is a hard ceiling for any one request as well. One 128k-token request on Llama 3 8B needs 128,000 × 128 KB = 16 GB of cache. If the pool is 10 GB, that request cannot run at all, even on an idle server. Engines check this at startup: if `max_model_len` tokens of cache do not fit in the pool, they refuse to start or lower the limit.

## Concurrency is not throughput

Fitting 200 sequences in memory does not mean running 200 is a good idea. Each decode step must read every active sequence's cache. At 200 sequences of 2,000 tokens, that is 51 GB of KV reads per step, three times the weight read. The step takes about 22 ms instead of 6 ms, and each user's stream slows to about 45 tokens per second. Memory sets the maximum concurrency; your latency target often sets a lower one.

## Ways to fit more

- **FP8 KV cache.** One byte per element instead of two halves the cache. Quality loss is usually small with per-tensor or per-head scales, but test it on your tasks.
- **Smaller weights.** Moving weights from BF16 to FP8 frees 8 GB on an 8B model, which is 65,000 more tokens of cache.
- **Prefix sharing.** Requests with the same system prompt can share its cache blocks. A 2,000-token system prompt shared by 100 requests saves 25 GB.
- **Offloading.** Move the cache of idle or paused conversations to CPU memory or SSD and bring it back when needed (Module 20).
- **More GPUs.** Tensor parallelism splits both the weights and the cache, so the pool per GPU grows.

## Deep dive: concurrency from first principles

From the pool size ⟦m331⟧, KV bytes per token ⟦m332⟧, and the average number of tokens each active sequence holds ⟦m333⟧:

⟦M329⟧

The subtle part is ⟦m334⟧. A sequence's KV grows as it decodes, from ⟦m335⟧ at the start to ⟦m336⟧ at the end. Averaged over its lifetime it holds about ⟦m337⟧ tokens. Paged engines allocate as tokens arrive, so

⟦M330⟧

<div class="ex">
<div class="lab">Worked example</div>

**the same pool, two allocation policies**

52 GB pool, Llama 3 8B (128 KB per token), prompts of 1,500 tokens, outputs of 500, maximum context 8,192.

Reserving the maximum: ⟦m338⟧ sequences.

Paged, average lifetime footprint ⟦m339⟧ tokens: ⟦m340⟧ sequences on average. Peaks are lower because sequences that happen to be long at the same moment need more.

</div>

~~~dg-raw
%%chart Sequences that fit in a 52 GB pool as average context grows (Llama 3 8B).
x = 512, 1024, 2048, 4096, 8192, 16384, 32768
logx = 1
logy = 1
xlab = Average tokens held per sequence
ylab = Concurrent sequences
BF16 KV: 793, 396, 198, 99, 49, 24, 12
FP8 KV: 1586, 793, 396, 198, 99, 49, 24
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

KV bytes per token is 2 × layers × KV heads × head size × bytes. Multiply by tokens in flight and compare with the pool. Memory sets the hard limit on concurrency, but latency targets often set a lower one.

</div>
