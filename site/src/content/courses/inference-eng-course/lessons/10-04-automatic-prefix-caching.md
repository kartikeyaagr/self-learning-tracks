---
id: "10.04"
course: "inference-eng-course"
module: "10"
slug: "automatic-prefix-caching"
title: "Automatic Prefix Caching"
keyIdeas: ["Keep KV blocks of finished requests and index them by a hash of content plus prefix.", "New requests reuse the longest cached prefix and prefill only the rest.", "Hash = f(parent hash, block tokens, extras like LoRA id or image hash).", "Eviction: LRU among blocks with refcount zero.", "Benefit: lower TTFT and more throughput, proportional to hit rate."]
question: "How does block-hash-based prefix caching decide that a new request can reuse cached KV, and why must the hash include the parent block?"
answer: "The prompt is split into full blocks; each block's hash is computed from its tokens plus the hash of the previous block. The engine looks up each hash in a table of cached blocks and reuses blocks until the first miss. Including the parent hash is required because a block's K and V depend on all earlier tokens, so identical tokens after different prefixes produce different KV."
sources: [{"title": "vLLM: Automatic prefix caching design", "url": "https://docs.vllm.ai/en/latest/design/prefix_caching.html", "note": "Hash chain and eviction."}, {"title": "SGLang paper (RadixAttention)", "url": "https://arxiv.org/abs/2312.07104", "note": "Tree-based alternative."}]
tags: ["prefix", "caching"]
---

## The mechanism

Automatic prefix caching, introduced in lesson 08.04, keeps the KV blocks of finished or running requests indexed by their content. When a new request arrives, the engine finds the longest cached prefix of its prompt and skips prefill for that part.

~~~dg-flow A prefix cache lookup, block by block.
Prompt -> Split into full blocks -> Hash chain h1, h2, h3 -> *Look up each hash -> Reuse hits, prefill the rest
~~~

The details:

1. Split the prompt's token ids into full blocks of 16 tokens.
2. Compute each block's hash from its own tokens and the hash of the block before it. Because of the chain, a block's hash identifies the entire prefix up to that block.
3. Look the hashes up in order. Stop at the first miss.
4. Every hit is a block whose K and V are already computed. Add it to the request's block table and increase its reference count.
5. Prefill only the tokens after the last hit.

Two prompts that agree on their first 1,000 tokens share the first 62 block hashes (992 tokens). The remaining 8 tokens of agreement fall in a partly matched block and are recomputed.

## What goes into the hash

The KV cache depends on more than the token ids, so the hash must include anything else that changes K and V:

- **LoRA adapter id.** Different adapters produce different keys and values for the same tokens.
- **Multimodal inputs.** For image or audio tokens, a hash of the actual image or audio data.
- **A cache salt.** Some engines let you add a per-tenant value to the hash, so tenants never share cache entries. This prevents one tenant from detecting another's prompts by timing.

## Eviction

Cached blocks that no running request uses sit in an LRU list. They cost nothing until memory is needed. When the allocator needs a block and the free list is empty, it evicts the least recently used cached block. So prefix caching never reduces the memory available to running requests. It only uses memory that would otherwise be idle.

## The effect

| Prefix hit rate (share of prompt tokens) | TTFT effect | Throughput effect |
| --- | --- | --- |
| 0% | none, tiny overhead | none |
| 50% | about half the prefill work | noticeable on prefill-heavy loads |
| 90% | prefill mostly skipped | large for RAG, agents and multi-turn chat |

<div class="ex">
<div class="lab">Worked example</div>

**the gain on a shared-prompt workload**

A workload of 2,000-token prompts where the first 1,800 tokens are a shared system prompt, with 200-token outputs, on an 8B model.

Without caching: each request prefills 2,000 tokens, about 55 ms on an H100. Prefill uses about 35% of GPU time at the target load.

With caching: each request prefills about 210 tokens (the system prompt reuses 112 full blocks, 1,792 tokens). Prefill falls to about 8 ms and about 5% of GPU time.

p95 TTFT falls from 180 ms to 45 ms, and the freed compute lets the same GPU serve about 40% more requests per second at the same SLO.

</div>

## Caveats

- **Only full blocks match.** The last partial block of a shared prefix is always recomputed.
- **The final token is always computed.** The engine needs logits for the last prompt position, so even a fully cached prompt runs at least one token of prefill.
- **Cached blocks compete with running requests.** Under heavy load, the pool is full of active sequences, cached blocks are evicted quickly, and the hit rate falls exactly when you need it most.
- **Sampling-related outputs.** Prompt logprobs require the full prefill, so engines may bypass the cache for requests that ask for them.

## Metrics to watch

Engines report prefix cache queries and hits, counted in tokens or blocks. The hit rate is hits divided by queries. Watch it next to load: a hit rate that drops as load rises means the cache is too small for the working set, which lesson 10.06 covers.

## Deep dive: what a hit rate buys in capacity

If prefill takes a share ⟦m612⟧ of GPU time and a fraction ⟦m613⟧ of prefill tokens hit the cache, GPU time per request falls by ⟦m614⟧, so capacity rises by

⟦M611⟧

~~~dg-raw
%%chart Capacity gain from prefix caching, by hit rate and prefill share of GPU time.
x = 0, 0.2, 0.4, 0.6, 0.8, 1
xlab = Prefix-cache hit rate h
ylab = Capacity multiple
Chat (s = 0.25): 1, 1.05, 1.11, 1.18, 1.25, 1.33
Mixed (s = 0.5): 1, 1.11, 1.25, 1.43, 1.67, 2
RAG and agents (s = 0.75): 1, 1.18, 1.43, 1.82, 2.5, 4
%%
~~~

The same 60% hit rate is worth 18% more capacity on a chat workload and 82% on an agent workload. Measure ⟦m615⟧ from the phase split in 04.03 before estimating the benefit.

<div class="take">
<div class="lab">Key takeaway</div>

hash-chained blocks let a new request reuse the KV of any earlier matching prefix and skip that part of prefill. Eviction uses only idle memory, so caching is close to free, and the hit rate predicts the gain.

</div>
