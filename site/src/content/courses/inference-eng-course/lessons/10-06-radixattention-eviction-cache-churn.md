---
id: "10.06"
course: "inference-eng-course"
module: "10"
slug: "radixattention-eviction-cache-churn"
title: "RadixAttention, Eviction & Cache Churn"
keyIdeas: ["RadixAttention keeps cached KV in a radix tree keyed by token sequences.", "Longest-prefix match is a tree walk; shared prefixes are shared nodes.", "Eviction removes least recently used leaves first, keeping hot shared roots.", "Cache-aware scheduling: run requests with shared prefixes close together.", "Churn: working set larger than cache; hit rate drops and eviction overhead rises."]
question: "What does a radix tree give SGLang's prefix cache that a flat hash table of blocks does not, and what is cache churn?"
answer: "A radix tree stores cached sequences as shared paths, so finding the longest matching prefix is a single walk, sharing is explicit, and eviction can remove leaves (least recently used) without breaking parents still in use. It also helps schedule requests that share a prefix together. Cache churn is when entries are evicted before they are reused, because the working set is larger than the cache, so you pay to store KV that never hits."
sources: [{"title": "SGLang: Efficient Execution of Structured Language Model Programs", "url": "https://arxiv.org/abs/2312.07104", "note": "RadixAttention."}, {"title": "SGLang GitHub repository", "url": "https://github.com/sgl-project/sglang", "note": "Implementation."}]
tags: ["prefix", "sglang"]
---

## The radix tree

SGLang organizes its prefix cache as a **radix tree**, an approach it calls RadixAttention. Each edge holds a run of token ids and points to the KV blocks for those tokens. A request's prompt is matched by walking down from the root as far as its tokens agree with an edge. Where two prompts diverge, the tree branches.

~~~dg-flow A radix tree of cached prompts. Shared parts sit near the root.
Root -> *System prompt -> Document A -> Question 1
Root -> *System prompt -> Document B -> Question 2
~~~

Compared with the hash-chain approach in vLLM, the radix tree:

- matches at token granularity within an edge, though storage is still in blocks or pages,
- makes the shared structure explicit, which helps scheduling and eviction decisions,
- lets the engine see how many requests depend on each node.

Both approaches give similar hit rates for typical workloads.

## Eviction in a tree

Nodes in use by running requests are locked. Unlocked leaves are evicted first, least recently used. A node becomes a leaf, and so evictable, only when all its children are gone. As a result, shared nodes near the root, such as system prompts, stay in the cache because many requests keep them warm, while unique tails of finished requests are evicted first. This is the right priority: unique tails are the least likely to be reused.

## Cache-aware scheduling

SGLang can reorder the waiting queue to favour requests with the longest cached prefix. This raises the hit rate: requests that share a prefix run close together in time, before the prefix is evicted.

The risk is fairness. A request with no cached prefix can keep losing its place to cache-friendly ones. Engines limit the reordering, for example by falling back to arrival order for requests that have waited longer than some threshold.

## Churn

The cache helps only if the set of prefixes in active use, the **working set**, fits in the memory available for caching. When it does not, entries are evicted before they are reused. This is called **churn**.

| Working set compared with the cache | Behaviour |
| --- | --- |
| much smaller | high, stable hit rate |
| similar size | hit rate sensitive to the order of traffic |
| much larger | low hit rate, high eviction counts, little benefit |

<div class="ex">
<div class="lab">Worked example</div>

**churn under load**

An agent platform's prefixes (system prompt, tools and per-task scratchpads) total about 500,000 tokens of active working set per replica.

At low load, about 300,000 tokens of pool are free for caching. Hit rate: 70%.

At peak load, running sequences fill all but 50,000 tokens. Hit rate drops to 25%, so prefill work nearly triples exactly when the GPU is busiest, and TTFT climbs sharply.

Fixes that worked: FP8 KV (doubling the pool), prefix-aware routing (each replica sees a smaller working set), and CPU offload for cached blocks.

</div>

## Signs and fixes

Signs of churn:

- The hit rate falls as load rises.
- Eviction counts are high, and the same prefixes are computed repeatedly.
- TTFT rises faster with load than queueing alone explains.

Fixes:

- **More KV memory**: FP8 KV cache, quantized weights, GPUs with more memory, more tensor parallelism.
- **Prefix-aware routing**: each replica serves a subset of prefixes, so each working set is smaller (Module 22).
- **Tiered caching**: move evicted blocks to CPU memory or SSD instead of dropping them, and copy them back on a hit (Module 20). Copying 128 KB per token over PCIe is much cheaper than recomputing for long prefixes.
- **Prompt hygiene**: remove accidental variation, which inflates the working set with near-duplicates.

## Deep dive: what to evict from a radix tree

A node's value is the prefill it saves when hit, divided by the memory it holds. For a node of ⟦m623⟧ tokens at depth (offset) ⟦m624⟧, hit at rate ⟦m625⟧:

⟦M622⟧

Plain LRU uses only recency, a proxy for ⟦m626⟧. Deeper nodes save more attention per token, so at very long contexts a cost-aware policy keeps deep, frequently used branches that LRU would drop. Engines mostly still use LRU on leaves, because it is cheap and a parent can only be evicted after its children.

~~~dg-flow Eviction order in a radix tree under LRU. Only leaves are candidates.
Root -> System prompt, hot -> Tools A, hot -> *Task 17, cold leaf: evicted first
Root -> System prompt, hot -> Tools A, hot -> Task 18, active
Root -> System prompt, hot -> Tools B, warm
~~~

<div class="take">
<div class="lab">Key takeaway</div>

a radix tree makes prefix matching and LRU eviction natural, and keeps shared roots warm. The hit rate depends on the working set fitting the cache, so watch it against load, and route, size and tier the cache to avoid churn.

</div>
