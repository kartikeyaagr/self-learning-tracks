---
id: "27.04"
course: "inference-eng-course"
module: "27"
slug: "follow-kv-allocation-block-management"
title: "Follow KV Allocation & Block Management"
keyIdeas: ["Find the block pool, block hashing and the free-block queue.", "Trace allocate, append, free and prefix lookup for one request.", "Check eviction order and how cached blocks are reused.", "Compare with your own allocator from Module 08.", "Write a test that shows a prefix hit and an eviction."]
question: "Where in the code would you confirm how prefix cache hits are found and how evicted blocks are chosen?"
answer: "In the KV cache manager and block pool: the function that computes block hashes for a request's tokens, the lookup that walks those hashes against the cached-block map to find the longest hit, and the free-block queue that orders cached blocks with zero references for eviction (least recently used first). Tests for the block pool show these behaviors directly."
sources: [{"title": "vLLM: Automatic prefix caching design", "url": "https://docs.vllm.ai/en/latest/design/prefix_caching.html", "note": "Design doc to compare against the code."}, {"title": "vLLM GitHub repository", "url": "https://github.com/vllm-project/vllm", "note": "KV cache manager source."}]
tags: ["kv-cache", "internals"]
---

## Things to find

| Concept | What to look for |
| --- | --- |
| Block | block id, reference count, hash |
| Block pool | free queue, map from hash to cached block |
| Hashing | parent hash plus token ids plus extra keys (adapter id, image hashes) |
| Allocation | getting new blocks for scheduled tokens |
| Prefix lookup | the longest chain of cached hashes |
| Eviction | pop the least recently used free block and remove it from the cache map |
| Free | decrement references and push to the free queue |

## How the pieces fit

A block that a running request uses has a reference count above zero and cannot be evicted. When the request finishes, its blocks' counts drop. Blocks at zero go to the free queue, but stay in the hash map. If a new request matches their hash before they are reused, they are taken back from the free queue: a prefix cache hit at no cost.

~~~dg-raw
%%widget paged Allocate and free requests and watch blocks move between the pool and each request's block table.
%%
~~~

## Exercise

1. Send request A with a 1,000-token prompt.
2. Send request B with the same first 800 tokens.
3. Log the number of cached tokens found for B. With 16-token blocks, expect 800, since 800 is a multiple of 16. With a 700-token shared prefix, expect 688, rounded down to a block boundary.
4. Fill memory with other requests, and confirm A's blocks are evicted in least-recently-used order.

<div class="take">
<div class="lab">Key takeaway</div>

the KV manager is a block pool, a hash map and a free queue. Matching the code to your own allocator from earlier modules makes it quick to understand.

</div>
