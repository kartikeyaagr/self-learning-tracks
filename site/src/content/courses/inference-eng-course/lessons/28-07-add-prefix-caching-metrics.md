---
id: "28.07"
course: "inference-eng-course"
module: "28"
slug: "add-prefix-caching-metrics"
title: "Add Prefix Caching & Metrics"
keyIdeas: ["Add block hashing and a cached-block map to your allocator.", "On admission, reuse the longest cached prefix and skip its prefill.", "Keep freed blocks in an LRU for reuse; evict when needed.", "Export Prometheus metrics matching common engine metrics.", "Test hit rate on a shared-prefix workload."]
question: "Which metrics should your mini engine export so you can compare it with vLLM?"
answer: "Running and waiting requests, KV blocks used and free, prefix cache hit tokens and queries, preemptions, prompt and generation token counters, and histograms for TTFT, TPOT and E2E, all in Prometheus format at /metrics."
sources: [{"title": "vLLM: Automatic prefix caching design", "url": "https://docs.vllm.ai/en/latest/design/prefix_caching.html", "note": "Reference design."}, {"title": "Prometheus Python client", "url": "https://github.com/prometheus/client_python", "note": "Exporting metrics."}]
tags: ["mini-engine", "prefix"]
---

## Prefix caching

- **Hash** each full block together with its parent's hash, so a hash identifies the whole prefix up to that block.
- **On admission:** walk the request's block hashes from the start, take every hit, increment those blocks' reference counts, and set num_computed_tokens to the hit length.
- **On finish:** decrement reference counts. Keep blocks in the cache with a timestamp for LRU eviction.

```
h = None
for i, block_tokens in enumerate(full_blocks(prompt)):
    h = hash((h, tuple(block_tokens)))
    blk = cache.get(h)
    if blk is None:
        break
    blk.ref += 1; req.block_table.append(blk); req.num_computed_tokens += BLOCK
```

## Metrics

| Metric | Type |
| --- | --- |
| requests_running, requests_waiting | gauge |
| kv_blocks_used, kv_blocks_free | gauge |
| prefix_cache_hit_tokens, prefix_cache_query_tokens | counter |
| preemptions | counter |
| prompt_tokens, generation_tokens | counter |
| ttft_seconds, tpot_seconds, e2e_seconds | histogram |

Use the same names and types as a production engine where you can. Your dashboards and benchmark scripts will then work on both.

## Test

Send a workload where every request shares a system prompt. The hit rate should approach the shared fraction of prompt tokens, and TTFT should drop to match.

<div class="ex">
<div class="lab">Worked example</div>

**expected hit rate**

Every request has a 1,200-token shared system prompt and a 300-token unique question, 1,500 tokens in total.

With 16-token blocks, 1,200 tokens is exactly 75 blocks, so after the first request each one hits 1,200 tokens: an 80% token hit rate.

Prefill per request falls from 1,500 tokens to 300. TTFT should drop to roughly a fifth, plus fixed overhead.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

prefix caching is a hash map over your block pool, and metrics make the mini engine comparable with production engines.

</div>
