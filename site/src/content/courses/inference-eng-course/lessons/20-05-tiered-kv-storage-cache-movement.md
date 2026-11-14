---
id: "20.05"
course: "inference-eng-course"
module: "20"
slug: "tiered-kv-storage-cache-movement"
title: "Tiered KV Storage & Cache Movement"
keyIdeas: ["A tiered KV system moves blocks between HBM, CPU RAM, SSD and remote storage.", "Hide transfer time: async copies, pinned memory, separate streams, layer-wise pipelining.", "Prefetch based on signals: a conversation is active, a document is popular.", "Write-back on eviction should not block the critical path.", "Track hit rates per tier and transfer bandwidth."]
question: "How can an engine hide the latency of loading KV from CPU memory for a request with a cached prefix?"
answer: "Start the transfer as soon as the request is scheduled (or even when it arrives), and overlap it with other work: copy layer by layer so attention for layer 1 can start while later layers load, or schedule the request for a later step and run other requests meanwhile. Use asynchronous copies on separate CUDA streams and pinned host memory."
sources: [{"title": "LMCache", "url": "https://github.com/LMCache/LMCache", "note": "Layer-wise KV loading and storage backends."}, {"title": "CachedAttention / AttentionStore", "url": "https://arxiv.org/abs/2403.19708", "note": "Hierarchical KV caching for multi-turn conversations."}]
tags: ["kv-offload"]
---

## Movement paths

~~~dg-flow KV block movement between tiers.
GPU HBM -(evict, async)-> CPU DRAM -(evict)-> SSD or remote
SSD or remote -(prefetch)-> CPU DRAM -(load, layer by layer)-> *GPU HBM
~~~

Blocks move down when they are evicted and up when they are needed. Neither move should block the GPU.

## Hiding latency

- **Pinned host memory:** page-locked memory lets the GPU's copy engines use fast asynchronous DMA. Pageable memory forces a slower staging copy.
- **Separate CUDA streams:** copies run on their own stream while compute continues on the main one.
- **Layer-wise loading:** load the KV for layer l + 1 while computing layer l. If one layer's compute takes longer than one layer's copy, the load is almost fully hidden.
- **Deferred scheduling:** mark the request as loading and run other requests until its KV is ready.

<div class="ex">
<div class="lab">Worked example</div>

**does layer-wise loading hide a reload?**

A 32k-token prefix on an 8B model is 4 GB across 32 layers, so 128 MB per layer.

At 50 GB/s, each layer's copy takes about 2.6 ms.

If the request then prefills a 2k-token new turn, each layer's compute takes about 1.5 ms at 50% of H100 peak. The copy is longer than the compute, so about 1.1 ms per layer, 35 ms in total, is still exposed. That is still far better than 1.6 s of recompute, but not free.

</div>

## Prefetching

Predict what will be needed and start moving it early:

- When a user starts typing in a conversation, load that conversation's history.
- Keep popular documents and system prompts in CPU memory at all times.
- When an agent calls a tool, keep its context warm until the tool returns.

## Metrics

| Metric | Meaning |
| --- | --- |
| Hit rate per tier | where reuse comes from |
| Achieved load bandwidth | how efficiently links are used |
| Requests waiting on loads | latency the user sees from movement |
| Eviction write-back rate | pressure on lower tiers |

## Deep dive: when layer-by-layer loading hides the reload

If KV for layer ⟦m978⟧ loads while layer ⟦m979⟧ computes the new tokens, the load is hidden when, per layer,

⟦M977⟧

with ⟦m980⟧ new tokens and ⟦m981⟧ KV bytes per token per layer. For Llama 3 8B over PCIe at 50 GB/s, the factor is about 0.1: the new tokens must be at least a tenth of the reloaded prefix.

<div class="ex">
<div class="lab">Worked example</div>

**a follow-up question on a 32k document**

Prefix 32,768 tokens, new question 500 tokens. The condition needs about 3,400 new tokens.

Per layer: load ⟦m982⟧ ms; compute for 500 tokens about 0.4 ms. Most of the load is exposed: about 73 ms of the 86 ms total.

Still far better than recomputing 32k tokens (about 1.3 s), but the overlap helps little for short follow-ups.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

tiered KV works when movement is asynchronous and overlapped with compute. Prefetch where you can, and measure the load time that is still exposed.

</div>
