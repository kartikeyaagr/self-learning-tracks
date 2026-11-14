---
id: "22.07"
course: "inference-eng-course"
module: "22"
slug: "multi-model-multi-replica-serving"
title: "Multi-Model & Multi-Replica Serving"
keyIdeas: ["Many low-traffic models waste GPUs if each gets dedicated replicas.", "Options: multi-LoRA on a shared base, co-locate several small models per GPU, or swap models on demand.", "Popular models get dedicated replicas; long-tail models share.", "A routing layer maps model names to replica pools.", "Track per-model traffic to decide placement."]
question: "You must serve 12 fine-tuned variants of the same 8B base, each with low traffic. What are your options?"
answer: "Serve the base once with LoRA adapters and multi-LoRA batching, so all variants share weights and run in the same batches (best if fine-tunes are LoRA). If they are full fine-tunes, pack several models per GPU when memory allows, or use model swapping with fast loading for rarely used ones. Giving each variant its own GPU wastes capacity at low traffic."
sources: [{"title": "S-LoRA: Serving Thousands of Concurrent LoRA Adapters", "url": "https://arxiv.org/abs/2311.03285", "note": "Multi-adapter serving."}, {"title": "vLLM: LoRA adapters", "url": "https://docs.vllm.ai/en/latest/features/lora.html", "note": "Serving multiple LoRA adapters."}]
tags: ["multi-model"]
---

## Placement strategies

| Strategy | Good for | Trade-off |
| --- | --- | --- |
| Dedicated replicas | high-traffic models | idle cost when traffic is low |
| Multi-LoRA on one base | many fine-tunes of one base | adapter overhead, rank limits |
| Several models per GPU | small models | memory split, interference |
| On-demand swapping | rarely used models | cold start latency on first use |

## Routing

A gateway maps the model field of each request to a pool. Pools can be:

- One model with many replicas.
- One base model with many LoRA adapters loaded on demand.
- A shared pool that loads models when they are requested and evicts idle ones.

~~~dg-flow Gateway routing by model.
Request (model = x) -> *Gateway lookup -> Dedicated pool, LoRA pool or shared pool -> Replica
~~~

## Deciding by traffic

Plot requests per model, sorted. The curve usually has a long tail.

<div class="ex">
<div class="lab">Worked example</div>

**40 models on 24 GPUs**

The top 3 models take 85% of traffic. They get dedicated pools sized by capacity planning: 8, 6 and 4 GPUs.

30 customer fine-tunes of the same 8B base take 12% of traffic. They share 4 GPUs as LoRA adapters on that base.

7 rarely used models take 3%. They share 2 GPUs with on-demand loading. The first request after idle waits about 30 seconds, which is acceptable for those users.

Giving every model its own GPU would need 40 GPUs, most of them idle.

</div>

Revisit placement as traffic shifts. A tail model that becomes popular should move to its own pool.

## Deep dive: keep a model resident or load it on demand

Keeping a model loaded costs its GPU share all the time. Loading on demand costs ⟦m1032⟧ of delay for each request that arrives while it is unloaded. With LRU eviction, a model with request rate ⟦m1033⟧ and typical eviction time ⟦m1034⟧ (how long an idle model survives) is cold for a share of its requests of about

⟦M1031⟧

A model used every few seconds is never cold. A model used once an hour, with ⟦m1035⟧ minutes, is cold for 85% of its requests, and each of those requests waits the full load time. Either keep such models on a small dedicated GPU, accept the delay, or quantize them so more fit at once and ⟦m1036⟧ grows.

<div class="take">
<div class="lab">Key takeaway</div>

match the serving strategy to traffic per model. Share GPUs across the long tail with LoRA or co-location, and give dedicated replicas only to models that need them.

</div>
