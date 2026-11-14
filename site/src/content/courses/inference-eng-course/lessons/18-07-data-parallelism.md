---
id: "18.07"
course: "inference-eng-course"
module: "18"
slug: "data-parallelism"
title: "Data Parallelism"
keyIdeas: ["Data parallelism in serving: independent replicas, each with full weights.", "No per-step communication; scales throughput linearly if routing is balanced.", "Needs a router to spread requests (and ideally preserve cache locality).", "For MoE: data-parallel attention plus expert-parallel MLPs is a common design.", "Simplest and most efficient option when the model fits per replica."]
question: "For serving, what is data parallelism and when is it combined with expert parallelism?"
answer: "In serving, data parallelism means running several full replicas of the model, each handling different requests, with no communication between them during a step. For large MoE models, attention can be data parallel (each replica holds its own KV and requests) while the expert layers are spread across all GPUs with expert parallelism, so a DP group shares the experts through all-to-all communication."
sources: [{"title": "vLLM: Data parallel deployment", "url": "https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html", "note": "DP in vLLM including with EP."}, {"title": "DeepSeek-V3 Technical Report", "url": "https://arxiv.org/abs/2412.19437", "note": "DP attention with EP experts at scale."}]
tags: ["data-parallel"]
---

## Plain replicas

Data parallelism (DP), in its simplest form, means running several independent copies of the model. Each replica, whether one GPU or a TP group, holds the full model and serves its own requests. There is no communication between replicas during a step. Throughput grows almost linearly with the number of replicas, as long as traffic is spread evenly.

This is the default way to scale throughput. It is also the most robust: a failed replica affects only its own requests, and replicas can be added or removed independently.

## Routing decides how well it works

A load balancer picks a replica for each request, and the choice matters:

- **Round robin** spreads requests evenly by count, but ignores that requests differ in size by a thousand times.
- **Least loaded** sends each request to the replica with the fewest running or queued requests, or the most free KV cache. This keeps queues even.
- **Prefix-aware** sends requests to the replica most likely to have their prefix cached, balanced against load. This keeps cache hit rates high (Module 22).

<div class="ex">
<div class="lab">Worked example</div>

**routing with replicas**

Four replicas serve a chat product with a shared system prompt and multi-turn conversations.

Round robin: each turn of a conversation lands on a random replica, so the conversation's history is cached on the right replica only a quarter of the time. Prefix hit rate: 35%.

Session-sticky routing: hit rate 78%, and p95 TTFT falls from 320 ms to 140 ms. But during a burst, one replica gets several long conversations and its queue grows.

Prefix-aware routing with a load cap gets most of the cache benefit while keeping queues even.

</div>

## DP inside one engine

Some engines run several DP ranks as a single service rather than as separate servers. Each rank has its own scheduler, KV cache and batch, but the engine coordinates them. This matters most when DP is combined with expert parallelism, where all ranks must step together because they exchange tokens in every MoE layer.

## DP attention with EP experts for MoE

Large MoE models have a particular structure: attention weights are small, while expert weights are huge. That suggests a hybrid layout.

~~~dg-flow MoE serving with data-parallel attention and expert-parallel experts.
Requests split across DP ranks -> *Attention local per rank, own KV -> All-to-all dispatch -> Experts spread over all GPUs -> All-to-all combine
~~~

- **Attention is data parallel.** Each GPU holds a full copy of the attention weights, which is cheap, and runs attention for its own requests using its own local KV cache. There is no all-reduce for attention.
- **Experts are expert parallel.** Each GPU holds a share of the experts. Tokens are sent to the GPUs that hold their experts and the results are sent back.

Compared with tensor parallelism for attention, this avoids replicating the KV cache across GPUs (which TP must do when there are fewer KV heads than GPUs, and which MLA models always face), and it keeps the attention computation local.

## Limits of plain DP

- Each replica needs the full model, so DP alone cannot serve models larger than one replica's memory.
- Each replica has its own prefix cache, so the effective cache is fragmented unless routing is prefix-aware.
- Small replicas have less KV capacity each, so very long contexts may need TP or CP.

## Deep dive: imbalance from random routing

Sending ⟦m923⟧ requests at random to ⟦m924⟧ replicas gives each one ⟦m925⟧ on average, but the busiest replica gets about

⟦M922⟧

With 800 in-flight requests on 8 replicas: average 100, busiest about 120. That replica sets the tail latency. Sending each request to the less loaded of two random replicas ("power of two choices") cuts the excess from a square-root term to about ⟦m926⟧ requests, close to perfect balance at almost no cost.

<div class="take">
<div class="lab">Key takeaway</div>

data parallelism means replicas, the simplest way to scale throughput nearly linearly, and routing decides how well it works. For large MoE models, pair data-parallel attention with expert-parallel experts.

</div>
