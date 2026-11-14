---
id: "17.02"
course: "inference-eng-course"
module: "17"
slug: "vllm-vs-sglang-architecture"
title: "vLLM vs SGLang Architecture"
keyIdeas: ["Both: continuous batching, paged KV, prefix caching, chunked prefill, OpenAI API.", "vLLM: PagedAttention origin, block hash prefix caching, broad model and hardware support.", "SGLang: RadixAttention tree cache, cache-aware scheduling, structured program frontend.", "Both have moved toward CPU-overhead reduction (overlapped scheduling, separate processes).", "Performance differences depend on workload and version; benchmark both."]
question: "Name two architectural ideas that distinguish SGLang from vLLM historically, and one area where both have converged."
answer: "SGLang introduced RadixAttention, a radix-tree prefix cache with cache-aware scheduling, and a frontend language for structured LLM programs; it also emphasized a lightweight, overlap-friendly scheduler. vLLM introduced PagedAttention and a block-based KV manager with hash-based prefix caching. Both now support continuous batching, chunked prefill, prefix caching, speculative decoding, structured output and multi-GPU parallelism."
sources: [{"title": "SGLang paper", "url": "https://arxiv.org/abs/2312.07104", "note": "RadixAttention and frontend language."}, {"title": "vLLM: Architecture overview", "url": "https://docs.vllm.ai/en/latest/design/arch_overview.html", "note": "vLLM's components."}, {"title": "SGLang documentation", "url": "https://docs.sglang.ai/", "note": "Engine features."}]
tags: ["vllm", "sglang"]
---

## A shared foundation

vLLM and SGLang are the two most widely used open-source serving engines. They share most of their core ideas, and each has adopted good ideas from the other over time.

| Feature | vLLM | SGLang |
| --- | --- | --- |
| Continuous batching | yes | yes |
| Paged KV cache | fixed-size blocks with block tables | token-level or small pages, managed with a radix tree |
| Prefix caching | hash-chained blocks | radix tree (RadixAttention) |
| Chunked prefill | yes | yes |
| Speculative decoding | yes, several methods | yes, several methods |
| Structured output | yes | yes |
| Tensor, pipeline, expert and data parallelism | yes | yes |
| Prefill-decode disaggregation | yes | yes |
| OpenAI-compatible server | yes | yes |

## vLLM's distinguishing ideas

- **PagedAttention**, which introduced block-based KV memory to LLM serving.
- **Broad coverage**: a very large set of model architectures and hardware backends, including NVIDIA, AMD, TPUs and others through plugins.
- **A redesigned engine core** that isolates the scheduler loop in its own process, with the API server, tokenization and detokenization elsewhere, to keep the GPU fed.
- **torch.compile integration** with piecewise CUDA graphs as a standard part of the model runner.

## SGLang's distinguishing ideas

- **RadixAttention**: the prefix cache as a radix tree, with LRU eviction of leaves, which keeps shared prefixes warm naturally.
- **Cache-aware scheduling**: longest-prefix-first ordering of the waiting queue, to raise hit rates.
- **A frontend language** for programs that make several LLM calls, sharing prefixes across them, such as parallel branches of an agent.
- **Overlap scheduling**: the CPU prepares the next batch while the GPU runs the current one, hiding scheduler overhead.
- Strong focus on large MoE models such as DeepSeek, with expert parallelism and disaggregation at scale.

## How their prefix caches differ in practice

For most workloads the hit rates are similar. Differences appear at the edges:

- The radix tree matches at token granularity and makes shared structure explicit, which helps scheduling decisions.
- Hash-chained blocks are simpler and match only full blocks, losing up to one block per request.
- Cache-aware scheduling can raise hit rates under load at some cost in fairness.

## What to check for your decision

These details change from release to release, so check the current versions:

- **Model support on day one** for the architectures you use, including attention variants and quantization formats.
- **Performance on your workload**, using the fair method from lesson 17.01. Prefix-heavy traffic, MoE models and long contexts are where differences tend to appear.
- **Operational fit**: metrics, configuration, logging, deployment tooling, integration with your router and autoscaler.
- **Community and support**: how fast issues get answered and how often releases break things you rely on.

<div class="ex">
<div class="lab">Worked example</div>

**a decision that went both ways**

A team ran an agent product with heavy prefix sharing on a DeepSeek-style MoE model, and a separate chat product on a dense 8B model.

On the agent workload, SGLang's cache-aware scheduling gave a 12% higher prefix hit rate and 9% higher SLO-safe capacity.

On the chat workload, the two engines were within 3% of each other, and the team's existing tooling was built around vLLM.

They ran both: SGLang for the agent fleet and vLLM for chat. The deciding factors were measured, not assumed.

</div>

## Deep dive: block-level versus token-level matching

vLLM's prefix cache matches whole blocks by hash. SGLang's radix tree matches token by token. For a shared prefix of ⟦m880⟧ tokens, the reusable part is

⟦M879⟧

The difference is at most ⟦m881⟧ tokens per request, on average about ⟦m882⟧. At block size 16 that is about 8 tokens: negligible for a 2,000-token prefix, but up to half the reuse for many short shared prefixes, such as a 20-token instruction in front of every request. In practice, the bigger differences come from eviction policy and from how the scheduler orders requests to keep shared prefixes hot.

<div class="take">
<div class="lab">Key takeaway</div>

vLLM and SGLang share most core techniques and differ in prefix cache design, scheduling details and ecosystem. Choose by measurements on your workload and by operational fit, not by reputation.

</div>
