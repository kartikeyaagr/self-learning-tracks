---
id: "11.04"
course: "ml-system-design"
module: "11"
slug: "case-study-cost-optimised-llm-inference"
title: "Case study: cost-optimised LLM inference"
keyIdeas: ["LLM inference has two phases: prefill (processing the prompt, compute-bound) and decode (one token at a time, memory-bandwidth-bound), with different optimisations.", "Throughput and cost are dominated by batching and KV-cache memory: continuous batching and paged KV caches keep GPUs busy without wasting memory.", "Quantisation, prefix caching, speculative decoding and routing easy requests to smaller models reduce cost further, each with a quality check.", "The design target is cost per request under latency objectives (time to first token and time per output token), measured at realistic traffic."]
question: "Design an inference service for a 70-billion-parameter chat model that handles 2,000 requests per second at peak, with average prompts of 1,500 tokens and responses of 300 tokens. Targets: time to first token under 1 second at p95 and 30 ms per output token. Minimise cost. Explain batching, memory, model optimisations, routing and capacity planning."
answer: "Estimate load: 600,000 output tokens per second and 3 million prompt tokens per second. Serve the model in 8-bit or 4-bit weights with tensor parallelism across a few GPUs per replica. Use continuous batching so new requests join at each decode step, and a paged KV cache so memory is not reserved for the maximum length; KV memory per token (layers times 2 times KV heads times head size times bytes) decides how many sequences fit. Separate or chunk prefill so long prompts do not stall decoding. Cache shared prefixes such as system prompts. Add speculative decoding with a small draft model where acceptance is high. Route simple requests to a smaller model using a classifier, with quality checked on a sample. Plan capacity from a load test of tokens per second per replica at the latency targets, plus headroom, with autoscaling on queue depth. Monitor TTFT, TPOT, GPU utilisation, KV-cache usage, cost per thousand tokens and quality on a fixed evaluation set."
rubric: ["Separates prefill and decode and explains their different bottlenecks", "Computes token throughput and KV-cache memory per sequence from the numbers given", "Uses continuous batching and paged KV caches", "Handles long prompts with chunked or disaggregated prefill to protect decode latency", "Applies quantisation, prefix caching and speculative decoding with quality checks", "Routes requests across model sizes with a quality safeguard", "Plans capacity from load tests at the latency objectives, with autoscaling", "Monitors latency percentiles, utilisation, cost per token and output quality"]
sources: [{"title": "Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)", "url": "https://arxiv.org/abs/2309.06180", "note": "Paged KV caches and continuous batching."}, {"title": "Leviathan, Kalman and Matias, Fast Inference from Transformers via Speculative Decoding (2022)", "url": "https://arxiv.org/abs/2211.17192", "note": "Using a small draft model to speed up decoding."}, {"title": "Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (2022)", "url": "https://arxiv.org/abs/2210.17323", "note": "Weight quantisation with small quality loss."}, {"title": "Hugging Face, Text Generation Inference: PagedAttention", "url": "https://huggingface.co/docs/text-generation-inference/conceptual/paged_attention", "note": "A practical explanation of paged KV caches."}]
practice: ["plan-kv-cache-capacity", "simulate-continuous-batching", "optimize-throughput-and-cost-under-latency-slo", "speculative-decoding-acceptance"]
tags: ["LLM serving", "batching", "KV cache", "quantisation", "routing"]
---

## Requirements

Clarify traffic and targets. Assume 2,000 requests per second at peak, average prompts of 1,500 tokens and outputs of 300 tokens, time to first token (TTFT) under 1 second at p95, and time per output token (TPOT) under 30 ms. The objective is cost per request while meeting these, with no measurable quality loss on an evaluation set.

## Load in tokens

- Prompt tokens: $2{,}000 \times 1{,}500 = 3$ million per second (prefill).
- Output tokens: $2{,}000 \times 300 = 600{,}000$ per second (decode).
- Concurrent sequences: each request decodes for about $300 \times 0.03 = 9$ seconds, so about 18,000 sequences are in flight at peak.

## Prefill and decode

Prefill processes all prompt tokens in parallel and is compute-bound. Decode generates one token per sequence per step; each step reads all the weights and the KV cache, so it is memory-bandwidth-bound and benefits from large batches, because many sequences share each weight read.

## KV-cache memory

Per token, the KV cache holds $2 \times L \times h_{kv} \times d \times b$ bytes. For a 70B model with 80 layers, 8 KV heads (grouped-query attention), head size 128 and 16-bit values, that is $2 \times 80 \times 8 \times 128 \times 2 \approx 330$ KB per token. A sequence of 1,800 tokens needs about 590 MB. With 18,000 sequences in flight, the cache totals about 10 TB across the fleet, which sets the number of replicas as much as compute does. Quantising the KV cache to 8 bits halves this.

## Batching and memory management

- Continuous batching: at each decode step, finished sequences leave and new ones join, so the batch stays full.
- Paged KV cache: allocate cache in small blocks on demand rather than reserving the maximum length per sequence, so far more sequences fit.
- Chunked prefill, or separate prefill and decode pools, so a long prompt does not stall every sequence's next token.

<div class="ex">
<div class="lab">Worked example</div>

**the long prompts that hurt everyone**

A service saw TPOT jump from 25 ms to 120 ms whenever a few users sent 30,000-token documents. Each long prefill ran inside the shared batch and blocked decoding for everyone on that replica. Splitting prefill into 2,000-token chunks interleaved with decode steps brought TPOT back under 35 ms at p99, with a modest increase in TTFT for the long requests only.

</div>

## Model optimisations

- Weight quantisation to 8 or 4 bits shrinks memory and speeds decode; check quality on the evaluation set.
- Prefix caching: many requests share a long system prompt; reuse its KV cache instead of recomputing it.
- Speculative decoding: a small draft model proposes several tokens and the large model verifies them in one pass; it helps most at low batch sizes and when acceptance is high.
- Routing: a classifier sends simple requests to a smaller model; sample both paths for quality review.

## Serving

~~~dg-flow Inference service. Routing and prefix caching cut work before the expensive model runs.
Requests -(router: size, priority)-> Small model pool
Requests -(router)-> Prefix cache lookup -(new tokens only)-> Prefill (chunked) -(KV blocks)-> *Decode with continuous batching -(tokens, streamed)-> Clients
~~~

## Capacity planning

Load-test one replica with realistic prompt and output length distributions and find the highest throughput at which p95 TTFT and TPOT still meet targets. Divide peak load by that and add headroom (30% is common) for spikes and failures. Autoscale on queue depth and KV-cache use, since GPU utilisation alone hides memory pressure.

## Monitoring

Track TTFT and TPOT percentiles, tokens per second per GPU, KV-cache utilisation and evictions, router decisions, cost per thousand tokens, and quality scores on a fixed evaluation set after each optimisation.

## Likely follow-up questions

- "Is a bigger batch always cheaper?" Up to the point where TPOT breaks the target or the KV cache runs out; beyond that, latency rises sharply.
- "When does disaggregating prefill and decode pay off?" When prompts are long and varied, so that prefill interference dominates decode latency.

## Trade-offs

- Batch size versus per-token latency.
- Quantisation and routing versus quality.
- Headroom versus idle cost.

<div class="take">
<div class="lab">Key takeaway</div>

size LLM serving in tokens and KV-cache bytes, keep decode batches full with continuous batching and paged caches, protect decode from long prefills, and cut cost with quantisation, prefix caching, speculative decoding and routing, each checked against quality.

</div>
