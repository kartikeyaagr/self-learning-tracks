---
id: "07.05"
course: "ai-engineering-course"
module: "07"
slug: "match-the-optimization-to-the-bottleneck"
title: "Match the optimization to the bottleneck"
keyIdeas: ["Name the symptom first: slow time to first token (TTFT), slow time per output token (TPOT), low throughput, or out-of-memory.", "Slow TTFT on long prompts: prefill is compute-bound. Try prompt/prefix caching, shorter context (better retrieval), FlashAttention, chunked prefill.", "Slow TPOT: decode is memory-bandwidth bound. Try weight quantization, speculative decoding, a smaller or distilled model, GQA models.", "Low throughput / cost per token: batch more. Paged attention, continuous batching, KV cache quantization, MoE trade-offs.", "Out of memory: quantize weights and KV cache, paging, tensor parallelism across GPUs.", "Measure before and after on your real traffic; an optimization for one bottleneck can hurt another (e.g. bigger batches raise TPOT)."]
question: "Your chatbot has slow first tokens on long documents, but streaming speed is fine. Which bottleneck is it, and which two optimizations would you try first?"
answer: "Slow first token but fine streaming means prefill is the bottleneck: processing the long prompt is compute-heavy (attention cost grows with length). First fixes: prefix/prompt caching so repeated documents or system prompts aren't re-prefilled, and cutting the context sent (retrieve relevant chunks instead of whole documents). Also ensure FlashAttention and chunked prefill. Quantization and speculative decoding mainly help decode, so they're not the first choice here."
sources: [{"title": "Horace He: Making Deep Learning Go Brrrr From First Principles", "url": "https://horace.io/brrr_intro.html", "note": "Compute, memory bandwidth and overhead as the three bottlenecks."}, {"title": "Databricks: LLM Inference Performance Engineering: Best Practices", "url": "https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices", "note": "TTFT, TPOT, throughput and how batching and quantization move them."}, {"title": "NVIDIA: Mastering LLM Techniques: Inference Optimization", "url": "https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/", "note": "Overview of batching, KV cache, parallelism and quantization."}]
tags: ["Exercise"]
---

## Start from the symptom

Every optimization in modules 4 to 7 targets one bottleneck. Applying them at random wastes effort and sometimes makes things worse. Databricks' inference best-practices post uses these metrics:

- **TTFT** (time to first token): how long before streaming starts. Mostly prefill.
- **TPOT** (time per output token): streaming speed. Mostly decode.
- **Throughput**: tokens per second across all users, which sets cost.
- **Memory headroom**: whether the model and enough KV cache fit at all.

## The map

| Symptom | Bottleneck | Try first |
| --- | --- | --- |
| Slow TTFT on long prompts | Prefill compute (attention grows with n^2) | Prefix / prompt caching, send less context, FlashAttention, chunked prefill |
| Slow TPOT | Decode memory bandwidth | Weight quantization, speculative decoding, smaller or distilled model |
| Low throughput, high cost | Batch too small | Continuous batching, paged attention, KV cache quantization |
| Out of memory | Weights + KV cache | Quantize weights and KV, GQA/MLA models, tensor parallelism |
| Long context too slow at all | Quadratic attention | Sliding-window / sparse attention models, retrieval instead of stuffing |
| Want big-model quality at small-model compute | Parameters vs FLOPs | Mixture-of-experts model |

~~~dg-flow Diagnose before optimizing.
Measure TTFT, TPOT, throughput, memory -(find worst)-> *Name the bottleneck -(pick from map)-> One change -(re-measure)-> Keep or revert
~~~

## Trade-offs to watch

- **Bigger batches** raise throughput but raise each user's TPOT.
- **Quantization** speeds decode and frees memory, but can cost accuracy on maths and code.
- **Speculative decoding** helps most at small batch sizes; at very large batches the GPU is no longer idle and gains shrink.
- **MoE** cuts compute per token but needs more memory and inter-GPU communication.
- **Distillation** is the biggest win for a narrow task, and the most work.

<div class="ex">
<div class="lab">Worked example</div>

**slow first token on long documents**

Streaming is fine, so decode isn't the problem. The first token is slow only when documents are long: prefill bound.

Try first: (1) prefix caching, if many requests share the same document or system prompt; (2) send less: retrieve the relevant chunks instead of the whole document.

Then check the server uses FlashAttention and chunked prefill so one long prompt doesn't stall others.

</div>

## Find the right fix from the symptom

Tick the symptoms you see in your deployment. The figure ranks optimizations by how many of your symptoms they address.

~~~dg-raw
%%widget diagnose From observed symptoms to the optimizations that address them.
~~~

## Measure first: four numbers

Before choosing anything, measure these on real traffic:

1. **Time to first token (TTFT)** at p50 and p99. Dominated by queueing and prefill.
2. **Inter-token latency** (time per output token). Dominated by decode and batch size.
3. **Throughput** in tokens per second per GPU at your latency target.
4. **GPU memory headroom**: how much is weights, how much KV cache, how much free.

Then match: long TTFT with long prompts points at prefill (prefix caching, FlashAttention, chunked prefill). Slow inter-token latency at small batch points at decode memory bandwidth (quantization, speculative decoding). Low throughput with memory full points at the KV cache (paging, GQA, FP8 KV, quantized weights to free space).

<div class="ex">
<div class="lab">Worked example</div>

**two teams, same model, opposite fixes**

Team A runs an internal coding assistant: 20 users, long prompts, short answers. Their TTFT is 4 s. Prefix caching of the repository context cuts it to 0.6 s. Speculative decoding would barely help.

Team B runs a public chatbot: 2,000 concurrent users, short prompts, long answers, GPUs at full memory. Paged attention plus FP8 KV doubles the batch size and halves cost per token. Prefix caching helps a little; speculative decoding hurts at their batch size.

Same model, same hardware. The workload picks the fix.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

measure TTFT, TPOT, throughput and memory, name the bottleneck, then pick the optimization built for it. Prefill problems and decode problems have different fixes.

</div>
