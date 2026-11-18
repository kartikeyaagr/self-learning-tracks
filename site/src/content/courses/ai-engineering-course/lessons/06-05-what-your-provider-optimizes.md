---
id: "06.05"
course: "ai-engineering-course"
module: "06"
slug: "what-your-provider-optimizes"
title: "What your provider optimizes"
keyIdeas: ["Five serving optimizations shape what you pay and wait for: paged KV cache with prefix caching, FlashAttention, low-precision weights (FP8, MXFP4, NVFP4), speculative decoding and MoE serving.", "Prefix caching is the one an API user controls most: a stable, byte-identical prefix turns repeated input into cheap cached reads, and one changing token at the top turns it into expensive writes.", "Quantization is where quality can silently move: the same open-weights model can score differently at two providers, so evaluate the endpoint, not the model name.", "Speculative decoding speeds up predictable output at low batch and changes nothing about quality; at high batch the gain shrinks.", "Diagnose from the symptom (slow first token, slow streaming, low throughput, out of memory) before picking a fix."]
question: "Your app sends a 6,000-token system prompt plus tools on every request, then a 300-token user turn, and gets 400 tokens back. Cached input is billed at a tenth of the input price and cache writes at 1.25 times. Explain what prefix caching changes in cost and latency, the mistake that would make it cost more than no caching, and which serving optimization you would suspect if a cheaper provider's copy of the same open model fails more tool calls."
answer: "With a warm cache only the 300 new tokens are prefilled at full price; the 6,000-token prefix is read from stored KV blocks, so input cost for that part drops about 90% and time to first token falls because prefill skips those tokens. At illustrative prices of 3 per million input and 15 per million output this takes a request from about 0.025 to 0.0087 dollars, and output becomes most of the bill. The mistake is putting anything that changes per request (a timestamp, the user's name, a shuffled tool list) before the stable part: every request then misses and pays the write premium, about 0.029 dollars, more than not caching. For the tool-call failures, suspect a different weight or KV precision (FP8 or 4-bit), or a different chat template or tool parser, and test the endpoint on your own tool-call eval rather than trusting the model name."
rubric: ["Explains that a cache hit skips prefill for the prefix, cutting both input cost and time to first token", "Computes or estimates the per-request saving and notes output tokens then dominate", "Names the cache-breaking mistake (changing content before the stable prefix) and why it costs more than no cache", "Attributes cross-provider quality differences to quantization or serving differences, not the model itself", "Proposes measuring the specific endpoint on an eval set"]
sources: [{"title": "Kwon et al.: Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)", "url": "https://arxiv.org/abs/2309.06180", "note": "Paged KV blocks, block sharing and the throughput gain they enable."}, {"title": "Anthropic: Prompt caching", "url": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching", "note": "Cache breakpoints, TTLs, and the 1.25x write / 0.1x read multipliers."}, {"title": "NVIDIA: Introducing NVFP4 for efficient and accurate low-precision inference", "url": "https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/", "note": "NVFP4 versus MXFP4: 16-value blocks with E4M3 scales on Blackwell."}, {"title": "Leviathan et al.: Fast Inference from Transformers via Speculative Decoding (2022)", "url": "https://arxiv.org/abs/2211.17192", "note": "Draft-then-verify with an unchanged output distribution."}]
practice: ["prompt-cache-cost", "prefix-cache-matching-and-reuse", "ttft-tpot-itl-and-token-throughput", "mxfp4-dequantize", "speculative-decoding-loop"]
tools: ["vLLM", "SGLang", "TensorRT-LLM", "Claude API", "OpenAI API"]
tags: ["Inference", "Cost", "Latency"]
---

## Five optimizations you can see from outside

You rarely run the GPUs behind your app, but you pay for what happens on them. Every hosted endpoint and self-hosted engine (vLLM, SGLang, TensorRT-LLM) stacks the same few optimizations, and each leaves fingerprints in your bill, latency graphs or eval scores. For each: what you can observe, which knobs you hold, and where Inference Engineering has the internals.

Two facts from [one model call](lesson:aieng:01.03) and the [KV cache](lesson:aieng:04.04) explain most of what follows. Prefill processes the prompt in parallel and sets time to first token (TTFT). Decode reads every weight and the whole KV cache for each token, is memory-bandwidth bound, and sets time per output token (TPOT). Price comes from how many sequences fit in a batch.

| Optimization | What changes for you | What you control |
| --- | --- | --- |
| Paged KV + prefix caching | Cached-input discount, lower TTFT on repeated prefixes | Prompt order, cache breakpoints, TTL; engine flags when self-hosting |
| FlashAttention | Long context is affordable at all; nothing visible per request | Nothing on an API; engine picks the kernel |
| Low-precision weights and KV | Lower price, faster decode, possible quality shift | Provider and tier choice; quantization flags when self-hosting |
| Speculative decoding | Faster streaming on predictable text, same output distribution | Predicted-output parameters; draft config when self-hosting |
| MoE serving | Big-model quality at small-model price per token | Model choice; memory planning when self-hosting |

~~~dg-flow Where each optimization acts on one request.
Request -(hash prefix blocks)-> *Prefix cache hit? -(prefill the rest, FlashAttention)-> First token -(decode: quantized weights, speculative drafts)-> Streamed tokens -(blocks freed)-> Paged KV pool
~~~

## Paged KV cache and prefix caching

Older servers reserved one contiguous KV region per request, sized for the maximum length; the vLLM paper measured that only 20–38% of that memory held real tokens. Paging stores the cache in fixed-size blocks (16 tokens by default in vLLM) mapped through a per-sequence block table, so waste falls to the unfilled tail of the last block and the freed memory becomes a bigger batch. [PagedAttention](lesson:ieng:08.03) has the allocator.

Requests that start with the same tokens can share physical blocks. Engines index finished blocks by a hash of their tokens and everything before them ([automatic prefix caching](lesson:ieng:10.04)); a new request reuses the longest matching prefix and prefills only the remainder. Hosted APIs expose this as cached input tokens, billed at a fraction of the input price, sometimes with a premium on the first write. Anthropic documents 1.25× for a 5-minute write, 2× for a 1-hour write and 0.1× for reads; other providers cache automatically above a minimum prompt length.

<div class="ex">
<div class="lab">Worked example</div>

**A 6,000-token system prompt and tool list on every call.** Each request adds a 300-token user turn and returns 400 tokens. Illustrative prices, an earlier Sonnet-class list price (current models differ, check the pricing page): \$3 per million input, \$15 per million output, \$0.30 cached read, \$3.75 five-minute write.

| Case | Input cost | Output cost | Per request | At 2M requests a month |
| --- | --- | --- | --- | --- |
| No caching | 6,300 × \$3/M = \$0.0189 | \$0.0060 | \$0.0249 | \$49,800 |
| Warm cache | 6,000 × \$0.30/M + 300 × \$3/M = \$0.0027 | \$0.0060 | \$0.0087 | \$17,400 |
| Timestamp at the top (every call misses) | 6,000 × \$3.75/M + \$0.0009 = \$0.0234 | \$0.0060 | \$0.0294 | \$58,800 |

Caching cuts the bill 65%, and output becomes 69% of what is left, so the next saving is shorter answers, not shorter prompts. The third row is the classic bug: one changing value before the stable prefix makes every call a write, and caching now costs 18% more than not caching. TTFT drops on hits too, because the 6,000 prefix tokens are never prefilled.

</div>

What you control: put tools, system prompt and long documents first and keep them byte-identical (sorted tool lists, no dates); put per-request content last; place explicit breakpoints where the API asks for them; check the cached-token field in every usage response. Self-hosting, prefix caching is on by default in current vLLM (`--enable-prefix-caching`) and SGLang (RadixAttention), and routing requests that share a prefix to the same replica matters as much as the flag. [Cost budgeting](lesson:aieng:12.04) has the TTL and multi-turn arithmetic.

## FlashAttention

Attention is memory-bound: standard attention writes the full $T \times T$ score matrix to GPU main memory and reads it back for softmax. FlashAttention tiles queries, keys and values into on-chip SRAM, uses an online softmax (running max and sum, rescaled as blocks arrive) and never stores the full matrix. The result is exact attention with memory linear in sequence length, which is a large part of why 128k-token contexts are sold at all. [FlashAttention](lesson:ieng:14.03) walks through the tiling.

From outside there is nothing to switch on; its effect is prefill that stays practical at 100k tokens. If a self-hosted engine falls back to a slow attention path (unsupported head size or GPU), long-prompt TTFT is where you notice.

## Low precision: FP8, MXFP4, NVFP4

Weight memory is $M = N \cdot b / 8$ bytes for $N$ parameters at $b$ bits, and decode speed tracks bytes read per token. Fewer bits mean fewer GPUs and faster streaming. The formats you will meet in 2025–26 model cards and provider pages:

| Format | Value bits | Scale | Effective bits | Hardware | Where you see it |
| --- | --- | --- | --- | --- | --- |
| BF16 | 16 | none | 16 | all | Default checkpoints |
| FP8 (E4M3) | 8 | per tensor, channel or 128×128 block | ~8 | Hopper and newer | DeepSeek-V3.2 ships FP8 weights |
| MXFP4 | 4 (E2M1) | one power-of-two scale per 32 values | 4.25 | Blackwell native; emulated elsewhere | gpt-oss MoE weights |
| NVFP4 | 4 (E2M1) | E4M3 scale per 16 values, plus a tensor scale | ~4.5 | Blackwell | NVIDIA-quantized checkpoints |

For a 70B dense model that is 141 GB in BF16, 71 GB in FP8 and about 40 GB in NVFP4. NVIDIA reports 1% or less accuracy loss from FP8 to NVFP4 on DeepSeek-R1-0528; gpt-oss was post-trained with MXFP4 so its published evals already include the quantization. KV-cache precision is a separate choice: an FP8 cache halves its memory and doubles how many long sequences fit ([KV cache quantization](lesson:ieng:11.05)).

The risk is that quality moves where your eval looks and a general benchmark does not. Losses show up first in long multi-step maths, code, long-context recall, rarer languages and tool calling. Moonshot's K2 Vendor Verifier found large differences in tool-call schema accuracy between providers serving the same open weights; precision is one cause, chat templates and tool parsers are others. Whatever the cause, you evaluate the endpoint, not the model name. Knobs: pick providers that state their precision, prefer a first-party endpoint for agentic work, and self-hosting, use the `--quantization` and `--kv-cache-dtype fp8` flags and measure with [quality regression tests](lesson:ieng:11.07). Formats and methods (GPTQ, AWQ, SmoothQuant) are in [BF16, FP8, INT8 and 4-bit inference](lesson:ieng:11.02) and [AWQ, GPTQ and SmoothQuant](lesson:ieng:11.03).

## Speculative decoding

Decode leaves compute idle while it waits on memory, so a cheap draft (a small model, extra heads such as EAGLE, or n-grams copied from the prompt) proposes $k$ tokens and the target model checks all of them in one pass. A rejection-sampling rule keeps the output distribution exactly the target's. With per-token acceptance rate $\alpha$:

$$
\mathbb{E}[\text{tokens per target pass}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}
$$

At $\alpha = 0.7$ and $k = 4$ that is 2.77 tokens per pass, about 2.3× faster after paying for a draft that costs 5% of a target step. You observe streaming speed that varies with content: fast on code edits, JSON and quoted input, slower on prose. Some APIs let you pass the expected output (OpenAI's Predicted Outputs) for edit-style tasks; rejected predicted tokens are still billed as output. Self-hosting, vLLM takes `--speculative-config` with a method (`ngram`, `eagle3`, a draft model) and `num_speculative_tokens`. The gain shrinks at large batch, where the GPU is no longer idle ([acceptance and draft length](lesson:ieng:16.03), [Medusa, EAGLE and multi-token prediction](lesson:ieng:16.06)).

## MoE serving and service tiers

Most frontier-class open models are mixtures of experts: memory scales with total parameters, compute per token with active ones. That is why a 117B-parameter gpt-oss can be priced like a small model. [MoE for app engineers](lesson:aieng:06.06) covers what that means for model choice and self-hosting; [MoE inference](lesson:ieng:19.01) covers the engine side.

Providers also sell one model at several latency–price points: batch APIs (typically half price, up to 24-hour turnaround), flex tiers at batch rates with occasional unavailability, and priority tiers that cost more for steadier latency. Evals and backfills belong on the cheap tiers.

## From symptom to fix

Measure TTFT and TPOT at p50 and p95, throughput at your latency target, and memory headroom, then look up the worst one.

| Symptom | Bottleneck | Try first | Read |
| --- | --- | --- | --- |
| Slow TTFT on long prompts | Prefill compute | Prefix caching, send less context, chunked prefill | [ieng 10.04](lesson:ieng:10.04) |
| Slow TPOT at small batch | Decode bandwidth | Lower-precision weights, speculative decoding, smaller or distilled model | [ieng 16.03](lesson:ieng:16.03) |
| Low throughput, high cost per token | Batch too small | Continuous batching, paged KV, FP8 KV cache | [ieng 08.03](lesson:ieng:08.03) |
| Out of memory | Weights plus KV | Quantize weights and KV, GQA/MLA models, tensor parallelism | [ieng 03.06](lesson:ieng:03.06) |
| Long context too slow at any batch | Attention over all tokens | Sliding-window or sparse-attention models, retrieval instead of stuffing | [KV cache](lesson:aieng:04.04) |
| Need big-model quality at small-model cost | Parameters versus FLOPs | An MoE model | [06.06](lesson:aieng:06.06) |

## Where this breaks

- **Prefix caching across replicas.** Round-robin load balancing finds a cold cache on every replica; use prefix-aware routing.
- **Bigger batches raise TPOT.** Throughput tuning trades against each user's streaming speed; fix the latency target first.
- **Speculation at high load.** The draft's work competes with real requests and can lower throughput.
- **Quantization judged on perplexity.** It barely moves at 4 bits while tool calling drops.

## What interviewers probe

- Why a cache hit lowers TTFT and cost but not TPOT.
- What breaks a prompt cache and how you would detect it in usage logs.
- When speculative decoding stops helping, and why it does not change quality.
- Given a symptom, which bottleneck and which first fix.

<div class="take">
<div class="lab">Key takeaway</div>

Prefix caching is the optimization you control most, and a stable prefix plus a cached-token check in your logs captures it. Quantization is where quality can quietly change, so evaluate each endpoint. For everything else, start from the symptom and look up the bottleneck before reaching for a fix.

</div>
