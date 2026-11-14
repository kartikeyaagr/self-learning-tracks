---
id: "03.07"
course: "inference-eng-course"
module: "03"
slug: "lab-cached-vs-uncached-generation-model-fit"
title: "Lab: Cached vs Uncached Generation & Model Fit"
keyIdeas: ["Compare your cache-free loop with your cached loop on the same prompts.", "Plot step time vs position for both.", "Predict KV bytes per token from config.json, then check against measured memory growth.", "Predict max concurrency for a context length and test it until OOM or preemption.", "Explain the gap between prediction and measurement."]
question: "Your measured max concurrency is 30% lower than your KV formula predicted. Name three plausible reasons."
answer: "Activation and CUDA graph memory took more than assumed, leaving a smaller KV pool; block-level allocation wastes the unused part of each sequence's last block; and the engine reserves blocks for the maximum possible output length or keeps a watermark of free blocks to avoid preemption. Prompt lengths in the test may also be longer than the assumed average."
sources: [{"title": "vLLM documentation", "url": "https://docs.vllm.ai/en/latest/", "note": "Startup logs show KV cache size."}, {"title": "Hugging Face: KV cache strategies", "url": "https://huggingface.co/docs/transformers/main/en/kv_cache", "note": "Cache options to compare."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Prove to yourself that the KV cache formulas predict reality. You will compare cached and uncached generation, predict memory growth from the config file, and find the concurrency limit of a real engine. Being within 10 to 20% of your predictions means your mental model is good enough to size hardware without trial and error.

## Part 1: cached and uncached generation

Use the loops from lessons 02.05 and 03.04. Run both with the same 500-token prompt, greedy decoding and 256 new tokens.

- Record the time of every step.
- Confirm the generated tokens match.
- Plot step time against position for both loops on one chart.

Expected result: the uncached loop's step time climbs steadily with position. The cached loop's stays nearly flat. Total time differs by an order of magnitude for long outputs.

## Part 2: predict memory growth

From `config.json`, compute KV bytes per token for your model. For Llama 3 8B, that is 128 KB. Then generate 2,000 tokens with a dynamically growing cache, and read `torch.cuda.memory_allocated()` every 100 tokens.

- If the cache grows token by token, memory should rise by about 12.5 MB per 100 tokens (128 KB × 100).
- If the cache is pre-allocated, you should see one jump at the start and a flat line afterwards.
- If the library grows the cache by concatenating tensors, you may see a sawtooth pattern as old buffers are freed and new, larger ones allocated.

## Part 3: predict model fit and concurrency

Fill in the predicted column before you run anything:

1. Weight memory: parameters × bytes per parameter.
2. KV pool: 0.9 × total memory − weights − about 3 GB for activations, graphs and context.
3. Maximum concurrent sequences at 2,000 and 8,000 tokens per sequence.

Then start vLLM or SGLang with your model. Read the KV capacity from the startup log. Run a load test with fixed 2,000-token sequences (for example 1,500 in and 500 out) and raise concurrency until the engine starts queueing or preempting requests. Repeat at 8,000 tokens.

| Item | Predicted | Measured | Gap |
| --- | --- | --- | --- |
| Weights | | | |
| KV bytes per token | | | |
| KV pool (from log) | | | |
| Max concurrency at 2k | | | |
| Max concurrency at 8k | | | |

<div class="ex">
<div class="lab">Worked example</div>

**a typical result**

Predicted for Llama 3 8B on one H100: pool 53 GB, 414,000 tokens, 207 sequences at 2k.

vLLM's log reported 49.6 GB of KV cache, or 387,000 tokens. The measured onset of preemption was at about 190 concurrent 2k-token requests.

The 7% gap came from a larger activation peak than assumed and from block rounding: each sequence's last block is partly empty, so 2,000 tokens occupy 2,000 rounded up to a multiple of 16.

</div>

## Explaining the gaps

Common causes of a mismatch:

- **Activation peak** larger than your estimate, because of `max_num_batched_tokens` or the sampler's buffers.
- **Block rounding**: each sequence wastes on average half a block.
- **Reserved space**: some engines keep a few blocks free as a safety margin, or reserve space for growth before admitting a request.
- **Allocator fragmentation** in long-running processes.
- **A different dtype than you assumed**: a model loaded in FP32 by mistake doubles everything.

## Write-up

Half a page: your predictions, your measurements, the gaps and their causes. Add one recommendation, such as "with FP8 KV cache this GPU would serve about 380 concurrent 2k-token requests", and say how you would verify it.

## Deep dive: what to plot, and the shape to expect

Three plots tell you whether your model of memory and time is right.

1. **Step time against position, cached and uncached.** Cached should be nearly flat (slight upward slope from KV reads); uncached should rise roughly linearly at first.
2. **Allocated memory against position.** It should rise in a straight line with slope ⟦m341⟧ per token. Fit the slope and compare with ⟦m342⟧.
3. **Maximum batch before out-of-memory against context length.** Should follow ⟦m343⟧, a straight line of slope −1 on a log-log plot.

<div class="ex">
<div class="lab">Worked example</div>

**checking the memory slope**

You record allocated memory every 100 tokens while generating 2,000 tokens at batch 1. A linear fit gives a slope of 131,200 bytes per token.

Predicted: ⟦m344⟧ bytes. The 0.1% difference is allocator rounding. The model is right.

If the slope had been 524,288, the model would be running without GQA, or your cache would be storing all 32 heads.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

predict first, then measure. When your formula and the engine's log agree within 20%, you can size hardware and set concurrency limits with confidence.

</div>
