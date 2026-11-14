---
id: "10.02"
course: "inference-eng-course"
module: "10"
slug: "chunked-prefill-choosing-the-chunk-size"
title: "Chunked Prefill & Choosing the Chunk Size"
keyIdeas: ["Split long prompts into chunks; run one chunk per step with decodes.", "Step time stays bounded, so ITL stays smooth.", "In vLLM the chunk size is effectively max_num_batched_tokens minus decode tokens.", "Too small: slower TTFT and lower prefill efficiency. Too large: stalls return.", "Pick the largest chunk that keeps step time under your ITL target."]
question: "How does chunked prefill fix interference, and what is the cost of a chunk size that is too small?"
answer: "It splits a long prompt into chunks of at most N tokens and processes one chunk per step alongside the ongoing decodes, so each step stays bounded and decoding users keep getting tokens. A chunk that is too small means many steps to finish the prompt, which raises that request's TTFT and loses some prefill efficiency, since small matmuls use the GPU less well and attention over previous chunks must be re-read each step."
sources: [{"title": "Sarathi-Serve", "url": "https://arxiv.org/abs/2403.02310", "note": "Chunked prefill and stall-free batching."}, {"title": "vLLM: Optimization and tuning", "url": "https://docs.vllm.ai/en/latest/configuration/optimization.html", "note": "Chunked prefill settings."}]
tags: ["prefill", "chunked"]
---

## How chunked prefill works

Chunked prefill splits a long prompt into pieces and processes one piece per step, alongside the ongoing decodes. Each step stays bounded, so decoding users keep receiving tokens.

~~~dg-flow A 6,000-token prompt with a 2,048-token budget.
Step 1: decodes + chunk 1 -> Step 2: decodes + chunk 2 -> *Step 3: decodes + chunk 3 -> First token
~~~

In each step:

- Every running decode sequence contributes one token.
- The remaining token budget goes to the next chunk of the long prompt.
- The chunk's tokens compute Q, K and V, write K and V into the cache, and attend to themselves plus every earlier chunk of the same prompt, which is already in the cache.
- Only after the last chunk does the prompt's final position produce logits, and the first output token is sampled.

## What it costs

Chunking is not free, but the costs are small for sensible chunk sizes.

- **Matmul efficiency.** A matmul with 512 or more rows already runs near the compute roof on most GPUs. Splitting a 16,000-token prefill into 2,048-token chunks loses only a few percent of matmul efficiency.
- **Extra attention reads.** Chunk n must read the K and V of chunks 1 to n − 1 from memory. For a prompt split into c chunks, the total KV read is about c ÷ 2 times the prompt's KV size. For very long prompts with small chunks, this adds up.
- **More steps.** TTFT for the long prompt rises, because it now spans several steps, each shared with decodes.

## Choosing the chunk size

| Chunk budget | Step time | TTFT of the long prompt | ITL p99 for others |
| --- | --- | --- | --- |
| 512 | short | high | best |
| 2,048 | moderate | moderate | good |
| 8,192 | long | low | worse |
| unlimited | very long | lowest | stalls |

A practical rule: measure the step time for one full chunk plus your typical decode batch. Pick the largest chunk whose step time keeps ITL within its target.

~~~dg-raw
%%widget roof Prefill mode, with the x-axis as tokens per step. Past the ridge each extra token costs real time, so chunks much larger than the ridge stretch the step.
mode = Prefill
n = 11
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**sizing the chunk from an ITL target**

The ITL p99 target is 60 ms. A pure decode step with 100 sequences takes 15 ms on an 8B model.

Each prefill token costs about 16 GFLOP ÷ 600 TFLOP/s ≈ 27 µs. The step can absorb (60 − 15) ÷ 0.027 ≈ 1,670 prefill tokens.

A budget of 1,536 or 2,048 is about right. With 2,048 the step takes about 70 ms in the worst case, slightly over target, so 1,536 is the safer choice.

A 16,000-token prompt now needs about 11 steps of 55 to 60 ms each, so its TTFT is about 650 ms instead of about 450 ms unchunked. That is the price of smooth streams for everyone else.

</div>

## How it fits the scheduler

Chunked prefill pairs naturally with the decode-first token budget from lesson 07.04:

1. Decodes take their tokens first.
2. Partially prefilled prompts continue with the next chunk.
3. New prompts start with whatever budget remains.

In recent vLLM versions chunked prefill is on by default, and the token budget is `max_num_batched_tokens`. SGLang exposes a chunked prefill size setting. TensorRT-LLM supports it as well.

## When chunking is not enough

Chunking bounds each step, but long prefills still consume compute that decodes could have used, and TTFT for long prompts rises. When prefill load is heavy, for example with many long RAG prompts, two further options exist:

- **Prefix caching**, so repeated parts of prompts are not prefilled at all (the next lessons).
- **Prefill-decode disaggregation**, which runs prefill on separate GPUs so it never shares a step with decode (Module 21).

## Deep dive: later chunks cost more

A chunk of ⟦m606⟧ tokens starting at offset ⟦m607⟧ does the usual linear work, plus attention against everything before it:

⟦M604⟧

The linear part is the same for every chunk. The attention part grows with the offset, so the last chunk of a long prompt is much heavier than the first.

~~~dg-raw
%%chart Step time of each 2,048-token chunk of a 32k prompt, Llama 3 8B at 600 TFLOP/s achieved.
x = 1, 4, 8, 12, 16
xlab = Chunk number
ylab = ms
Linear layers: 51, 51, 51, 51, 51
Attention: 1.8, 12.8, 27.5, 42.2, 56.9
Total: 53, 64, 79, 93, 108
%%
~~~

A token budget fixes tokens per step, not time per step. At long contexts, the 16th chunk takes twice as long as the first, and the ITL of everyone sharing those steps doubles too. Some schedulers therefore shrink the chunk as the offset grows, aiming at a constant FLOP budget:

⟦M605⟧

<div class="take">
<div class="lab">Key takeaway</div>

chunked prefill splits long prompts across steps so decodes never stall. Size the chunk from your ITL target and the measured cost per prefill token, then accept the longer TTFT for long prompts.

</div>
