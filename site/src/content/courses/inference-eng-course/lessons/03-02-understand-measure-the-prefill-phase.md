---
id: "03.02"
course: "inference-eng-course"
module: "03"
slug: "understand-measure-the-prefill-phase"
title: "Understand & Measure the Prefill Phase"
keyIdeas: ["Prefill processes the whole prompt in one forward pass and writes its KV cache.", "Prefill time grows with prompt length: linear for matmuls, quadratic for attention.", "Prefill is compute bound: many tokens per weight read.", "Prefill sets time to first token (TTFT), plus any queueing.", "Measure TTFT across prompt lengths to see the curve."]
question: "Why is prefill usually compute bound while decode is memory bound? Use arithmetic intensity in your answer."
answer: "In prefill, a matmul processes thousands of prompt tokens against each weight read, so arithmetic intensity (FLOPs per byte) is high and tensor cores stay busy; time is set by FLOPs. In decode each weight is used for only one token per sequence, so intensity is low and time is set by how fast weights and KV cache can be read."
sources: [{"title": "Horace He: Making Deep Learning Go Brrrr", "url": "https://horace.io/brrr_intro.html", "note": "Compute bound vs memory bound explained."}, {"title": "Splitwise: Efficient generative LLM inference using phase splitting", "url": "https://arxiv.org/abs/2311.18677", "note": "Measured differences between prefill and decode."}]
tags: ["prefill"]
---

## What prefill does

When a request arrives, the engine runs the entire prompt through the model in one forward pass. At every layer it computes Q, K and V for all prompt tokens, writes K and V into the cache, computes causal attention among the prompt tokens, and runs the MLP on all of them. At the end it needs logits only for the last position, which it samples to produce the first output token.

Prefill sets the time to first token. For long prompts, such as retrieval-augmented chat, document summarization or coding agents with large contexts, it is the largest part of the user's wait.

## Why prefill is compute bound

In prefill, each linear layer multiplies a [P, d] matrix of activations by a [d, d_out] weight matrix. Each weight loaded from memory is used P times, once per prompt token.

Arithmetic intensity is the number of FLOPs done per byte moved. For a matmul with P rows on BF16 weights:

- FLOPs: 2 × P × d × d_out
- Weight bytes: 2 × d × d_out
- Intensity: about P FLOPs per byte (ignoring activation bytes)

An H100 delivers about 990 TFLOP/s of dense BF16 and 3.35 TB/s of bandwidth, so it needs about 295 FLOPs per byte before compute becomes the limit. A prompt of a few hundred tokens or more crosses that line. From there on, the tensor cores set the speed.

~~~dg-raw
%%widget roof In Prefill mode, the x-axis is prompt tokens per pass. A few hundred tokens are enough to reach the compute roof.
mode = Prefill
n = 9
%%
~~~

## How prefill time scales

Two terms make up prefill cost:

- **Linear layers**: about 2 × parameters × P FLOPs. This grows linearly with prompt length.
- **Attention**: about 4 × layers × d × P² ÷ 2 FLOPs with causal masking (for the score and value products). This grows with P².

For an 8B model, the linear term is about 16 GFLOP per token, and the attention term per token is about 4 × 32 × 4,096 × P ÷ 2 ≈ 0.26 MFLOP × P. They become equal around P ≈ 60,000 tokens. Below about 8k tokens, prefill is close to linear. Beyond 32k, the quadratic term takes over.

<div class="ex">
<div class="lab">Worked example</div>

**estimating TTFT from FLOPs**

A 4,000-token prompt to Llama 3 8B on one H100.

Linear FLOPs: 16 GFLOP × 4,000 = 64 TFLOP. Attention FLOPs: about 0.26 MFLOP × 4,000 × 4,000 ≈ 4.2 TFLOP. Total about 68 TFLOP.

At a realistic 60% of peak (about 600 TFLOP/s), prefill takes about 115 ms.

Add tokenization, scheduling and the first sampling step, and a TTFT of about 130 to 150 ms on an idle server is what to expect.

</div>

## Measuring prefill

Send single requests to an idle server with `max_tokens = 1` and prompts of increasing length. TTFT is then almost entirely prefill.

| Prompt tokens | Llama 3 8B on one H100 (rough) |
| --- | --- |
| 512 | 20 to 40 ms |
| 2,048 | 60 to 120 ms |
| 8,192 | 250 to 500 ms |
| 32,768 | 1.5 to 3 s |

Treat these as orders of magnitude. Your engine, kernels and settings move them. The shape is what you should check: roughly linear, then curving upward as attention takes over.

## What prefill does to everyone else

Prefill is not free for other users. While one long prompt is processed, the GPU's compute is fully occupied. If the engine runs that prefill as one block, every decoding request on the same GPU stalls until it finishes. A 32k-token prefill can pause other users' streams for two seconds.

Engines solve this with **chunked prefill** (Module 10): split long prompts into chunks of a few thousand tokens and mix each chunk with the ongoing decode steps. Prefill takes a little longer, but decodes keep flowing.

## Ways to make prefill cheaper

- **Prefix caching**: if the start of the prompt was seen before (a shared system prompt, an earlier conversation turn), reuse its KV cache and skip that part of prefill.
- **FP8 compute**: on Hopper and newer GPUs, FP8 matmuls roughly double peak FLOPs.
- **Better attention kernels**: FlashAttention keeps the quadratic part efficient.
- **Tensor parallelism**: split the work across GPUs to cut TTFT for very long prompts.
- **Disaggregation**: run prefill on separate GPUs from decode, so neither phase disturbs the other (Module 21).

## Deep dive: prefill FLOPs, exactly

For a prompt of ⟦m297⟧ tokens through ⟦m298⟧ layers of width ⟦m299⟧ with ⟦m300⟧ non-embedding parameters:

⟦M295⟧

The attention term is ⟦m301⟧ for ⟦m302⟧ and ⟦m303⟧ over the full square, halved by the causal mask. Its share of the total is

⟦M296⟧

For Llama 3 8B, ⟦m304⟧. Attention is half the work when ⟦m305⟧ tokens.

~~~dg-raw
%%chart Prefill time for Llama 3 8B on one H100 at 50% of peak, split into linear and attention work.
x = 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000
logx = 1
logy = 1
xlab = Prompt tokens
ylab = Seconds
Linear layers: 0.03, 0.06, 0.12, 0.24, 0.49, 0.97, 1.94, 3.88
Attention: 0.0005, 0.002, 0.008, 0.034, 0.136, 0.54, 2.17, 8.69
Total: 0.031, 0.062, 0.128, 0.274, 0.626, 1.51, 4.11, 12.57
%%
~~~

## Deep dive: why prefill is compute bound

Prefill multiplies a ⟦m306⟧ activation by each weight. Its arithmetic intensity is roughly ⟦m307⟧ FLOPs per byte (00.01). With an H100's ridge at about 295, any prompt longer than a few hundred tokens is compute bound. That is why prefill time is predicted from FLOPs and decode time from bytes.

<div class="ex">
<div class="lab">Worked example</div>

**TTFT from first principles**

A 4,000-token prompt on Llama 3 8B: ⟦m308⟧ FLOPs.

At 50% of 989 TFLOP/s: 130 ms. Add 10 to 20 ms of fixed overhead: TTFT is about 145 ms with an empty queue.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

prefill processes the whole prompt in one compute-bound pass and sets TTFT. Its cost grows linearly with length until attention's quadratic term takes over, and long prefills stall other users unless chunked.

</div>
