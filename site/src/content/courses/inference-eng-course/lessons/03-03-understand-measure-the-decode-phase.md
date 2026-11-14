---
id: "03.03"
course: "inference-eng-course"
module: "03"
slug: "understand-measure-the-decode-phase"
title: "Understand & Measure the Decode Phase"
keyIdeas: ["Decode generates one token per sequence per step, using the KV cache.", "Each step reads all weights plus every active sequence's KV cache.", "At small batch, decode is memory bound: step time is almost flat as batch grows.", "Batching is the main lever for decode throughput, until compute or KV memory runs out.", "Decode speed per user is measured as time per output token (TPOT) or inter-token latency (ITL)."]
question: "Decode at batch 1 runs at 40 ms per token. At batch 32 it runs at 48 ms per token. Explain why throughput rose so much."
answer: "Decode is memory bound: each step reads all weights once regardless of batch size, and that read dominates time. At batch 32 the same weight read serves 32 sequences, so step time rises only slightly (more KV reads and some compute) while tokens per step rise 32 times. Throughput goes from 25 to about 670 tokens per second."
sources: [{"title": "Orca: A Distributed Serving System for Transformer-Based Generative Models", "url": "https://www.usenix.org/conference/osdi22/presentation/yu", "note": "Iteration-level scheduling for decode."}, {"title": "Lilian Weng: Large Transformer Model Inference Optimization", "url": "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/", "note": "Overview of decode costs."}]
tags: ["decode"]
---

## What decode does

After prefill, the engine enters a loop. At each step, every active sequence contributes exactly one token, its most recent one. The model runs on those tokens, appends one new K and V vector per layer to each sequence's cache, and produces logits for each sequence's next token.

Decode sets the time per output token, the speed at which text streams in. For chat, coding and agents with long outputs, it is most of the GPU time.

## Why decode is memory bound

With a batch of B sequences, every linear layer multiplies a [B, d] matrix by a weight matrix. Each weight loaded from memory is used B times. At B = 1, each 2-byte weight supports 2 FLOPs, an intensity of 1 FLOP per byte. The H100 needs about 295 to be compute bound. So the GPU spends almost all its time waiting for memory, and its tensor cores sit mostly idle.

The step time at small batch is therefore close to:

step time ≈ (weight bytes + KV bytes read) ÷ memory bandwidth

~~~dg-flow What one decode step spends its time on.
Read all weights -> Read KV for each sequence -> *Small matmuls -> One token per sequence
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the batch-1 ceiling**

Llama 3 8B in BF16 has 16 GB of weights. An H100 reads 3.35 TB/s at best and about 3.0 TB/s in practice.

Step time: 16 GB ÷ 3.0 TB/s ≈ 5.3 ms, plus a small KV read. Ceiling: about 190 tokens per second for one user.

The math needs 16 GFLOP per step. At 5.3 ms, that is 3 TFLOP/s, about 0.3% of the GPU's peak.

</div>

## Batching is the main lever

Adding sequences to the batch adds work but does not add weight reads. The weights are read once per step and shared by every sequence. So throughput rises almost linearly with batch size at first.

Three things eventually stop the climb:

- **KV reads.** Each sequence's cache must be read every step. At large batch and long context, KV bytes can exceed weight bytes.
- **Compute.** At a batch of a few hundred, the matmuls approach the compute roof.
- **Memory capacity.** Every sequence's cache must fit in the KV pool.

~~~dg-raw
%%widget batch Raise the batch and watch throughput climb while each user's speed falls. Then raise the context and see KV reads take over.
model = Llama 3 8B
gpu = H100 SXM (80 GB)
ctx = 11
b = 4
%%
~~~

| Batch | Step time | Total tokens/s | Tokens/s per user |
| --- | --- | --- | --- |
| 1 | 6 ms | 170 | 170 |
| 16 | 7 ms | 2,300 | 145 |
| 64 | 10 ms | 6,400 | 100 |
| 256 | 25 ms | 10,200 | 40 |

These numbers are illustrative for an 8B model at about 2k tokens of context on an H100. The pattern is what matters: total throughput keeps rising, but each user's speed falls. Somewhere on this curve you hit your latency target, and that is where you should run.

## Decode attention at long context

The KV read grows with every token. For one Llama 3 8B sequence:

| Context | KV cache | KV read time per step on H100 |
| --- | --- | --- |
| 2k | 256 MB | about 0.08 ms |
| 32k | 4 GB | about 1.3 ms |
| 128k | 16 GB | about 5.3 ms |

At 128k tokens, reading one sequence's cache takes as long as reading all the weights. A batch of 16 such sequences would take about 90 ms per step for KV reads alone. That is why long-context serving needs smaller caches (GQA, MLA, FP8 KV), sparse attention, or many GPUs.

## Measuring decode

Use short prompts (so prefill is negligible) and a long, fixed output length (`max_tokens = 512` with `ignore_eos`). Vary concurrency from 1 to 256. For each level, record:

- TPOT at p50 and p95, measured from token arrival times.
- Total output tokens per second across all requests.

Plot both against concurrency on one chart with two y-axes. The crossing point between "throughput still rising fast" and "TPOT rising fast" is the efficient region for this model and GPU.

## Deep dive: TPOT from bytes

At batch ⟦m311⟧ and average context ⟦m312⟧, one decode step reads all weights once and every sequence's KV once:

⟦M309⟧

where ⟦m313⟧ is KV bytes per token (128 KB for Llama 3 8B in BF16). TPOT for each user is ⟦m314⟧, and total throughput is ⟦m315⟧.

~~~dg-raw
%%bars Bytes read per decode step for Llama 3 8B (GB): weights stay fixed, KV grows with batch × context.
unit = GB
stack = Weights, KV
Batch 1, 2k context | 16+0.27
Batch 16, 2k context | 16+4.3
Batch 64, 2k context | 16+17.2
Batch 64, 8k context | 16+68.7 | | would not fit on one 80 GB GPU
%%
~~~

The crossover where KV bytes equal weight bytes is at ⟦m316⟧ tokens in flight. Above it, KV, not weights, dominates each step, and a smaller KV (GQA, FP8 KV) matters more than smaller weights.

<div class="ex">
<div class="lab">Worked example</div>

**predicting throughput at batch 64**

2k average context: bytes per step ⟦m317⟧ GB. At 2.8 TB/s: 11.9 ms, plus about 1 ms overhead: about 13 ms.

Throughput: ⟦m318⟧ tokens/s. Each user sees about 77 tokens/s.

With FP8 KV the KV term halves: 24.6 GB, 9.8 ms, about 6,500 tokens/s. A 33% gain from KV precision alone.

</div>

## Deep dive: when decode stops being memory bound

Compute per step is ⟦m319⟧ FLOPs plus attention. It reaches the memory time when

⟦M310⟧

With long contexts the KV term grows with ⟦m320⟧ too, so decode can stay memory bound at any batch size. In practice, short-context decode on an H100 becomes compute bound only around batch 200 to 300, and KV capacity usually runs out first.

<div class="take">
<div class="lab">Key takeaway</div>

decode is memory bound, so batching more sequences is nearly free at first. The limits are each user's latency, the KV reads that grow with context, and the KV pool's capacity.

</div>
