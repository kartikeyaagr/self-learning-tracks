---
id: "00.01"
course: "inference-eng-course"
module: "00"
slug: "math-for-memory-throughput-speedups"
title: "Math for Memory, Throughput & Speedups"
keyIdeas: ["Memory for weights = parameters x bytes per parameter (FP32 4, BF16 2, FP8 1, INT4 0.5).", "A forward pass costs about 2 x parameters FLOPs per token.", "Decode at small batch is limited by bandwidth: tokens/s is roughly bandwidth divided by bytes read per step.", "Speedup claims should use the same baseline, same workload and same metric. Amdahl's law caps gains from optimizing one part.", "Keep units straight: GB vs GiB, tokens per second per request vs per GPU."]
question: "A 70B-parameter model is stored in BF16. How much memory do the weights need, and what is the decode speed ceiling on a GPU with 3.35 TB/s of bandwidth at batch size 1?"
answer: "70 billion parameters times 2 bytes is 140 GB, so it does not fit on one 80 GB GPU. If it did, each decode step must read all weights once, so the ceiling is about 3.35 TB/s divided by 140 GB, roughly 24 tokens per second at batch 1, ignoring KV reads and overheads."
sources: [{"title": "kipply: Transformer Inference Arithmetic", "url": "https://kipp.ly/transformer-inference-arithmetic/", "note": "Worked examples of FLOP and byte counts."}, {"title": "Horace He: Making Deep Learning Go Brrrr", "url": "https://horace.io/brrr_intro.html", "note": "Compute, memory and overhead as the three costs."}]
tags: ["math"]
---

## Three numbers you will use every day

Almost every inference estimate comes from three quantities:

- **Bytes**: how much data has to move or be stored. Weights, KV cache, activations.
- **FLOPs**: how many floating-point operations the math needs.
- **Rates**: how fast the hardware can move bytes (memory bandwidth, in TB/s) and do FLOPs (in TFLOP/s).

Divide bytes by bandwidth and FLOPs by compute rate. Whichever time is larger is your lower bound. You will be surprised how often this five-second calculation predicts a real benchmark within 30%.

## Bytes for weights

Weight memory = parameters × bytes per parameter.

| Precision | Bytes per parameter | 8B model | 70B model |
| --- | --- | --- | --- |
| FP32 | 4 | 32 GB | 280 GB |
| BF16 or FP16 | 2 | 16 GB | 140 GB |
| FP8 or INT8 | 1 | 8 GB | 70 GB |
| INT4 (plus scales) | about 0.55 | about 4.4 GB | about 39 GB |

A 70B model in BF16 does not fit on one 80 GB GPU. In FP8 it fits, with about 10 GB left for everything else, which is not much.

## FLOPs for a forward pass

For a dense transformer, one token through the network costs about **2 × parameters** FLOPs: each weight is multiplied once and added once. An 8B model needs about 16 GFLOP per token. Attention adds a term that grows with context length, which matters past a few thousand tokens.

## The decode ceiling

During decode with one user, each step produces one token and must read every weight once. So:

decode tokens/s ≤ memory bandwidth ÷ weight bytes

<div class="ex">
<div class="lab">Worked example</div>

**Llama 3 8B on an H100 at batch 1**

Weights: 8B × 2 bytes = 16 GB. Bandwidth: 3.35 TB/s.

Ceiling: 3,350 GB/s ÷ 16 GB ≈ 209 tokens/s. Real engines reach 150 to 180, about 75 to 85% of the ceiling.

Compute needed: 16 GFLOP per token × 200 tokens/s = 3.2 TFLOP/s, about 0.3% of the GPU's 990 TFLOP/s. The GPU is almost idle on compute; it is waiting on memory.

</div>

That last line is the most important fact about LLM decoding: at small batch sizes it is **memory-bound**. Batching fixes it, because one weight read serves every sequence in the batch.

~~~dg-raw
%%widget roof Decode at batch 1 sits far left of the ridge. Raise the batch and watch it climb toward the compute roof.
mode = Decode
n = 0
%%
~~~

## Speedups and Amdahl's law

When you optimize one part of a system, the total speedup is limited by the part you did not touch. If attention is 30% of step time and you make it 2× faster:

new time = 0.7 + 0.3 ÷ 2 = 0.85, so the speedup is 1 ÷ 0.85 ≈ 1.18×, not 2×.

This is Amdahl's law. Before you spend a week on a kernel, measure what fraction of time it takes. A 10× faster kernel for something that takes 5% of the step is a 4.7% end-to-end gain.

## Little's law

For any stable system: **requests in flight = arrival rate × time in system**. If 20 requests per second arrive and each takes 3 seconds, you have 60 requests in flight on average. You need enough KV memory and batch slots for 60, plus headroom for spikes. This one equation connects a traffic forecast to a hardware size.

## Units that trip people up

- GB versus GiB: 1 GB = 10⁹ bytes, 1 GiB = 2³⁰ ≈ 1.074 × 10⁹. GPU spec sheets mix them. An "80 GB" H100 has 80 GiB-ish usable, minus driver and context overhead.
- Spec-sheet FLOPs are often quoted **with sparsity**, which doubles the number. The H100 SXM's dense BF16 number is about 990 TFLOP/s; the sparse number is about 1,980.
- Bandwidth you actually achieve is usually 80 to 90% of the spec.

## Deep dive: arithmetic intensity, derived

Arithmetic intensity (AI) is FLOPs done per byte moved from memory. For a matrix multiply of an activation ⟦m31⟧ by a weight ⟦m32⟧ in BF16 (2 bytes per value):

⟦M25⟧

⟦M26⟧

So for decode, intensity is roughly the batch size. A GPU is compute bound only when AI exceeds its ridge point:

⟦M27⟧

Attainable performance follows the roofline:

⟦M28⟧

~~~dg-raw
%%chart The H100 roofline (BF16 dense). Decode at batch 1 sits at AI ≈ 1, prefill of a long prompt far to the right.
x = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
logx = 1
logy = 1
xlab = Arithmetic intensity (FLOP per byte)
ylab = Attainable TFLOP/s
Roofline: 3.35, 6.7, 13.4, 26.8, 53.6, 107, 214, 429, 858, 989, 989, 989
%%
~~~

## Deep dive: Amdahl's law, derived

Split a step's time into a part ⟦m33⟧ you speed up by a factor ⟦m34⟧ and a part ⟦m35⟧ you do not touch. New time is ⟦m36⟧ of the old, so

⟦M29⟧

The limit is the important part: if attention is 30% of a step, an infinitely fast attention kernel gives at most ⟦m37⟧.

~~~dg-raw
%%chart Amdahl: overall speedup against the speedup of the part you optimized.
x = 1, 2, 4, 8, 16, 32, 64
logx = 1
xlab = Speedup of the optimized part (s)
ylab = Overall speedup
p = 0.9: 1, 1.82, 3.08, 4.71, 6.4, 7.8, 8.77
p = 0.5: 1, 1.33, 1.6, 1.78, 1.88, 1.94, 1.97
p = 0.3: 1, 1.18, 1.29, 1.36, 1.39, 1.41, 1.42
%%
~~~

## Deep dive: Little's law, derived

Over a long window of length ⟦m38⟧, let ⟦m39⟧ requests arrive and each spend on average ⟦m40⟧ seconds in the system. The total request-seconds in the window is ⟦m41⟧. Dividing by ⟦m42⟧ gives the average number in the system ⟦m43⟧, and ⟦m44⟧ is the arrival rate ⟦m45⟧:

⟦M30⟧

It holds for any arrival pattern and any scheduler, as long as the system is stable. In inference it converts between three things you measure separately: requests per second, latency, and concurrency (which sets KV memory).

<div class="ex">
<div class="lab">Worked example</div>

**from traffic to KV memory**

12 requests/s, each lasting 9 s end to end on average. By Little's law, ⟦m46⟧ requests are in flight.

If each holds on average 1,800 tokens of context at 128 KB per token, KV in use is ⟦m47⟧.

Halving latency to 4.5 s, with the same traffic, halves the KV you need. Faster serving is also a memory saving.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

weights ÷ bandwidth gives the batch-1 decode ceiling, 2 × parameters gives FLOPs per token, and Amdahl and Little tell you which optimizations and how much hardware are worth it.

</div>
