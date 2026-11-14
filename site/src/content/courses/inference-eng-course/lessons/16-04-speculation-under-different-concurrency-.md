---
id: "16.04"
course: "inference-eng-course"
module: "16"
slug: "speculation-under-different-concurrency-levels"
title: "Speculation Under Different Concurrency Levels"
keyIdeas: ["Speculation spends spare compute to cut latency.", "At low concurrency there is spare compute: large gains.", "At high concurrency compute is busy: rejected tokens waste capacity.", "Throughput can fall at high load even when latency per token improves.", "Adaptive policies reduce or disable speculation as batch size grows."]
question: "Why does speculative decoding give big speedups at batch 1 and can slow things down at batch 128?"
answer: "At batch 1 decode is memory bound and the GPU has lots of idle compute, so verifying k extra tokens per sequence is almost free. At batch 128 decode is close to compute bound; each verified token costs real compute, and rejected draft tokens are wasted work that could have served other users. Throughput can drop even if per-request latency improves slightly."
sources: [{"title": "vLLM: Speculative decoding", "url": "https://docs.vllm.ai/en/latest/features/spec_decode.html", "note": "Configuration and caveats."}, {"title": "Optimizing Speculative Decoding for Serving Large Language Models Using Goodput", "url": "https://arxiv.org/abs/2406.14066", "note": "Adaptive speculation under load."}]
tags: ["speculative", "batching"]
---

## The trade

Speculative decoding spends extra compute, on drafting and on verifying tokens that may be rejected, to reduce the number of sequential target passes. That is a good trade when compute is idle and a bad one when it is not.

| Concurrency | Spare compute | Effect of speculation |
| --- | --- | --- |
| 1 to 4 | a lot | large TPOT gain |
| 16 to 32 | some | moderate gain |
| 128 and more | little | small gain, or a loss |

## Why speculation can lose

At large batch, the decode step is close to compute bound. Every extra position in the verification pass now costs real time, shared by every sequence in the batch.

With acceptance 0.7 and k = 4, the expected tokens per round is about 2.8. But the verification pass processes 5 positions per sequence. About 2.2 of those 5 produce nothing useful. When compute is the limit, that wasted work displaces tokens other users could have received.

<div class="ex">
<div class="lab">Worked example</div>

**the same speculation at two loads**

Batch 1: a normal step takes 6 ms. A speculative round (k = 4, cheap n-gram drafts, 5 positions verified) takes 6.4 ms and yields 2.8 tokens on average. Time per token: 2.3 ms, 2.6 times faster.

Batch 128: a normal step takes 20 ms for 128 tokens. A speculative step verifies 640 positions and, now compute bound, takes about 60 ms, yielding 128 × 2.8 = 358 tokens. Throughput: 6,000 tokens per second, against 6,400 without speculation.

At batch 128, speculation still cuts TPOT for each user (60 ÷ 2.8 ≈ 21 ms against 20 ms is about even here), but total throughput falls 6%, so the GPU serves fewer users within the SLO.

</div>

## Measuring the crossover

Sweep concurrency with speculation on and off, using the same workload. Plot TPOT and throughput against concurrency for both.

~~~dg-flow The typical outcome of the sweep.
Low load: speculation wins on TPOT -> *Crossover at moderate load -> High load: no speculation wins on throughput
~~~

The crossover point depends on:

- **The GPU's ridge point.** GPUs with more compute per byte of bandwidth leave more room for speculation.
- **The model size.** Larger models are more memory bound at a given batch, so speculation helps up to higher concurrency.
- **The acceptance rate.** Higher acceptance moves the crossover to higher load.
- **The draft cost.** Free drafts such as n-gram lookup move it further than a draft model does.

## Adaptive policies

Production engines avoid a fixed on or off setting:

- **Batch threshold.** Disable speculation when the running batch exceeds a threshold found from the sweep.
- **Dynamic k.** Choose k each step to maximize expected goodput given the current batch size and recent acceptance. As the batch grows, k shrinks toward 0.
- **Per-tenant policy.** Enable speculation only for latency-sensitive requests, such as interactive chat or voice, and not for batch jobs.

~~~dg-raw
%%widget batch Move the batch size toward 256 and watch the per-token cost rise. That rising cost is what speculation's extra positions have to pay.
model = Llama 3 8B
gpu = H100 SXM (80 GB)
ctx = 11
b = 3
%%
~~~

## Speculation and the SLO

Speculation helps TPOT most at low load, which is when TPOT is already lowest. Its value depends on what limits your capacity:

- If the TPOT SLO binds at the expected load, and the expected load is moderate, speculation can raise SLO-safe capacity.
- If throughput at high load limits capacity, speculation may lower it.
- For latency-critical products with low concurrency per GPU, such as code completion or voice, it is often the single biggest TPOT improvement available.

## Deep dive: speed-up as a function of batch size

Combining 16.01 and 16.02, with drafting costing ⟦m847⟧:

⟦M846⟧

~~~dg-raw
%%chart Predicted per-user speed-up against batch size (α = 0.8, k = 4, c = 0.05).
x = 1, 16, 32, 64, 128, 256
logx = 1
xlab = Batch size
ylab = Speed-up
Speculation: 2.8, 2.8, 2.8, 2.19, 1.24, 0.72
Break-even*: 1, 1, 1, 1, 1, 1
%%
~~~

Choosing the best ⟦m848⟧ for each batch size and acceptance rate gives a policy table:

| Acceptance α | Batch 1 | Batch 64 | Batch 128 | Batch 256 |
| --- | --- | --- | --- | --- |
| 0.6 | k = 4, 1.92× | k = 2, 1.78× | k = 1, 1.40× | off |
| 0.7 | k = 6, 2.35× | k = 3, 2.03× | k = 1, 1.48× | off |
| 0.8 | k = 8, 3.09× | k = 3, 2.37× | k = 1, 1.57× | off |
| 0.9 | k = 13, 4.67× | k = 3, 2.76× | k = 1, 1.66× | off |

The model ignores KV reads, which make verification relatively cheaper at long contexts. Treat the table as a starting point for the measured sweep in 16.08.

<div class="take">
<div class="lab">Key takeaway</div>

speculation trades spare compute for lower latency. It shines at low load and must back off at high load, so measure the crossover on your hardware and use a load-aware policy.

</div>
